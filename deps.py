# -*- coding: utf-8 -*-
"""
deps.py
Detect and (optionally) install the optional third-party Python packages
used by the Spatial Predictive Modeling tab (scikit-learn, pykrige).

DESIGN NOTES -- why this module is written the way it is
--------------------------------------------------------
Earlier versions of this module installed packages straight into QGIS's own
Python environment with `pip install --upgrade`, then immediately imported
them into the running process. Both of those things can hard-crash QGIS, so
neither is done any more:

1. NEVER upgrade QGIS's own numpy/scipy.
   scikit-learn depends on numpy and scipy. `pip install --upgrade
   scikit-learn` therefore also upgrades numpy and scipy. QGIS has numpy
   already loaded, and GDAL / processing / matplotlib are compiled against
   that exact numpy ABI. On Windows the DLLs are file-locked, so the upgrade
   half-fails and leaves a broken install; on Linux it succeeds and breaks
   the ABI. Either way QGIS crashes later.

   Instead everything is installed into a private folder inside the plugin
   (`<plugin>/libs`) using `pip --target`, with a constraints file that pins
   numpy/scipy/joblib/threadpoolctl to the versions QGIS is *already*
   running. QGIS's environment is never modified, so no admin rights, no
   --user fallback, and no ABI risk.

2. NEVER import a package to find out whether it is installed.
   Importing freshly installed compiled extensions into a process that has
   already loaded a different build of numpy segfaults. `importlib.reload()`
   on a C-extension package is unsafe for the same reason. Detection here is
   done purely from installed distribution *metadata* on disk, which cannot
   crash the process.

3. A newly installed package is only usable after QGIS restarts.
   That is stated plainly to the user rather than papered over with
   `site.addsitedir()` tricks that work on some machines and not others.

4. `<plugin>/libs` is APPENDED to sys.path, never prepended, so that if a
   dependency does end up duplicated there, QGIS's own copy still wins.
"""

import sys
import os
import re
import glob
import shutil
import tempfile
import subprocess
import importlib.metadata as importlib_metadata

# name shown to the user -> (pip package name, module to try importing)
PACKAGES = {
    "scikit-learn": ("scikit-learn", "sklearn"),
    "pykrige": ("pykrige", "pykrige"),
}

# Dependencies that QGIS ships itself. If they are already present we pin
# them to the exact running version so pip resolves a compatible
# scikit-learn/pykrige instead of pulling a newer numpy in beside them.
SHARED_DEPS = ("numpy", "scipy", "joblib", "threadpoolctl")

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
LIBS_DIR = os.path.join(PLUGIN_DIR, "libs")


# ---------------------------------------------------------------------------
# sys.path handling
# ---------------------------------------------------------------------------

def ensure_libs_on_path():
    """Append the plugin-private libs folder to sys.path (idempotent).

    Appended, not inserted at position 0, so QGIS's own numpy/scipy always
    take precedence over anything that may have landed in libs/.
    """
    if os.path.isdir(LIBS_DIR) and LIBS_DIR not in sys.path:
        sys.path.append(LIBS_DIR)
    return LIBS_DIR


# Run on import so `from . import deps` is enough to activate the folder.
ensure_libs_on_path()


# ---------------------------------------------------------------------------
# Interpreter resolution
# ---------------------------------------------------------------------------

_resolved_python_cache = None


def _is_python_named(path):
    """True if the file *name* looks like a real Python interpreter.

    This check exists specifically so we never execute `qgis-bin.exe -m pip`.
    On Windows sys.executable is qgis-bin.exe, and running it with arguments
    launches a SECOND QGIS instance, which then fights the running one for
    the profile lock before we time out and kill it.
    """
    if not path:
        return False
    name = os.path.basename(path).lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return bool(re.fullmatch(r"python(\d+(\.\d+)?)?w?", name))


def _subprocess_kwargs():
    """Keep pip from flashing a console window on Windows."""
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return kwargs


def _looks_usable(python_path):
    if not _is_python_named(python_path):
        return False
    if not os.path.isfile(python_path):
        return False
    try:
        result = subprocess.run(
            [python_path, "-m", "pip", "--version"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=30, **_subprocess_kwargs()
        )
        return result.returncode == 0
    except Exception:
        return False


def resolve_python_executable():
    """Best-guess Python interpreter path that supports `-m pip`.

    Cached for the lifetime of the QGIS session.
    """
    global _resolved_python_cache
    if _resolved_python_cache:
        return _resolved_python_cache

    candidates = []
    if _is_python_named(sys.executable):
        candidates.append(sys.executable)

    base_dir = os.path.dirname(sys.executable or "")
    if base_dir:
        # Typical Windows layout: .../QGIS 3.xx/bin/qgis-bin.exe with the
        # real interpreter at .../QGIS 3.xx/apps/PythonXY/python.exe
        for name in ("python3.exe", "python.exe", "python3", "python"):
            candidates.append(os.path.join(base_dir, name))
        parent = os.path.dirname(base_dir)
        for apps_dir in (os.path.join(parent, "apps"), parent):
            if not os.path.isdir(apps_dir):
                continue
            try:
                for entry in sorted(os.listdir(apps_dir)):
                    if entry.lower().startswith("python"):
                        for name in ("python3.exe", "python.exe", "python3", "python"):
                            candidates.append(os.path.join(apps_dir, entry, name))
                            candidates.append(os.path.join(apps_dir, entry, "bin", name))
            except OSError:
                pass

    # sysconfig knows where the interpreter for THIS Python lives, even when
    # sys.executable has been overwritten by the host application.
    try:
        import sysconfig
        for key in ("BINDIR", "installed_base", "base"):
            d = sysconfig.get_config_var(key) or sysconfig.get_path("scripts")
            if not d:
                continue
            for name in ("python3.exe", "python.exe", "python3", "python"):
                candidates.append(os.path.join(d, name))
    except Exception:
        pass

    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if _looks_usable(c):
            _resolved_python_cache = c
            return c

    # Fall back to sys.executable so error messages show something concrete.
    _resolved_python_cache = sys.executable
    return sys.executable


def python_executable():
    return resolve_python_executable()


# ---------------------------------------------------------------------------
# Detection (metadata only -- never imports, so it can never crash QGIS)
# ---------------------------------------------------------------------------

def _normalize(name):
    return re.sub(r"[-_.]+", "-", (name or "")).lower()


def _find_distribution_version(dist_name):
    """Return the installed version string for a distribution, or None.

    Uses installed metadata only. Handles the fact that some QGIS builds ship
    an older importlib.metadata that does not normalize distribution names
    (so `version("pykrige")` can miss a dist recorded as "PyKrige").
    """
    if not dist_name:
        return None
    try:
        return importlib_metadata.version(dist_name)
    except Exception:
        pass

    target = _normalize(dist_name)
    try:
        for dist in importlib_metadata.distributions():
            try:
                if _normalize(dist.metadata["Name"]) == target:
                    return dist.version
            except Exception:
                continue
    except Exception:
        pass
    return None


def _module_present_on_disk(module_name):
    """Filesystem fallback: is there a package/module of this name on sys.path?

    Deliberately does not import anything -- importlib.util.find_spec() can
    execute parent package __init__ files, so we just look for the files.
    """
    if not module_name:
        return False
    for entry in list(sys.path):
        if not entry or not os.path.isdir(entry):
            continue
        pkg_init = os.path.join(entry, module_name, "__init__.py")
        if os.path.isfile(pkg_init):
            return True
        if glob.glob(os.path.join(entry, module_name + ".py")):
            return True
        if glob.glob(os.path.join(entry, module_name + ".*.pyd")) or \
           glob.glob(os.path.join(entry, module_name + ".*.so")):
            return True
    return False


def check_package(module_name, dist_name=None):
    """Returns (installed: bool, version: str or None).

    Never imports the package. Safe to call from the GUI thread at any time.
    """
    ensure_libs_on_path()
    version = _find_distribution_version(dist_name) or _find_distribution_version(module_name)
    if version:
        return True, version
    if _module_present_on_disk(module_name):
        return True, "unknown"
    return False, None


def check_all():
    """Returns dict: display_name -> (installed, version, pip_name)."""
    out = {}
    for display_name, (pip_name, module_name) in PACKAGES.items():
        installed, version = check_package(module_name, dist_name=pip_name)
        out[display_name] = (installed, version, pip_name)
    return out


def import_is_live(module_name):
    """True if the module is already imported in THIS session.

    Used by the UI to tell the difference between "installed and usable now"
    and "installed on disk, needs a QGIS restart".
    """
    return module_name in sys.modules


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def _write_constraints(log=print):
    """Pin the dependencies QGIS already ships to their running versions.

    Returns a path to a constraints file, or None if nothing to pin.
    """
    pins = []
    for name in SHARED_DEPS:
        version = _find_distribution_version(name)
        if version:
            pins.append(f"{name}=={version}")
    if not pins:
        return None
    fd, path = tempfile.mkstemp(prefix="samkit_constraints_", suffix=".txt", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(pins) + "\n")
    log("Pinning QGIS's existing packages so they are never modified:")
    for pin in pins:
        log(f"    {pin}")
    return path


def _prune_duplicate_shared_deps(log=print):
    """Remove anything from libs/ that duplicates a package QGIS already has.

    The constraints file makes pip resolve the identical version, so the copy
    in libs/ is redundant. Deleting it keeps exactly one numpy/scipy on the
    path and removes any chance of a shadowed, ABI-mismatched second copy.
    """
    if not os.path.isdir(LIBS_DIR):
        return
    for name in SHARED_DEPS:
        env_version = None
        # Look it up outside libs/ -- i.e. QGIS's own copy.
        saved_path = list(sys.path)
        try:
            sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.abspath(LIBS_DIR)]
            env_version = _find_distribution_version(name)
        finally:
            sys.path = saved_path
        if not env_version:
            continue
        removed = False
        for pattern in (name, name.replace("-", "_")):
            for path in glob.glob(os.path.join(LIBS_DIR, pattern)) + \
                        glob.glob(os.path.join(LIBS_DIR, pattern + "-*.dist-info")) + \
                        glob.glob(os.path.join(LIBS_DIR, pattern + ".libs")) + \
                        glob.glob(os.path.join(LIBS_DIR, pattern + ".py")):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.remove(path)
                    removed = True
                except Exception:
                    pass
        if removed:
            log(f"Removed redundant copy of '{name}' from libs/ "
                f"(QGIS already provides {name} {env_version}).")


def pip_install(pip_names, log=print, extra_args=None, cancelled=None):
    """Install packages into the plugin-private libs/ folder.

    Streams pip output line-by-line to `log`. Returns True on success.

    `cancelled` may be a zero-argument callable returning True to abort; the
    pip subprocess is then terminated. This lets the UI run the install on a
    worker thread and still let the user stop it.

    QGIS's own site-packages are never written to, so this needs no admin
    rights and cannot corrupt the QGIS install.
    """
    if not pip_names:
        return True

    python_exe = resolve_python_executable()
    log(f"Target Python interpreter: {python_exe}")
    log(f"Install location (plugin-private, QGIS is not modified): {LIBS_DIR}")

    if not _is_python_named(python_exe):
        log("")
        log("Could not locate a usable python executable for this QGIS install.")
        log("Please use the standalone installer instead: open QGIS's Python "
            "Console (Plugins > Python Console) and run "
            "'install_dependencies.py' from the samkit plugin folder.")
        return False

    try:
        os.makedirs(LIBS_DIR, exist_ok=True)
    except OSError as e:
        log(f"Could not create {LIBS_DIR}: {e}")
        return False

    constraints = _write_constraints(log=log)

    cmd = [
        python_exe, "-m", "pip", "install",
        "--target", LIBS_DIR,
        "--upgrade",
        # Never build from source. A source build of scipy/scikit-learn needs a
        # C/Fortran toolchain and can run for 20+ minutes looking like a hang.
        "--only-binary=:all:",
        "--disable-pip-version-check",
        "--no-warn-script-location",
        "--no-input",
    ]
    if constraints:
        cmd += ["--constraint", constraints]
    cmd += list(extra_args or []) + list(pip_names)

    log("")
    log("Running: " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    log("")

    proc = None
    try:
        env = dict(os.environ)
        # Stop a stray PYTHONPATH/user-site from redirecting the install.
        env["PYTHONNOUSERSITE"] = "1"
        env.pop("PIP_TARGET", None)
        env.pop("PIP_USER", None)

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1, env=env, **_subprocess_kwargs()
        )
        for line in proc.stdout:
            if cancelled is not None and cancelled():
                log("Cancelled by user -- stopping pip …")
                try:
                    proc.terminate()
                except Exception:
                    pass
                return False
            log(line.rstrip())
        proc.wait()
        rc = proc.returncode
    except Exception as e:
        log(f"Failed to run pip: {e}")
        log("")
        log("You can install manually instead. Open a terminal (OSGeo4W Shell on "
            "Windows) and run:")
        log(f'  "{python_exe}" -m pip install --target "{LIBS_DIR}" '
            f'--only-binary=:all: {" ".join(pip_names)}')
        return False
    finally:
        if constraints:
            try:
                os.remove(constraints)
            except OSError:
                pass

    if rc == 0:
        _prune_duplicate_shared_deps(log=log)
        ensure_libs_on_path()
        log("")
        log("Install finished successfully.")
        log("IMPORTANT: please restart QGIS now. Python cannot safely load a "
            "newly installed compiled package into a session that is already "
            "running -- the package will be picked up automatically on the "
            "next start.")
        return True

    log("")
    log(f"pip exited with return code {rc}.")
    if rc is not None and rc != 0:
        log("Common causes: no internet access, a corporate proxy blocking "
            "pypi.org, or no prebuilt wheel available for this QGIS Python "
            "version.")
        log("If you are behind a proxy, try adding: --proxy "
            "http://user:pass@host:port")
    log("You can also install manually: open a terminal (OSGeo4W Shell on "
        "Windows) and run:")
    log(f'  "{python_exe}" -m pip install --target "{LIBS_DIR}" '
        f'--only-binary=:all: {" ".join(pip_names)}')
    return False


# ---------------------------------------------------------------------------
# WhiteboxTools binary (used by the Environmental Covariates tab)
# ---------------------------------------------------------------------------

def download_whitebox_binary(log=print, cancelled=None):
    """Fetch the WhiteboxTools executable the 'whitebox' package drives.

    The 'whitebox' wheel is only a ~70 KB pure-Python wrapper; the actual
    whitebox_tools binary is downloaded on first use by whitebox.download_wbt()
    (from whiteboxgeo.com, with a GitHub mirror as backup). We run that download
    in a SEPARATE Python subprocess -- never importing whitebox into the running
    QGIS process -- so it cannot disturb the live session, and so the binary
    lands inside the same whitebox package we just installed into libs/.

    Returns True on success. Safe to call after pip-installing 'whitebox'.
    """
    python_exe = resolve_python_executable()
    if not _is_python_named(python_exe):
        log("Could not locate a usable python executable to download WhiteboxTools.")
        return False

    ensure_libs_on_path()
    # libs/ goes first on the subprocess path so it imports the whitebox we just
    # installed (rather than an older copy elsewhere), and the WBT folder is
    # created next to that same package where _find_whitebox_tools() looks.
    program = (
        "import sys, os\n"
        "sys.path.insert(0, {libs!r})\n"
        "import whitebox\n"
        "whitebox.download_wbt(verbose=True)\n"
        "exe = 'whitebox_tools.exe' if os.name == 'nt' else 'whitebox_tools'\n"
        "p = os.path.join(os.path.dirname(whitebox.__file__), 'WBT', exe)\n"
        "print('WhiteboxTools binary: ' + (p if os.path.isfile(p) else 'NOT FOUND'))\n"
    ).format(libs=LIBS_DIR)

    cmd = [python_exe, "-c", program]
    log("")
    log("Downloading the WhiteboxTools executable (one-time, ~30-50 MB)…")

    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PIP_TARGET", None)
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1, env=env, **_subprocess_kwargs()
        )
        for line in proc.stdout:
            if cancelled is not None and cancelled():
                log("Cancelled by user -- stopping the WhiteboxTools download …")
                try:
                    proc.terminate()
                except Exception:
                    pass
                return False
            log(line.rstrip())
        proc.wait()
        rc = proc.returncode
    except Exception as e:
        log(f"Failed to download WhiteboxTools: {e}")
        return False

    if rc == 0:
        log("WhiteboxTools executable is ready.")
        return True
    log(f"WhiteboxTools download exited with code {rc}. Common causes: no "
        f"internet access or a proxy blocking whiteboxgeo.com / github.com.")
    return False


# ---------------------------------------------------------------------------
# Messaging
# ---------------------------------------------------------------------------

def missing_dependency_message(display_name):
    pip_name, module_name = PACKAGES[display_name]
    installed, version = check_package(module_name, dist_name=pip_name)

    if installed and not import_is_live(module_name):
        return (
            f"'{display_name}' (v{version}) is installed in SAMKit's private "
            f"library folder:\n  {LIBS_DIR}\n\n"
            f"but it could not be loaded into the QGIS session that is currently "
            f"running.\n\nPlease restart QGIS once. Packages installed while QGIS "
            f"is running are only usable after a restart."
        )

    python_exe = resolve_python_executable()
    return (
        f"'{display_name}' is not installed.\n\n"
        f"You can install it from the 'Python dependencies' panel at the top of "
        f"this tab. It is installed into SAMKit's own folder:\n  {LIBS_DIR}\n"
        f"so your QGIS installation is never modified and no admin rights are "
        f"needed.\n\n"
        f"To install it manually instead, run:\n"
        f'  "{python_exe}" -m pip install --target "{LIBS_DIR}" '
        f'--only-binary=:all: {pip_name}\n\n'
        f"After installing, restart QGIS once."
    )
