# -*- coding: utf-8 -*-
r"""
install_dependencies.py
Standalone installer for SAMKit's optional Python packages (scikit-learn,
pykrige), independent of the in-app installer in Tab 2. Use this if the
in-app "Install checked packages" button fails on a given machine.

WHAT IT DOES (and deliberately does not do)
--------------------------------------------
Packages are installed into a private folder inside the SAMKit plugin
(`<plugin>/libs`) using `pip --target`, with a constraints file pinning
numpy/scipy/joblib/threadpoolctl to the versions QGIS is already running.

  * QGIS's own site-packages are never written to -> no admin rights needed,
    nothing to break, no --user fallback that shadows QGIS's numpy later.
  * numpy and scipy are never upgraded. Upgrading them under a running QGIS
    breaks the ABI that GDAL/processing/matplotlib are compiled against and
    crashes QGIS.
  * `--only-binary=:all:` means pip will never try to compile scipy or
    scikit-learn from source, which needs a toolchain and can run for 20+
    minutes looking like a hang.

After it finishes, RESTART QGIS. Newly installed compiled packages cannot
safely be loaded into a Python session that is already running.

HOW TO RUN THIS (two ways) -- prefer method A, it's the most reliable:

A) Inside QGIS's own Python Console (recommended -- guarantees the correct
   environment no matter how QGIS was installed):
     1. In QGIS: Plugins menu -> Python Console (or Ctrl+Alt+P).
     2. Click the "Show Editor" icon, then "Open script..." and pick this
        file (it sits in your samkit plugin folder), then click the
        "Run script" (play) button.
        -- OR --
        Just open this file in any text editor, copy its entire contents,
        paste them into the Python Console's input line, and press Enter.
     3. Read the printed output. It tells you exactly what happened.
     4. Restart QGIS.

B) From a terminal/command line, if you know which Python QGIS uses:
     - Windows (OSGeo4W Shell): run `py3_env` first, then:
         python "C:\path\to\samkit\install_dependencies.py"
     - macOS/Linux, from a regular terminal:
         /path/to/qgis/bin/python3 /path/to/samkit/install_dependencies.py
     If you're not sure which Python that is, run method A once first --
     it prints the exact interpreter path you can reuse from a terminal.

This script only touches scikit-learn and pykrige. It never modifies QGIS
itself or any other package.
"""

import os
import re
import sys
import glob
import shutil
import tempfile
import subprocess
import importlib.metadata as importlib_metadata


PACKAGES = [("scikit-learn", "sklearn"), ("pykrige", "pykrige")]
SHARED_DEPS = ("numpy", "scipy", "joblib", "threadpoolctl")

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
LIBS_DIR = os.path.join(PLUGIN_DIR, "libs")


def _normalize(name):
    return re.sub(r"[-_.]+", "-", (name or "")).lower()


def _dist_version(dist_name):
    """Installed version of a distribution, or None. Never imports it."""
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


def _module_on_disk(module_name):
    for entry in list(sys.path) + [LIBS_DIR]:
        if not entry or not os.path.isdir(entry):
            continue
        if os.path.isfile(os.path.join(entry, module_name, "__init__.py")):
            return True
        if glob.glob(os.path.join(entry, module_name + ".py")):
            return True
    return False


def check_installed():
    """Status by pip name. Metadata/filesystem only -- no imports, so this
    cannot crash QGIS the way `__import__` of a fresh binary package can."""
    status = {}
    for pip_name, module_name in PACKAGES:
        version = _dist_version(pip_name) or _dist_version(module_name)
        if version:
            status[pip_name] = version
        elif _module_on_disk(module_name):
            status[pip_name] = "unknown"
        else:
            status[pip_name] = None
    return status


def _is_python_named(path):
    """Never execute qgis-bin.exe -- doing so launches a second QGIS."""
    if not path:
        return False
    name = os.path.basename(path).lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return bool(re.fullmatch(r"python(\d+(\.\d+)?)?w?", name))


def _subprocess_kwargs():
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def resolve_python():
    if _is_python_named(sys.executable):
        return sys.executable
    base_dir = os.path.dirname(sys.executable or "")
    candidates = []
    for name in ("python3.exe", "python.exe", "python3", "python"):
        candidates.append(os.path.join(base_dir, name))
    parent = os.path.dirname(base_dir)
    apps_dir = os.path.join(parent, "apps")
    if os.path.isdir(apps_dir):
        for entry in sorted(os.listdir(apps_dir)):
            if entry.lower().startswith("python"):
                for name in ("python3.exe", "python.exe"):
                    candidates.append(os.path.join(apps_dir, entry, name))
    try:
        import sysconfig
        bindir = sysconfig.get_config_var("BINDIR")
        if bindir:
            for name in ("python3.exe", "python.exe", "python3", "python"):
                candidates.append(os.path.join(bindir, name))
    except Exception:
        pass
    for c in candidates:
        if _is_python_named(c) and os.path.isfile(c):
            return c
    return sys.executable


def write_constraints():
    pins = []
    for name in SHARED_DEPS:
        version = _dist_version(name)
        if version:
            pins.append(f"{name}=={version}")
    if not pins:
        return None, []
    fd, path = tempfile.mkstemp(prefix="samkit_constraints_", suffix=".txt", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(pins) + "\n")
    return path, pins


def prune_duplicates():
    """Drop anything in libs/ that duplicates a package QGIS already ships."""
    if not os.path.isdir(LIBS_DIR):
        return
    for name in SHARED_DEPS:
        saved = list(sys.path)
        try:
            sys.path = [p for p in sys.path
                        if os.path.abspath(p or ".") != os.path.abspath(LIBS_DIR)]
            env_version = _dist_version(name)
        finally:
            sys.path = saved
        if not env_version:
            continue
        for pattern in (name, name.replace("-", "_")):
            for path in (glob.glob(os.path.join(LIBS_DIR, pattern)) +
                         glob.glob(os.path.join(LIBS_DIR, pattern + "-*.dist-info")) +
                         glob.glob(os.path.join(LIBS_DIR, pattern + ".libs")) +
                         glob.glob(os.path.join(LIBS_DIR, pattern + ".py"))):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.remove(path)
                    print(f"  Removed redundant '{name}' from libs/ "
                          f"(QGIS already provides {env_version}).")
                except Exception:
                    pass


def run_pip(cmd):
    print("Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    print("")
    try:
        env = dict(os.environ)
        env["PYTHONNOUSERSITE"] = "1"
        env.pop("PIP_TARGET", None)
        env.pop("PIP_USER", None)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, bufsize=1, env=env,
                                **_subprocess_kwargs())
        for line in proc.stdout:
            print(line.rstrip())
        proc.wait()
        return proc.returncode
    except Exception as e:
        print(f"Failed to run pip: {e}")
        return 1


def install_packages(pip_names):
    python_exe = resolve_python()
    print(f"\nUsing interpreter: {python_exe}")
    print(f"Installing into:   {LIBS_DIR}")
    print("(QGIS's own site-packages are NOT modified.)")

    if not _is_python_named(python_exe):
        print("\nERROR: could not find a real python executable for this QGIS install.")
        print("sys.executable points at:", sys.executable)
        print("Try method B in the docstring above, from an OSGeo4W Shell.")
        return False

    os.makedirs(LIBS_DIR, exist_ok=True)
    constraints, pins = write_constraints()
    if pins:
        print("\nPinning packages QGIS already provides, so they are never changed:")
        for pin in pins:
            print("   ", pin)

    cmd = [python_exe, "-m", "pip", "install",
           "--target", LIBS_DIR,
           "--upgrade",
           "--only-binary=:all:",
           "--disable-pip-version-check",
           "--no-warn-script-location",
           "--no-input"]
    if constraints:
        cmd += ["--constraint", constraints]
    cmd += list(pip_names)

    print("")
    try:
        rc = run_pip(cmd)
    finally:
        if constraints:
            try:
                os.remove(constraints)
            except OSError:
                pass

    if rc == 0:
        print("")
        prune_duplicates()
    return rc == 0


def main():
    print("=" * 70)
    print("SAMKit standalone dependency installer")
    print("=" * 70)
    print(f"Python interpreter in use:\n  {sys.executable}")

    print("\nChecking current status...")
    status = check_installed()
    for pkg, version in status.items():
        print(f"  {pkg}: {'installed (v%s)' % version if version else 'NOT installed'}")

    missing = [pkg for pkg, version in status.items() if not version]
    if not missing:
        print("\nEverything is already installed. Nothing to do.")
        return

    print(f"\nInstalling: {', '.join(missing)}")
    install_packages(missing)

    print("\nRe-checking status...")
    status = check_installed()
    for pkg, version in status.items():
        print(f"  {pkg}: {'installed (v%s)' % version if version else 'STILL NOT installed'}")

    if all(status.values()):
        print("\n" + "=" * 70)
        print("SUCCESS -- now RESTART QGIS.")
        print("=" * 70)
        print("The packages are on disk but cannot be loaded into a QGIS session")
        print("that is already running. They will be picked up automatically on")
        print("the next start; nothing else needs to be done.")
    else:
        print("\nSome packages are still missing. Common causes:")
        print("  - No internet access, or a proxy/firewall blocking pypi.org")
        print("    (try adding: --proxy http://user:pass@host:port to the pip command)")
        print("  - No prebuilt wheel for this QGIS Python version. Check the pip")
        print("    output above for 'Could not find a version that satisfies'.")
        print("  - A read-only plugin folder. Check you can write to:")
        print(f"      {LIBS_DIR}")
        print("\nNote: do NOT work around this by running plain 'pip install")
        print("--upgrade scikit-learn' into QGIS's Python. That upgrades numpy and")
        print("scipy underneath QGIS and will break your QGIS installation.")


if __name__ == "__main__":
    main()
