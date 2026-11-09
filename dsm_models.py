# -*- coding: utf-8 -*-
"""
dsm_models.py
Model fitting + full-raster prediction for the Spatial Predictive Modeling tab.

Validation strategy
--------------------
Soil/environmental sample datasets are often small (tens to a few hundred
points). A single random 80/20 train/test split is a very unstable way to
estimate accuracy on small datasets. To give a much more stable estimate,
models here can be validated with **k-fold cross-validation** (fold
assignments generated internally via `kfold_indices`, or supplied externally
by the caller so the UI layer can save the exact split to disk and reuse the
identical assignment for metrics). An alternative **spatial block split** or
an **external validation layer** (fit on one dataset, evaluate on a
completely separate one) are supported at the UI layer using the
`compute_metrics` function exposed here.

Variogram fitting and fallback
-------------------------------
Kriging-based models (Ordinary Kriging, Regression Kriging) auto-fit
variogram parameters (nugget/sill/range, or slope/nugget
for a linear model) from the data by default, the same way gstat's
`fit.variogram()` does. pykrige does not always raise an exception on a
degenerate/near-singular fit -- it can silently return a near-zero partial
sill instead -- so `_fit_variogram_model_with_fallback` checks for that and
falls back to a linear variogram model, mirroring the fallback pattern used
in comparable geostatistics workflows (e.g. gstat-based R scripts that catch
`attr(vgm, 'singular')` and retry with `vgm("Lin")`). Users can also supply
explicit nugget/sill/range values to skip auto-fitting entirely.

Optional third-party dependencies (imported lazily, with clear error
messages if missing):
  - scikit-learn : Random Forest, SVM, and the trend model for Regression Kriging
  - pykrige      : Ordinary/Regression Kriging
"""

import numpy as np

from . import deps


class MissingDependencyError(RuntimeError):
    pass


def compute_metrics(y_true, y_pred):
    """
    Standard accuracy metrics for continuous predictions, comparable to what
    common DSM validation workflows report (e.g. R's `goof()`/`goof2()`):
      - r2   : prediction-efficiency R2 (1 - SSres/SStot; equals Nash-Sutcliffe
               efficiency). Note this is NOT the same quantity as the R2 of a
               linear regression of predicted-on-observed -- that alternative
               definition is also reported here as `r2_regression` since some
               workflows (including gstat/ithir-style `goof()` scripts) use it.
      - rmse : root mean squared error
      - mae  : mean absolute error
      - bias : mean(predicted) - mean(observed) (systematic over/under-prediction)
      - ccc  : Lin's (1989) concordance correlation coefficient -- penalizes both
               poor correlation AND departure from the 1:1 line, widely reported
               in DSM validation literature alongside R2/RMSE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(resid)))
    bias = float(np.mean(y_pred) - np.mean(y_true))

    mx, my = np.mean(y_true), np.mean(y_pred)
    s2x, s2y = np.var(y_true), np.var(y_pred)
    sxy = np.mean((y_true - mx) * (y_pred - my))
    denom = s2x + s2y + (mx - my) ** 2
    ccc = float(2 * sxy / denom) if denom > 0 else float("nan")

    r2_regression = float("nan")
    if len(y_true) > 2 and np.std(y_pred) > 1e-12:
        try:
            slope, intercept = np.polyfit(y_true, y_pred, 1)
            fitted = slope * y_true + intercept
            ss_res_lm = np.sum((y_pred - fitted) ** 2)
            ss_tot_lm = np.sum((y_pred - np.mean(y_pred)) ** 2)
            r2_regression = float(1 - ss_res_lm / ss_tot_lm) if ss_tot_lm > 0 else float("nan")
        except Exception:
            pass

    return {"rmse": rmse, "r2": r2, "mae": mae, "bias": bias, "ccc": ccc,
            "r2_regression": r2_regression}


# backwards-compatible private alias
_metrics = compute_metrics


def kfold_indices(n, k, seed=42):
    """Return a list of k arrays of row indices (a random partition of range(n))."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)


_kfold_indices = kfold_indices  # backwards-compatible private alias


def repeated_split_indices(n, k, calibration_fraction=0.7, seed=42):
    """
    Repeated random calibration/validation splits (Monte Carlo CV).
    Returns a list of k arrays of VALIDATION row indices; in each one a fresh
    random `calibration_fraction` of the points is used for calibration and the
    remaining points for validation. Unlike k-fold, the validation sets of
    different repetitions can overlap.
    """
    n = int(n)
    n_val = int(round(n * (1.0 - float(calibration_fraction))))
    n_val = max(1, min(n - 1, n_val))
    rng = np.random.RandomState(seed)
    return [np.sort(rng.permutation(n)[:n_val]) for _ in range(int(k))]


def is_partition(folds, n):
    """True if `folds` is a k-fold style partition (every row in exactly one fold)."""
    allv = np.concatenate([np.asarray(f, dtype=int) for f in folds]) if len(folds) else np.array([], int)
    return len(allv) == int(n) and len(np.unique(allv)) == int(n)


def _finish_repeated_cv(label, y, fold_records, cv_out, log):
    """
    Pool the held-out predictions of repeated random splits. Because a point can
    be validated in several repetitions, predictions are stacked (one entry per
    point per repetition) instead of one per point.
    fold_records: list of (fold_index, test_idx, predictions).
    """
    y = np.asarray(y, dtype=float)
    if fold_records:
        pidx = np.concatenate([np.asarray(t, dtype=int) for _, t, _ in fold_records])
        pred = np.concatenate([np.asarray(p, dtype=float) for _, _, p in fold_records])
        fidx = np.concatenate([np.full(len(t), i, dtype=int) for i, t, _ in fold_records])
    else:
        pidx, pred, fidx = np.array([], int), np.array([], float), np.array([], int)
    obs = y[pidx]
    valid = ~np.isnan(pred)
    metrics = None
    if valid.sum() < 5:
        log(f"[{label}] not enough successful repetitions to compute validation metrics.")
    else:
        metrics = compute_metrics(obs[valid], pred[valid])
        log(f"[{label}] repeated random-split validation ({len(fold_records)} repetitions, "
            f"n={int(valid.sum())} pooled held-out predictions of "
            f"{len(np.unique(pidx[valid]))} distinct points): "
            f"R2={metrics['r2']:.3f}  RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  "
            f"bias={metrics['bias']:.3f}  CCC={metrics['ccc']:.3f}")
    if cv_out is not None:
        cv_out["observed"] = obs
        cv_out["predicted"] = pred
        cv_out["fold_assignment"] = fidx
        cv_out["point_index"] = pidx
        cv_out["repeated_split"] = True
        cv_out["metrics"] = metrics
        cv_out["per_fold"] = per_fold_metrics(obs, pred, fidx)
    return metrics


def folds_to_assignment(n, folds):
    """Return an (n,) int array giving each row's fold number (0-based, -1 if unassigned)."""
    assignment = np.full(int(n), -1, dtype=int)
    for i, idx in enumerate(folds):
        assignment[np.asarray(idx, dtype=int)] = i
    return assignment


def per_fold_metrics(y_true, y_pred, fold_assignment, min_n=2):
    """
    Accuracy metrics computed separately within each fold (or each group in
    `fold_assignment`), in addition to the pooled out-of-fold metrics. Pooled
    metrics are the headline number to report; the per-fold breakdown shows how
    stable that number is across folds, which matters a lot for the small
    sample sizes typical of field-collected soil/environmental data.

    Returns a list of dicts, each the usual metrics dict plus 'fold' (1-based)
    and 'n'.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fold_assignment = np.asarray(fold_assignment)
    rows = []
    for f in sorted({int(v) for v in fold_assignment if int(v) >= 0}):
        sel = (fold_assignment == f) & ~np.isnan(y_pred) & ~np.isnan(y_true)
        if int(sel.sum()) < min_n:
            continue
        met = compute_metrics(y_true[sel], y_pred[sel])
        met["fold"] = f + 1
        met["n"] = int(sel.sum())
        rows.append(met)
    return rows


def _store_cv(cv_out, y, oof_pred, fold_assignment, metrics):
    """
    Fill the caller-supplied `cv_out` dict (if any) with the held-out
    predictions themselves, not just their summary metrics, so the UI layer
    can write a full observed-vs-predicted table and plot. Kept as an opt-in
    output parameter so the fit_* functions' existing return signatures are
    unchanged.
    """
    if cv_out is None:
        return
    cv_out["observed"] = np.asarray(y, dtype=float)
    cv_out["predicted"] = np.asarray(oof_pred, dtype=float)
    cv_out["fold_assignment"] = np.asarray(fold_assignment)
    cv_out["metrics"] = metrics
    cv_out["per_fold"] = per_fold_metrics(y, oof_pred, fold_assignment)


def predict_over_stack(predict_fn, stack, block_rows=256):
    """
    Apply predict_fn (X -> y, X shape (n_pixels, n_features)) over a raster stack of
    shape (n_features, rows, cols), row-block at a time to limit memory use.
    Returns a (rows, cols) array with np.nan where any input band is nan.
    """
    n_feat, rows, cols = stack.shape
    out = np.full((rows, cols), np.nan, dtype=np.float64)
    for r0 in range(0, rows, block_rows):
        r1 = min(rows, r0 + block_rows)
        block = stack[:, r0:r1, :]  # (n_feat, br, cols)
        br = r1 - r0
        flat = block.reshape(n_feat, br * cols).T  # (br*cols, n_feat)
        valid = ~np.any(np.isnan(flat), axis=1)
        preds = np.full(flat.shape[0], np.nan)
        if np.any(valid):
            preds[valid] = predict_fn(flat[valid])
        out[r0:r1, :] = preds.reshape(br, cols)
    return out


# ---------------------------------------------------------------------------
# Variogram parameter handling + degenerate-fit fallback
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# gstat-compatible sample variogram + weighted least-squares fit
# (reproduces gstat::variogram() defaults and fit.variogram(fit.method = 7))
# ---------------------------------------------------------------------------

def _vgm_model(h, model, nugget, psill, rng):
    h = np.asarray(h, dtype=float)
    a = max(rng, 1e-12)
    if model == "spherical":
        g = np.where(h < a, 1.5 * h / a - 0.5 * (h / a) ** 3, 1.0)
    elif model == "exponential":           # gstat 'Exp': range parameter a, 1 - exp(-h/a)
        g = 1.0 - np.exp(-h / a)
    elif model == "gaussian":              # gstat 'Gau': 1 - exp(-(h/a)^2)
        g = 1.0 - np.exp(-(h / a) ** 2)
    else:
        raise ValueError(model)
    return nugget + psill * g


def gstat_sample_variogram(xs, ys, z, cutoff=None, width=None):
    """Classical sample variogram with gstat defaults: cutoff = 1/3 of the
    bounding-box diagonal, width = cutoff / 15. Returns (np, dist, gamma)."""
    xs, ys, z = (np.asarray(v, dtype=float) for v in (xs, ys, z))
    if cutoff is None:
        cutoff = np.hypot(xs.max() - xs.min(), ys.max() - ys.min()) / 3.0
    if width is None:
        width = cutoff / 15.0
    i, j = np.triu_indices(len(z), k=1)
    d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
    g = 0.5 * (z[i] - z[j]) ** 2
    keep = d <= cutoff
    d, g = d[keep], g[keep]
    b = np.minimum((d / width).astype(int), int(np.ceil(cutoff / width)) - 1)
    nb = np.bincount(b)
    ok = nb > 0
    dist = np.bincount(b, weights=d)[ok] / nb[ok]
    gam = np.bincount(b, weights=g)[ok] / nb[ok]
    return nb[ok].astype(float), dist, gam


def fit_variogram_gstat(xs, ys, z, model="spherical"):
    """
    Fit nugget/psill/range like gstat::fit.variogram(fit.method = 7): weighted
    least squares with weights N_j / h_j^2 on the gstat-default sample variogram.
    Returns {'nugget','psill','range','sill'} or None if the fit is degenerate.
    """
    from scipy.optimize import least_squares
    n_, h, g = gstat_sample_variogram(xs, ys, z)
    if len(h) < 3:
        return None
    w = np.sqrt(n_ / np.maximum(h, 1e-12) ** 2)
    best = None
    for r0 in (h.max() / 3.0, h.max() / 2.0, h.max(), h.max() * 0.15):
        for fn in (0.0, 0.25, 0.5):
            x0 = [fn * g.max(), max(g.max() * (1 - fn), 1e-9), r0]
            try:
                res = least_squares(lambda p: w * (_vgm_model(h, model, p[0], p[1], p[2]) - g), x0,
                                    bounds=([0, 0, 1e-9], [np.inf, np.inf, np.inf]))
            except Exception:
                continue
            if best is None or res.cost < best.cost:
                best = res
    if best is None:
        return None
    nug, ps, rg = best.x
    if ps < 1e-8 or not np.all(np.isfinite(best.x)):
        return None
    return {"nugget": float(nug), "psill": float(ps), "range": float(rg), "sill": float(nug + ps),
            "gstat_units": True}


def _variogram_param_list(variogram_model, params):
    """Convert a {'nugget','sill','range'} (or {'nugget','slope'}) dict into the
    ordered list pykrige expects for the given model shape."""
    if params is None:
        return None
    if variogram_model == "linear":
        return [params.get("slope", 1.0), params.get("nugget", 0.0)]
    if variogram_model == "power":
        return [params.get("scale", 1.0), params.get("exponent", 1.0), params.get("nugget", 0.0)]
    # spherical / exponential / gaussian: PyKrige's LIST form is [TOTAL sill, range,
    # nugget] (it subtracts the nugget internally, despite the docstring saying
    # 'psill'; verified: with this convention SAMKit reproduces gstat::krige to 6e-8).
    # The UI 'sill' is also the total sill (nugget + partial sill).
    if "psill" in params:
        sill = params["psill"] + params.get("nugget", 0.0)
    else:
        sill = params.get("sill", 1.0)
    rng = params.get("range", 1.0)
    if params.get("gstat_units"):
        # gstat range parameter a -> pykrige range: Exp uses 3a, Gau uses 7a/4
        if variogram_model == "exponential":
            rng = 3.0 * rng
        elif variogram_model == "gaussian":
            rng = 7.0 * rng / 4.0
    return [sill, rng, params.get("nugget", 0.0)]


def _is_degenerate_fit(kobj, variogram_model):
    try:
        params = kobj.variogram_model_parameters
    except Exception:
        return False
    if params is None or len(params) == 0:
        return False
    if not np.all(np.isfinite(params)):
        return True
    if variogram_model in ("spherical", "exponential", "gaussian"):
        psill = params[0]
        return psill < 1e-6
    if variogram_model == "linear":
        slope = params[0]
        return slope < 1e-8
    return False


def _fit_variogram_model_with_fallback(krige_cls, kwargs, variogram_model,
                                        variogram_parameters=None, log=print, label="Kriging",
                                        fitting="gstat"):
    """
    Construct a pykrige kriging object (OrdinaryKriging),
    falling back to a linear variogram if the requested model fails to fit or
    fits a degenerate (near-zero sill) variogram, and manual parameters were
    not explicitly supplied (auto-fit only).
    """
    if (variogram_parameters is None and fitting == "gstat"
            and variogram_model in ("spherical", "exponential", "gaussian")):
        gp = fit_variogram_gstat(kwargs["x"], kwargs["y"], kwargs["z"], variogram_model)
        if gp is None:
            log(f"[{label}] gstat-style '{variogram_model}' fit degenerate; "
                f"falling back to a linear variogram model.")
            kobj = krige_cls(variogram_model="linear", variogram_parameters=None, **kwargs)
            return kobj, "linear"
        vparams = _variogram_param_list(variogram_model, gp)
        return krige_cls(variogram_model=variogram_model, variogram_parameters=vparams, **kwargs), variogram_model
    vparams = _variogram_param_list(variogram_model, variogram_parameters)
    try:
        kobj = krige_cls(variogram_model=variogram_model, variogram_parameters=vparams, **kwargs)
        if vparams is None and variogram_model != "linear" and _is_degenerate_fit(kobj, variogram_model):
            log(f"[{label}] variogram fit for '{variogram_model}' looks degenerate "
                f"(near-zero sill); falling back to a linear variogram model.")
            kobj = krige_cls(variogram_model="linear", variogram_parameters=None, **kwargs)
            variogram_model = "linear"
        return kobj, variogram_model
    except Exception as e:
        if variogram_model != "linear" and vparams is None:
            log(f"[{label}] variogram fit for '{variogram_model}' failed ({e}); "
                f"falling back to a linear variogram model.")
            kobj = krige_cls(variogram_model="linear", variogram_parameters=None, **kwargs)
            return kobj, "linear"
        raise


def _cross_validate_feature_model(fit_fn, predict_fn, X, y, n_folds=5, seed=42,
                                   label="Model", log=print, min_train=4, folds=None,
                                   cv_out=None):
    """
    Generic k-fold CV for feature-based (X -> y) models. fit_fn(Xtr, ytr) -> model,
    predict_fn(model, Xte) -> yhat. Returns pooled-out-of-fold metrics dict or None.
    If `folds` (list of index arrays) is supplied, it is used as-is instead of
    generating a new random split -- this lets the caller keep the saved fold
    assignment files perfectly consistent with the metrics that get reported.
    """
    n = X.shape[0]
    if n < 8:
        log(f"[{label}] only {n} usable samples -- too few for reliable "
            f"cross-validation; skipping validation metrics. (Prediction will "
            f"still be produced, but treat it with caution.)")
        return None
    if folds is None:
        k = max(2, min(n_folds, n))
        folds = kfold_indices(n, k, seed)
    else:
        k = len(folds)
    all_idx = np.arange(n)
    oof_pred = np.full(n, np.nan)
    fold_records = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.setdiff1d(all_idx, test_idx)
        if len(train_idx) < min_train or len(test_idx) == 0:
            continue
        model = fit_fn(X[train_idx], y[train_idx])
        oof_pred[test_idx] = predict_fn(model, X[test_idx])
        fold_records.append((i, test_idx, oof_pred[test_idx].copy()))
    if not is_partition(folds, n):
        return _finish_repeated_cv(label, y, fold_records, cv_out, log)
    valid = ~np.isnan(oof_pred)
    if valid.sum() < 5:
        log(f"[{label}] not enough successful CV folds to compute validation metrics.")
        _store_cv(cv_out, y, oof_pred, folds_to_assignment(n, folds), None)
        return None
    metrics = compute_metrics(y[valid], oof_pred[valid])
    log(f"[{label}] {k}-fold cross-validation (n={int(valid.sum())} pooled predictions): "
        f"R2={metrics['r2']:.3f}  RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  "
        f"bias={metrics['bias']:.3f}  CCC={metrics['ccc']:.3f}")
    _store_cv(cv_out, y, oof_pred, folds_to_assignment(n, folds), metrics)
    return metrics


def _cross_validate_ok(xs, ys, z, variogram_model, variogram_parameters=None, n_folds=5,
                        seed=42, log=print, label="OrdinaryKriging", min_train=4, folds=None,
                        cv_out=None, fitting="gstat"):
    from pykrige.ok import OrdinaryKriging
    n = len(z)
    if n < 8:
        log(f"[{label}] only {n} usable samples -- too few for reliable "
            f"cross-validation; skipping validation metrics.")
        return None
    if folds is None:
        k = max(2, min(n_folds, n))
        folds = kfold_indices(n, k, seed)
    else:
        k = len(folds)
    all_idx = np.arange(n)
    oof_pred = np.full(n, np.nan)
    fold_records = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.setdiff1d(all_idx, test_idx)
        if len(train_idx) < min_train or len(test_idx) == 0:
            continue
        try:
            ok, _ = _fit_variogram_model_with_fallback(
                OrdinaryKriging,
                dict(x=xs[train_idx], y=ys[train_idx], z=z[train_idx],
                     verbose=False, enable_plotting=False),
                variogram_model, variogram_parameters, log=log, label=label, fitting=fitting)
            pred, _ = ok.execute("points", xs[test_idx], ys[test_idx])
            oof_pred[test_idx] = pred
            fold_records.append((i, test_idx, np.asarray(pred, dtype=float)))
        except Exception as e:
            log(f"[{label}] fold {i + 1}/{k} failed and was skipped: {e}")
    if not is_partition(folds, n):
        return _finish_repeated_cv(label, z, fold_records, cv_out, log)
    valid = ~np.isnan(oof_pred)
    if valid.sum() < 5:
        log(f"[{label}] not enough successful CV folds to compute validation metrics.")
        _store_cv(cv_out, z, oof_pred, folds_to_assignment(n, folds), None)
        return None
    metrics = compute_metrics(z[valid], oof_pred[valid])
    log(f"[{label}] {k}-fold cross-validation (n={int(valid.sum())} pooled predictions): "
        f"R2={metrics['r2']:.3f}  RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  "
        f"bias={metrics['bias']:.3f}  CCC={metrics['ccc']:.3f}")
    _store_cv(cv_out, z, oof_pred, folds_to_assignment(n, folds), metrics)
    return metrics


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def rf_params(preset="r_randomforest", n_features=None, n_estimators=None):
    """
    Random-forest settings expressed in scikit-learn terms.
      'r_randomforest' : R randomForest defaults (Breiman 2001; Liaw & Wiener 2002):
                         ntree 500, mtry = max(floor(p/3), 1), nodesize 5
                         (R's nodesize k == sklearn min_samples_split k+1, leaf 1)
      'ranger'         : R ranger defaults: 500 trees, mtry = floor(sqrt(p)),
                         min.node.size 5
      'samkit_legacy'  : pre-1.0 SAMKit: 300 trees, all features, min_samples_leaf 2
    """
    p = int(n_features or 1)
    if preset == "ranger":
        d = dict(n_estimators=500, max_features=max(int(np.sqrt(p)), 1), min_samples_split=5, min_samples_leaf=1)
    elif preset == "samkit_legacy":
        d = dict(n_estimators=300, max_features=1.0, min_samples_split=2, min_samples_leaf=2)
    else:
        d = dict(n_estimators=500, max_features=max(p // 3, 1), min_samples_split=6, min_samples_leaf=1)
    if n_estimators:
        d["n_estimators"] = int(n_estimators)
    return d


def fit_random_forest(X, y, n_estimators=None, max_depth=None, min_samples_leaf=None,
                       random_state=42, n_folds=5, folds=None, log=print, cv_out=None,
                       preset="r_randomforest", max_features=None, min_samples_split=None):
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        raise MissingDependencyError(deps.missing_dependency_message("scikit-learn"))

    prm = rf_params(preset, X.shape[1], n_estimators)
    if max_features is not None:
        prm["max_features"] = max_features
    if min_samples_leaf is not None:
        prm["min_samples_leaf"] = min_samples_leaf
    if min_samples_split is not None:
        prm["min_samples_split"] = min_samples_split
    log(f"[RandomForest] settings: {prm} (preset '{preset}')")

    def _fit(Xtr, ytr):
        m = RandomForestRegressor(max_depth=max_depth, random_state=random_state, n_jobs=-1, **prm)
        m.fit(Xtr, ytr)
        return m

    metrics = None
    if folds is not None or (n_folds and n_folds > 1):
        metrics = _cross_validate_feature_model(_fit, lambda m, Xte: m.predict(Xte),
                                                 X, y, n_folds=n_folds, seed=random_state,
                                                 label="RandomForest", log=log, folds=folds,
                                                 cv_out=cv_out)

    model = _fit(X, y)
    try:
        importances = model.feature_importances_
        log(f"[RandomForest] feature importances: {np.round(importances, 3).tolist()}")
    except Exception:
        pass

    return model, model.predict, metrics


# ---------------------------------------------------------------------------
# Support Vector Machine (SVR)
# ---------------------------------------------------------------------------

def fit_svm(X, y, feature_names, kernel="rbf", C=1.0, epsilon=0.1, gamma="scale",
            n_folds=5, folds=None, random_state=42, log=print, cv_out=None):
    try:
        from sklearn.svm import SVR
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import TransformedTargetRegressor
    except ImportError:
        raise MissingDependencyError(deps.missing_dependency_message("scikit-learn"))

    def make_model():
        base = make_pipeline(StandardScaler(), SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma))
        return TransformedTargetRegressor(regressor=base, transformer=StandardScaler())

    metrics = None
    if folds is not None or (n_folds and n_folds > 1):
        metrics = _cross_validate_feature_model(
            lambda Xtr, ytr: make_model().fit(Xtr, ytr),
            lambda m, Xte: m.predict(Xte),
            X, y, n_folds=n_folds, seed=random_state, label="SVM", log=log, folds=folds,
            cv_out=cv_out)

    model = make_model()
    model.fit(X, y)

    return model, model.predict, metrics


# ---------------------------------------------------------------------------
# Ordinary Kriging
# ---------------------------------------------------------------------------

def fit_ordinary_kriging(xs, ys, z, variogram_model="spherical", variogram_parameters=None,
                          n_folds=5, folds=None, random_state=42, log=print, cv_out=None,
                          variogram_fitting="gstat"):
    """
    variogram_parameters: optional dict with keys 'nugget' and either
    {'sill','range'} (spherical/exponential/gaussian) or {'slope'} (linear) or
    {'scale','exponent'} (power). If given, auto-fitting is skipped and these
    exact values are used.
    """
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError:
        raise MissingDependencyError(deps.missing_dependency_message("pykrige"))

    metrics = None
    if folds is not None or (n_folds and n_folds > 1):
        metrics = _cross_validate_ok(xs, ys, z, variogram_model, variogram_parameters,
                                      n_folds=n_folds, seed=random_state,
                                      label="OrdinaryKriging", log=log, folds=folds,
                                      cv_out=cv_out, fitting=variogram_fitting)

    ok, used_model = _fit_variogram_model_with_fallback(
        OrdinaryKriging, dict(x=xs, y=ys, z=z, verbose=False, enable_plotting=False),
        variogram_model, variogram_parameters, log=log, label="OrdinaryKriging",
        fitting=variogram_fitting)
    return ok, metrics


def predict_kriging_grid(ok_model, grid_x_centers, grid_y_centers_desc, backend="C", log=print,
                          return_variance=False):
    """
    grid_x_centers: 1D ascending array of pixel-center x coords.
    grid_y_centers_desc: 1D array of pixel-center y coords, in the ROW order of the
        output raster (typically descending, north to south).
    Returns (rows, cols) prediction array matching that row order, or a
    (prediction, variance) tuple of two such arrays if return_variance=True.
    """
    y_sorted = np.sort(grid_y_centers_desc)  # pykrige wants ascending
    try:
        z, ss = ok_model.execute("grid", grid_x_centers, y_sorted, backend=backend)
    except Exception as e:
        log(f"[Kriging] backend='{backend}' failed ({e}), retrying with backend='vectorized'")
        z, ss = ok_model.execute("grid", grid_x_centers, y_sorted, backend="vectorized")
    z = np.asarray(z)
    ss = np.asarray(ss)
    if grid_y_centers_desc[0] > grid_y_centers_desc[-1]:
        z = z[::-1, :]
        ss = ss[::-1, :]
    if return_variance:
        return z, ss
    return z


def predict_kriging_points(ok_model, xs, ys, backend="vectorized"):
    """Predict an Ordinary/Regression-Kriging model at arbitrary (xs, ys) points."""
    pred, _ = ok_model.execute("points", np.asarray(xs), np.asarray(ys), backend=backend)
    return np.asarray(pred)


# ---------------------------------------------------------------------------
# Regression Kriging = trend model (RF or linear) + kriged residuals (2-step)
# ---------------------------------------------------------------------------

def fit_regression_kriging(X, y, xs, ys, feature_names, trend="random_forest",
                            variogram_model="spherical", variogram_parameters=None,
                            n_estimators=None, n_folds=5, folds=None, random_state=42, log=print,
                            cv_out=None, residuals="oob", rf_preset="r_randomforest",
                            variogram_fitting="gstat"):
    """
    Two-step regression kriging (Hengl et al. 2007): trend model + ordinary
    kriging of the trend residuals.
    residuals='oob' (default, RF trend): residuals are y - out-of-bag prediction,
        the standard practice in R (y - randomForest$predicted); in-bag RF
        residuals ('inbag', pre-1.0 SAMKit) are far too small and understate the
        residual variogram and kriging variance.
    With trend='linear' the result corresponds to gstat KED / universal kriging.
    """
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError:
        raise MissingDependencyError(deps.missing_dependency_message("pykrige"))

    def _fit_trend(Xtr, ytr):
        if trend == "random_forest":
            try:
                from sklearn.ensemble import RandomForestRegressor
            except ImportError:
                raise MissingDependencyError(deps.missing_dependency_message("scikit-learn"))
            prm = rf_params(rf_preset, Xtr.shape[1], n_estimators)
            m = RandomForestRegressor(random_state=random_state, n_jobs=-1,
                                      oob_score=(residuals == "oob"), **prm)
            m.fit(Xtr, ytr)
            return m
        else:
            try:
                from sklearn.linear_model import LinearRegression
            except ImportError:
                raise MissingDependencyError(deps.missing_dependency_message("scikit-learn"))
            m = LinearRegression()
            m.fit(Xtr, ytr)
            return m

    def _trend_fitted(model, Xtr):
        if trend == "random_forest" and residuals == "oob" and hasattr(model, "oob_prediction_"):
            return model.oob_prediction_
        return model.predict(Xtr)

    metrics = None
    n = len(y)
    do_cv = folds is not None or (n_folds and n_folds > 1)
    if do_cv and n >= 8:
        if folds is None:
            k = max(2, min(n_folds, n))
            use_folds = kfold_indices(n, k, random_state)
        else:
            use_folds = folds
            k = len(use_folds)
        all_idx = np.arange(n)
        oof_pred = np.full(n, np.nan)
        fold_records = []
        for i in range(k):
            test_idx = use_folds[i]
            train_idx = np.setdiff1d(all_idx, test_idx)
            if len(train_idx) < 6 or len(test_idx) == 0:
                continue
            try:
                trend_m = _fit_trend(X[train_idx], y[train_idx])
                resid_train = y[train_idx] - _trend_fitted(trend_m, X[train_idx])
                ok_fold, _ = _fit_variogram_model_with_fallback(
                    OrdinaryKriging,
                    dict(x=xs[train_idx], y=ys[train_idx], z=resid_train,
                         verbose=False, enable_plotting=False),
                    variogram_model, variogram_parameters, log=log, label="RegressionKriging",
                    fitting=variogram_fitting)
                resid_test_pred, _ = ok_fold.execute("points", xs[test_idx], ys[test_idx])
                oof_pred[test_idx] = trend_m.predict(X[test_idx]) + resid_test_pred
                fold_records.append((i, test_idx, oof_pred[test_idx].copy()))
            except Exception as e:
                log(f"[RegressionKriging] fold {i + 1}/{k} failed and was skipped: {e}")
        valid = ~np.isnan(oof_pred)
        if not is_partition(use_folds, n):
            metrics = _finish_repeated_cv("RegressionKriging", y, fold_records, cv_out, log)
        elif valid.sum() >= 5:
            metrics = compute_metrics(y[valid], oof_pred[valid])
            log(f"[RegressionKriging] {k}-fold cross-validation "
                f"(n={int(valid.sum())} pooled predictions): "
                f"R2={metrics['r2']:.3f}  RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}  "
                f"bias={metrics['bias']:.3f}  CCC={metrics['ccc']:.3f}")
        else:
            log("[RegressionKriging] not enough successful CV folds to compute validation metrics.")
        if is_partition(use_folds, n):
            _store_cv(cv_out, y, oof_pred, folds_to_assignment(n, use_folds), metrics)
    elif do_cv:
        log(f"[RegressionKriging] only {n} usable samples -- too few for reliable "
            f"cross-validation; skipping validation metrics.")

    # Final model fit on all data
    trend_model = _fit_trend(X, y)
    trend_predict_fn = trend_model.predict
    resid_full = y - _trend_fitted(trend_model, X)
    log(f"[RegressionKriging] trend residuals ({residuals if trend == 'random_forest' else 'fitted'}, "
        f"full data): mean={np.mean(resid_full):.4f} std={np.std(resid_full):.4f}")

    ok_resid, _ = _fit_variogram_model_with_fallback(
        OrdinaryKriging, dict(x=xs, y=ys, z=resid_full, verbose=False, enable_plotting=False),
        variogram_model, variogram_parameters, log=log, label="RegressionKriging",
        fitting=variogram_fitting)

    return {
        "trend_predict_fn": trend_predict_fn,
        "ok_resid": ok_resid,
        "metrics": metrics,
    }


