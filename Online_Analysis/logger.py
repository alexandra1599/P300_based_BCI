import numpy as np
from datetime import datetime


def _pos_prob_scalar(p):
    # Return scalar p1 (prob of class 1). If empty/None, return nan.
    if p is None:
        return float("nan")
    a = np.asarray(p)
    if a.size == 0:
        return float("nan")
    # common shapes: [[p0,p1]], [p0,p1], [p1]
    if a.ndim == 2 and a.shape[1] == 2:
        return float(a[0, 1])
    if a.ndim == 1 and a.size == 2:
        return float(a[1])
    return float(a.ravel()[0])


def _scalar_int(x, default=-1):
    if x is None:
        return int(default)
    a = np.asarray(x)
    return int(a.ravel()[0]) if a.size else int(default)


def _scalar_float(x, default=np.nan):
    if x is None:
        return float(default)
    a = np.asarray(x, dtype=float)
    return float(a.ravel()[0]) if a.size else float(default)


def log_trial_prediction(log_path, run_idx, trial_idx, mode, prob, pred, threshold):
    ts = datetime.now().isoformat(timespec="milliseconds")
    prob_s = _pos_prob_scalar(prob)  # -> float (may be nan)
    pred_i = _scalar_int(pred, default=-1)
    thr_f = _scalar_float(threshold, default=np.nan)
    lbl_i = 1 if mode == 11 else 0

    # If prob is nan, force pred to -1 so it’s obvious downstream
    if np.isnan(prob_s):
        pred_i = -1

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"run {run_idx},trial {trial_idx},true label {lbl_i},{prob}, predicted label {pred_i}\n"
        )
