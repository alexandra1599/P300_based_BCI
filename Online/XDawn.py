import numpy as np
from dataclasses import dataclass
from typing import Tuple

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix
from typing import Tuple, Optional, Callable
import config
from mne import create_info, EpochsArray
from mne.preprocessing import Xdawn

# Optional: xgboost or lightgbm; here xgboost
from xgboost import XGBClassifier
from extract_features import extract_features


# ---------- Utilities ----------
def p300_window_to_samples(
    window_s: Tuple[float, float], sfreq: float, tmin: float
) -> Tuple[int, int]:
    """
    Convert (start_s, end_s) in epoch time to sample indices, given sfreq and tmin.
    Example: window_s=(0.30, 0.60), tmin=-0.20  -> samples covering 300-600 ms post-stim.
    """
    start_idx = int(round((window_s[0] - tmin) * sfreq))
    end_idx = int(round((window_s[1] - tmin) * sfreq))
    return max(start_idx, 0), max(end_idx, 0)


def compute_metrics(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    return dict(TPR=tpr, TNR=tnr, ACC=acc, TP=tp, TN=tn, FP=fp, FN=fn)


def best_threshold_by_tpr_tnr_product(y_true, y_prob):
    """
    Pick a single global threshold (0..1) maximizing TPR×TNR on OOF scores.
    """
    # search on a fine grid; you can do ROC convex hull etc., but this is fine
    grid = np.linspace(0.05, 0.95, 181)
    best = (0.5, -1.0)  # (thr, product)
    for thr in grid:
        m = compute_metrics(y_true, y_prob, thr)
        prod = m["TPR"] * m["TNR"]
        if prod > best[1]:
            best = (thr, prod)
    return best  # (thr, best_prod)


# ---------- Xdawn Featurizer ----------
@dataclass
class XdawnFeaturizer(BaseEstimator, TransformerMixin):
    n_components: int = 4
    p300_window_s: Tuple[float, float] = (0.25, 0.60)  # not used if feature_fn provided
    sfreq: float = 256.0
    tmin: float = -0.20
    concat_classes: bool = True
    feature_fn: Optional[Callable] = (
        None  # <--- NEW: your extractor (epochs, fs) -> (n_trials, n_features)
    )

    def fit(self, X, y):
        # X: (n_trials, n_channels, n_times)
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        n_trials, n_ch, _ = X.shape

        info = create_info([f"ch{c}" for c in range(n_ch)], self.sfreq, ch_types="eeg")
        events = np.c_[
            np.arange(n_trials, dtype=int),
            np.zeros(n_trials, dtype=int),
            y.astype(int),
        ].astype(int)
        event_id = {"non-target": 0, "target": 1}

        self._epochs_ = EpochsArray(
            X,
            info,
            events=events,
            event_id=event_id,
            tmin=self.tmin,
            baseline=None,
            verbose=False,
        )

        # Use covariance regularization for stability (recommended)
        self._xd_ = Xdawn(n_components=self.n_components, reg="ledoit_wolf")
        self._xd_.fit(self._epochs_)

        # cache window indices (used only if feature_fn is None)
        start = int(round((self.p300_window_s[0] - self.tmin) * self.sfreq))
        end = int(round((self.p300_window_s[1] - self.tmin) * self.sfreq))
        self._w_start = max(start, 0)
        self._w_end = max(end, 0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        n_trials, n_ch, _ = X.shape

        info = create_info([f"ch{c}" for c in range(n_ch)], self.sfreq, ch_types="eeg")
        # all-zero event code is fine at transform time
        events = np.c_[
            np.arange(n_trials, dtype=int),
            np.zeros(n_trials, dtype=int),
            np.zeros(n_trials, dtype=int),
        ].astype(int)
        epochs = EpochsArray(
            X,
            info,
            events=events,
            event_id={"non-target": 0},
            tmin=self.tmin,
            baseline=None,
            verbose=False,
        )

        # Xdawn.transform -> (n_trials, n_components * n_classes, n_times)
        X_proj = self._xd_.transform(epochs)

        if self.feature_fn is not None:
            # Reorder to (n_times, n_channels, n_trials) for your extractor.
            # Here "channels" are Xdawn components (for both classes concatenated).
            E = np.transpose(X_proj, (2, 1, 0))  # (n_times, n_comp*, n_trials)
            return self.feature_fn(
                E, fs=int(self.sfreq)
            )  # must return (n_trials, n_features)

        # Fallback: simple mean-in-window per component (if no feature_fn given)
        w0, w1 = self._w_start, min(self._w_end, X_proj.shape[-1])
        return X_proj[..., w0:w1].mean(axis=-1)  # (n_trials, n_comp*)


def run_leakage_safe_cv_with_xdawn(
    X,
    y,
    groups,
    sfreq,
    tmin=-0.20,
    p300_window_s=(0.30, 0.60),
    n_components=4,
    xgb_params=None,
    feature_fn=None,  # <- pass your extract_features here
):
    """
    X: (n_trials, n_channels, n_times)
    y: (n_trials,)
    groups: (n_trials,) run IDs
    """

    if xgb_params is None:
        xgb_params = dict(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            tree_method="hist",
            random_state=42,
        )

    logo = LeaveOneGroupOut()
    oof_prob = np.zeros(len(y), dtype=float)
    oof_true = y.copy()

    for tr_idx, te_idx in logo.split(X, y, groups=groups):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # 1) Fit Xdawn on training folds only
        xd = XdawnFeaturizer(
            n_components=n_components,
            p300_window_s=p300_window_s,  # ignored if feature_fn windows internally
            sfreq=sfreq,
            tmin=tmin,
            concat_classes=True,
            feature_fn=feature_fn,  # <- your extractor
        )
        xd.fit(X_tr, y_tr)

        # 2) Transform to features
        F_tr = xd.transform(X_tr)  # (n_train, n_features)
        F_te = xd.transform(X_te)  # (n_test,  n_features)

        # 3) Scale (fit on train only)
        scaler = StandardScaler()
        F_trs = scaler.fit_transform(F_tr)
        F_tes = scaler.transform(F_te)

        # 4) Class balance for this fold (optional but helpful)
        pos = int((y_tr == 1).sum())
        neg = int((y_tr == 0).sum())
        spw = neg / max(pos, 1)

        clf = XGBClassifier(**{**xgb_params, "scale_pos_weight": spw})
        clf.fit(F_trs, y_tr)
        oof_prob[te_idx] = clf.predict_proba(F_tes)[:, 1]

    # 5) One global threshold on OOF
    thr, best_prod = best_threshold_by_tpr_tnr_product(oof_true, oof_prob)

    # 6) Report per-run metrics at global thr
    runs = np.unique(groups)
    run_rows = []
    for r in runs:
        m = compute_metrics(oof_true[groups == r], oof_prob[groups == r], thr)
        run_rows.append((int(r), m["ACC"], m["TPR"], m["TNR"], m["TPR"] * m["TNR"]))
    run_rows.sort(key=lambda x: x[0])

    mean_prod = np.mean([row[4] for row in run_rows])
    std_prod = (
        np.std([row[4] for row in run_rows], ddof=1) if len(run_rows) > 1 else 0.0
    )

    return dict(
        threshold=thr,
        oof_metrics=compute_metrics(oof_true, oof_prob, thr),
        per_run=run_rows,
        mean_prod=mean_prod,
        std_prod=std_prod,
        oof_prob=oof_prob,
        oof_true=oof_true,
    )


import numpy as np
import time
import pickle


import numpy as np
import time


def compute_min_window_samples_for_bundle(bundle):
    """
    Decide how many samples to pull so the live window matches the offline training window.
    Prefer explicit values saved in the bundle; otherwise fall back to a sane default.
    Expected places to find this:
      - bundle.get("window_samples")
      - bundle["featurizer"].get("window_ms") with bundle["sfreq"]
    Fallback: 800 ms.
    """
    sf = int(round(bundle.get("sfreq", 256)))
    # 1) direct samples if present
    if "window_samples" in bundle and isinstance(
        bundle["window_samples"], (int, float)
    ):
        return int(bundle["window_samples"])
    # 2) via featurizer window_ms if present
    feat = bundle.get("featurizer", {}) or {}
    if "window_ms" in feat and isinstance(feat["window_ms"], (int, float)):
        return int(round(sf * (feat["window_ms"] / 1000.0)))
    # 3) via featurizer (tmin, tmax) in seconds
    if "tmin" in feat and "tmax" in feat:
        try:
            tmin = float(feat["tmin"])
            tmax = float(feat["tmax"])
            return int(round(sf * (tmax - tmin)))
        except Exception:
            pass
    # fallback ~0.8 s
    return int(round(sf * 0.8))


def align_channels_2d(X_S_C, online_ch_names, train_ch_order, fill_value=0.0):
    """
    Reorder 2D array (samples x channels) to match `train_ch_order`.
    If a training channel is missing online, pad that column with fill_value.
    Robust to cases where online_ch_names is longer than X_S_C's columns.
    """
    S, C_live = X_S_C.shape

    # Clip names to the actual number of columns
    online_ch_names = list(online_ch_names[:C_live])

    name2idx = {ch: i for i, ch in enumerate(online_ch_names)}
    cols = []
    missing = []

    for ch in train_ch_order:
        i = name2idx.get(ch, None)
        if i is None or i >= C_live:
            # channel not present live -> pad with fill_value
            missing.append(ch)
            cols.append(np.full((S,), fill_value, dtype=X_S_C.dtype))
        else:
            cols.append(X_S_C[:, i])

    X_aligned = np.stack(cols, axis=1) if cols else np.zeros((S, 0), dtype=X_S_C.dtype)
    extra = [ch for ch in online_ch_names if ch not in train_ch_order]

    return X_aligned, {"missing": missing, "extra": extra}


def classify_epoch_once_xdawn(
    eeg_state,
    bundle,
    online_channel_names,
    mode,
):
    """
    One-shot online classification using the saved xDAWN pipeline.
    - eeg_state.get_baseline_corrected_window(n) must return (C, S) with baseline already removed.
    - online_channel_names must be the LIVE list from LSL (order matches EEGStreamState rows).
    """
    import numpy as np
    import time

    # --- unpack & sanity ---
    if "pipe" not in bundle:
        raise RuntimeError("xDAWN bundle missing 'pipe'.")
    pipe = bundle["pipe"]
    fs = int(round(bundle.get("sfreq", 256)))

    train_ch = bundle.get("train_channels")
    if not train_ch:
        raise RuntimeError(
            "xDAWN bundle missing 'train_channels'. "
            "Retrain & save the exact training channel order (list of strings)."
        )

    # Try to discover how many channels xDAWN expects (from the trained filters)
    xd = None
    try:
        xd = pipe.named_steps.get("xdawn", None) or pipe.named_steps.get("xDAWN", None)
    except Exception:
        xd = None

    n_ch_expected = None
    if xd is not None:
        try:
            # mne.preprocessing.Xdawn stores filters_ shaped (n_comp * n_classes, n_channels)
            n_ch_expected = int(xd._xd_.filters_.shape[1])
        except Exception:
            pass

    # --- decide window length to pull ---
    need_S = compute_min_window_samples_for_bundle(bundle)

    # --- get live window (C_live, S_live), then (S_live, C_live) ---
    X_C_S, _ = eeg_state.get_baseline_corrected_window(need_S)
    X_S_C = X_C_S.T  # (S, C)

    C_live = X_S_C.shape[1]
    if len(online_channel_names) != C_live:
        # Keep going safely; tell yourself what's happening
        print(
            f"[xDAWN] Live columns = {C_live}, online_channel_names = {len(online_channel_names)}; "
            f"will clip names to match live data."
        )
        online_channel_names = online_channel_names[:C_live]

    # --- align to TRAINING channel order exactly ---
    X_S_C_aligned, rep = align_channels_2d(
        X_S_C, online_channel_names, train_ch, fill_value=0.0
    )

    C_train = X_S_C_aligned.shape[1]

    # Optional: hard guard — the aligned channels must match what xDAWN expects
    if n_ch_expected is not None and C_train != n_ch_expected:
        raise RuntimeError(
            f"xDAWN channel mismatch: trained filters expect {n_ch_expected} chans, "
            f"but aligned live has {C_train}. "
            f"Missing: {rep.get('missing')}, Extra online (ignored): {rep.get('extra')}. "
            f"Fix: ensure the bundle's 'train_channels' equals the exact channels used offline."
        )

    # shape (1, C_train, S_live)
    X_trial = X_S_C_aligned.T[None, :, :]

    # --- forward pass ---
    proba = pipe.predict_proba(X_trial)[0]  # [p0, p1]
    classes = np.array(bundle.get("class_labels", [0, 1]))
    idx0 = int(np.where(classes == 0)[0][0])
    idx1 = int(np.where(classes == 1)[0][0])

    p_non = float(proba[idx0])
    p_tar = float(proba[idx1])
    theta_tar = float(getattr(config, "THRESHOLD_TARGET", 0.5))
    theta_non = float(getattr(config, "THRESHOLD_NONTARGET", 0.5))

    if mode == 11:  # TARGET trial (true label = 1)
        # Predict class 1 if P(target) beats its threshold, else class 0
        yhat = 1 if p_tar >= theta_tar else 0
        conf = p_tar
    else:  # NON-TARGET trial (true label = 0)
        # Predict class 0 if P(non) beats its threshold, else class 1
        yhat = 0 if p_non >= theta_non else 1
        conf = p_non
    all_probs = [[time.time(), p_non, p_tar]]
    return yhat, conf, all_probs


