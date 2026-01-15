import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
    LeaveOneGroupOut,
)
from extract_features import extract_features
from sklearn.base import BaseEstimator, TransformerMixin, clone
from cca import get_cca_spatialfilter
import io
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    make_scorer,
    accuracy_score,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier
from pylsl import resolve_stream, StreamInlet


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def oof_proba_with_logo(X, y, groups, best_params):
    logo = LeaveOneGroupOut()
    oof = np.zeros_like(y, dtype=float)
    for tr_idx, te_idx in logo.split(X, y, groups):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te = X[te_idx]
        # per-fold pos weight (safer when class balance varies per fold)
        pos = int((y_tr == 1).sum())
        neg = int((y_tr == 0).sum())
        spw = neg / max(pos, 1)
        model = XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=spw,
            **best_params,
        )
        model.fit(X_tr, y_tr)
        oof[te_idx] = model.predict_proba(X_te)[:, 1]
    return oof


def select_channels(raw, labels, keep_channels=None):
    labels = list(labels)
    if keep_channels is None:
        idx = list(range(len(labels)))
        kept_labels = labels
    else:
        keep_list = list(keep_channels)  # keep order if list provided
        keep_set = set(keep_list)
        idx = [i for i, ch in enumerate(labels) if ch in keep_set]
        missing = [ch for ch in keep_list if ch not in labels]
        if missing:
            raise ValueError(f"Requested channels not found: {missing}")
        # preserve original labels order as they appear in 'labels'
        kept_labels = [labels[i] for i in idx]

    if raw.shape[0] == len(labels):
        eeg = raw[idx, :]
    elif raw.shape[1] == len(labels):
        eeg = raw[:, idx]
    else:
        raise ValueError(
            f"labels length ({len(labels)}) doesn't match raw shape {raw.shape}"
        )

    return eeg, kept_labels, idx


def get_channel_from_lsl(stream_type="EEG"):
    """
    Retrieve channel names from an LSL stream.

    Parameters:
        stream_type (str): The type of stream to resolve (default is 'EEG').

    Returns:
        list: A list of channel names from the resolved LSL stream.
    """
    print(f"Looking for a {stream_type} stream...")

    # Resolve the stream
    streams = resolve_stream("type", stream_type)
    if not streams:
        raise RuntimeError(f"No {stream_type} stream found.")

    # Create an inlet to the first available stream
    inlet = StreamInlet(streams[0])

    # Get stream info and channel names
    stream_info = inlet.info()
    desc = stream_info.desc()
    channel_names = []

    # Parse the channel names from the stream description
    channels = desc.child("channels").child("channel")
    while channels.name() == "channel":
        channel_names.append(channels.child_value("label"))
        channels = channels.next_sibling()

    return channel_names


def print_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity / Recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity

    print(f"Accuracy: {acc:.2f}")
    print(f"TPR: {tpr:.2f}")
    print(f"TNR: {tnr:.2f}")


def extract_features_runwise(runs):
    """
    runs: list-like, each element holds all trials for that run
    returns: list of 2D arrays, each (n_trials_in_run, n_features)
    """
    out = []
    for r in runs:
        out.append(extract_features(r))
    return out


def build_X_y_groups(runs_target, runs_nontarget, max_runs=None, run_start_index=1):
    """
    Concatenate features across runs and build a groups vector (one group per run).
    """
    if max_runs is None:
        max_runs = min(len(runs_target), len(runs_nontarget))
    else:
        max_runs = min(max_runs, len(runs_target), len(runs_nontarget))

    X_list, y_list, g_list = [], [], []

    # Precompute features per run so we only extract once
    feats_t = extract_features_runwise(runs_target[:max_runs])
    feats_nt = extract_features_runwise(runs_nontarget[:max_runs])

    for i in range(max_runs):
        Xt = feats_t[i]  # (n_pos_i, n_feat)
        Xnt = feats_nt[i]  # (n_neg_i, n_feat)
        X_run = np.vstack([Xt, Xnt])  # (n_pos_i+n_neg_i, n_feat)
        y_run = np.hstack(
            [np.ones(Xt.shape[0], dtype=int), np.zeros(Xnt.shape[0], dtype=int)]
        )
        grp = np.full(y_run.shape[0], run_start_index + i, dtype=int)

        X_list.append(X_run)
        y_list.append(y_run)
        g_list.append(grp)

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    groups = np.hstack(g_list)
    return X, y, groups


# --- custom metric: TPR x TNR ---
def tpr_tnr_product(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity
    return tpr * tnr


def print_metrics(y_true, y_pred, label=""):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prod = tpr * tnr
    if label:
        print(label)
    print(
        f"  Acc={acc:.3f}  Prec={prec:.3f}  Rec/TPR={tpr:.3f}  TNR={tnr:.3f}  F1={f1:.3f}  TPR×TNR={prod:.3f}"
    )


def per_run_balance(y, groups):
    for r in np.unique(groups):
        m = groups == r
        pos = int((y[m] == 1).sum())
        neg = int((y[m] == 0).sum())
        print(
            f"Run {int(r)}: n={m.sum()}  pos={pos}  neg={neg}  pos_rate={pos/(pos+neg):.2f}"
        )


def tpr_tnr_from_labels(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return tpr, tnr, tpr * tnr


def binarize(p, thr):
    return (p >= thr).astype(int)


def runwise_nested_threshold_eval(
    X,
    y,
    groups,
    best_params,
    base_pos_weight=None,
    thresholds=np.linspace(0.05, 0.95, 91),
):
    """
    Outer CV: leave one run out as test.
    Inner CV on the training runs to choose a threshold that maximizes TPR×TNR,
    then apply that threshold on the held-out run.
    """
    logo = LeaveOneGroupOut()
    results = []
    for test_run in np.unique(groups):
        te = groups == test_run
        tr = ~te

        X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]
        X_te, y_te = X[te], y[te]

        # out-of-fold probs on training runs for inner threshold search
        inner_logo = LeaveOneGroupOut()
        oof = np.zeros_like(y_tr, dtype=float)

        pos = int((y_tr == 1).sum())
        neg = int((y_tr == 0).sum())
        spw = base_pos_weight if base_pos_weight is not None else (neg / max(pos, 1))
        model = XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=spw,
            **best_params,
        )

        for tr_idx, va_idx in inner_logo.split(X_tr, y_tr, g_tr):
            model.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof[va_idx] = model.predict_proba(X_tr[va_idx])[:, 1]

        # pick threshold on training data only
        best_thr, best_prod = 0.5, -1.0
        for thr in thresholds:
            y_hat = binarize(oof, thr)
            _, _, prod = tpr_tnr_from_labels(y_tr, y_hat)
            if prod > best_prod:
                best_prod, best_thr = prod, thr

        # retrain on all training runs, test on held-out run with chosen thr
        model.fit(X_tr, y_tr)
        p_te = model.predict_proba(X_te)[:, 1]
        y_hat_te = binarize(p_te, best_thr)
        tpr, tnr, prod = tpr_tnr_from_labels(y_te, y_hat_te)
        results.append(
            {
                "run": int(test_run),
                "thr": float(best_thr),
                "TPR": float(tpr),
                "TNR": float(tnr),
                "TPRxTNR": float(prod),
            }
        )

    mean_prod = np.mean([r["TPRxTNR"] for r in results])
    std_prod = np.std([r["TPRxTNR"] for r in results])
    print("\nNested run-wise (leakage-safe) results:")
    for r in sorted(results, key=lambda d: d["run"]):
        print(
            f"  Run {r['run']}: thr={r['thr']:.2f}  TPR={r['TPR']:.3f}  TNR={r['TNR']:.3f}  TPR×TNR={r['TPRxTNR']:.3f}"
        )
    print(f"  Mean TPR×TNR={mean_prod:.3f}  Std={std_prod:.3f}")
    return results


def make_balanced_sample_weights(y):
    """
    Returns per-sample weights ~ N / (n_classes * n_class[y_i]).
    Heavier weight for the minority class.
    """
    y = np.asarray(y).ravel()
    classes = np.unique(y)
    n = len(y)
    n_classes = len(classes)

    # counts per class
    counts = {c: int((y == c).sum()) for c in classes}
    # avoid divide-by-zero
    for c in classes:
        if counts[c] == 0:
            counts[c] = 1

    # weight for each class
    w_class = {c: n / (n_classes * counts[c]) for c in classes}
    # map to per-sample weights
    return np.array([w_class[c] for c in y], dtype=float)


def project_cca_waveforms(E: np.ndarray, Wc: np.ndarray) -> np.ndarray:
    """
    Project EEG epochs onto CCA spatial weights to get component time series.
    E:  (S, C, T)  samples x channels x trials
    Wc: (C, K)     channels x components
    Returns:
        Y: (S, K, T)  component waveforms for each trial
    """
    S, C, T = E.shape
    K = Wc.shape[1]
    Y = np.empty((S, K, T), dtype=float)
    for t in range(T):
        # (S,C) @ (C,K) -> (S,K)
        Y[:, :, t] = E[:, :, t] @ Wc
    return Y


from sklearn.base import BaseEstimator, TransformerMixin


class CCAWaveformProjector(BaseEstimator, TransformerMixin):
    """
    Fit:    Wc from get_cca_spatialfilter on TRAIN only.
    Transform: project epochs with project_cca_waveforms -> (S,K,T),
               then either flatten to (T, S*K) or call a user feature fn.
    """

    def __init__(
        self,
        samples,
        channels,
        n_components=3,
        max_iter=5000,
        flatten=True,
        feature_fn=None,
    ):
        self.samples = int(samples)
        self.channels = int(channels)
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.flatten = bool(flatten)
        self.feature_fn = (
            feature_fn  # e.g., extract_features; expects (S, K, T) -> (T, F)
        )
        self.Wc_ = None

    def _to_cube(self, X):  # X: (T, S*C) -> (S, C, T)
        T = X.shape[0]
        return X.reshape(T, self.samples, self.channels).transpose(1, 2, 0)

    def fit(self, X, y):
        E = self._to_cube(X)  # (S,C,T)
        self.Wc_, _, _ = get_cca_spatialfilter(
            E, y, n_components=self.n_components, max_iter=self.max_iter
        )
        return self

    def transform(self, X):
        E = self._to_cube(X)  # (S,C,T)
        Y = project_cca_waveforms(E, self.Wc_)  # (S,K,T)
        if self.feature_fn is not None:
            # Your function should accept (S,K,T) and return (T,F)
            return self.feature_fn(Y)
        if self.flatten:
            # (S,K,T) -> (T, S*K)
            return Y.reshape(Y.shape[0] * Y.shape[1], Y.shape[2]).T
        # raw (S,K,T) is not a valid sklearn X; choose flatten=True or provide feature_fn
        raise ValueError("Set flatten=True or provide feature_fn to return (T, F).")


def feature_adapter(Y):
    # If extract_features returns 3D, force (T, F):
    Z = extract_features(Y)  # must produce per-trial features
    if Z.ndim == 3:  # e.g., (S', C', T)
        Z = Z.reshape(-1, Z.shape[-1]).T  # (T, S'*C')
    return Z  # must be (T, F)


def build_epoch_cube_y_groups(runs_target, runs_nontarget, max_runs=None, start_gid=1):
    if max_runs is None:
        max_runs = min(len(runs_target), len(runs_nontarget))

    S0, C0, _ = runs_target[0].shape
    E_list, y_list, g_list = [], [], []
    gid = start_gid
    for i in range(max_runs):
        Et = runs_target[i]
        En = runs_nontarget[i]
        E_run = np.concatenate([Et, En], axis=2)  # (S,C,Tt+Tn)
        y_run = np.hstack([np.ones(Et.shape[2], int), np.zeros(En.shape[2], int)])
        g_run = np.full(E_run.shape[2], gid, int)
        E_list.append(E_run)
        y_list.append(y_run)
        g_list.append(g_run)
        gid += 1

    E = np.concatenate(E_list, axis=2)
    y = np.hstack(y_list)
    groups = np.hstack(g_list)
    return E, y, groups, S0, C0


def aucs_overall(y_true, y_prob):
    """Overall (OOF) AUCs."""
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)  # = PR-AUC
    prevalence = np.mean(y_true)  # PR baseline
    return dict(roc_auc=roc_auc, pr_auc=pr_auc, pr_baseline=prevalence)


def aucs_per_run(y_true, y_prob, groups):
    """Per-run AUCs; returns list of dicts (skip runs with single-class y)."""
    out = []
    for r in np.unique(groups):
        m = groups == r
        y_r, p_r = y_true[m], y_prob[m]
        # Need both classes present; otherwise AUC is undefined.
        if len(np.unique(y_r)) < 2:
            out.append(dict(run=int(r), roc_auc=np.nan, pr_auc=np.nan))
            continue
        out.append(
            dict(
                run=int(r),
                roc_auc=roc_auc_score(y_r, p_r),
                pr_auc=average_precision_score(y_r, p_r),
            )
        )
    return out


def norm_1020(s):
    return (
        s.replace("FP", "Fp")
        .replace("FZ", "Fz")
        .replace("CZ", "Cz")
        .replace("PZ", "Pz")
        .replace("POZ", "POz")
        .replace("OZ", "Oz")
    )


def select_channels(raw, labels, keep_channels=None):
    """Your existing select_channels function"""
    labels = list(labels)
    if keep_channels is None:
        idx = list(range(len(labels)))
        kept_labels = labels
    else:
        keep_list = list(keep_channels)
        keep_set = set(keep_list)
        idx = [i for i, ch in enumerate(labels) if ch in keep_set]
        missing = [ch for ch in keep_list if ch not in labels]
        if missing:
            raise ValueError(f"Requested channels not found: {missing}")
        kept_labels = [labels[i] for i in idx]

    if raw.shape[0] == len(labels):
        eeg = raw[idx, :]
    elif raw.shape[1] == len(labels):
        eeg = raw[:, idx]
    else:
        raise ValueError(
            f"labels length ({len(labels)}) doesn't match raw shape {raw.shape}"
        )

    return eeg, kept_labels, idx
