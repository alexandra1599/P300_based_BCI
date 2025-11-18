import numpy as np
import io, sys, os, datetime, atexit
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    cross_val_score,
    cross_val_predict,
)
from sklearn.metrics import (
    confusion_matrix,
    make_scorer,
    accuracy_score,
    precision_recall_fscore_support,
)
import config
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from xgboost import XGBClassifier
from load_project_data import load_project_data
from extract_data import extract_data
from remove_aux import remove_aux
from filtering import filtering
from target_epochs import target_epochs
from run_analysis import run_analysis
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
    LeaveOneGroupOut,
)
from XDawn import (
    p300_window_to_samples,
    compute_metrics,
    best_threshold_by_tpr_tnr_product,
    XdawnFeaturizer,
    run_leakage_safe_cv_with_xdawn,
)
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from extract_features import extract_features
from xgboost import XGBClassifier
from tuneboost import tune_xgboost
import pickle
from sklearn.base import BaseEstimator, TransformerMixin, clone
from pylsl import resolve_stream, StreamInlet
from cca import get_cca_spatialfilter, rank_channels_component, load_trained_model_cca
from helpers import (
    build_epochs_flat_y_groups,
    TemporalCCAFeaturizer,
    oof_proba_with_logo_pipeline,
    Tee,
)
from sklearn.cross_decomposition import CCA

import matplotlib as plt
import mne
from matplotlib.cm import ScalarMappable  # Add colorbar explicitly
from matplotlib.colors import Normalize  # Add colorbar explicitly
from Topoplot import (
    common_vlim,
    p3_mean_amplitude,
    p3_peak_amplitude,
    plot_topo,
    import_montage,
)


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


def main():
    num_sub = int(input("How many subjects do you want to study ? \n"))
    nback_pre_target, nback_pre_nontarget = [], []
    nback_post_target, nback_post_nontarget = [], []
    nback_online_target, nback_online_nontarget = [], []

    model_pre_target, model_pre_nontarget = [], []
    model_post_target, model_post_nontarget = [], []

    for sub in range(num_sub):

        ID = int(input("Enter Subject ID (e.g., 102): "))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logs_dir = os.path.join(
            "/home/alexandra-admin/Documents/Offline/offline_logs", f"sub-{ID}"
        )
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, f"offline_train_{timestamp}.txt")

        # --- Tee stdout/stderr to file + console ---
        log_file = open(log_path, "w", buffering=1, encoding="utf-8")  # line-buffered
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)

        # Optional: prettier numpy printing
        np.set_printoptions(precision=3, suppress=True)
        print(f"📄 Logging *everything* to: {log_path}")

        n_sessions = int(
            input("How many sessions do you want to load for this subject? ")
        )

        for s in range(n_sessions):

            task = int(input("Do you want to load ?\n [1]Offline\n [2] Online\n"))
            model_filter = int(
                input(
                    "Do you want to use non-causal(offline) or causal(online) filter ?\n [1]Offline\n [2] Online\n"
                )
            )
            print(f"\n=== Loading Session {s+1} ===")
            session_input = input("Session (1 = Relaxation, 2 = tACS): ").strip()
            session_map = {"1": "Relaxation", "2": "tACS"}
            session = session_map.get(session_input, session_input)

            print("\nType options:")
            type_options = [
                "Eyes Closed pre",
                "Eyes Closed post",
                "Eyes Closed pre tACS",
                "Eyes Closed post tACS",
                "Eyes Closed find tACS",
                "Eyes Closed pre nothing",
                "Eyes Closed post nothing",
                "Relax",
                "Nothing",
                "Nback",
                "Nback + relax",
                "Online",
                "Nback + tACS",
                "EOG",
            ]
            for i, opt in enumerate(type_options):
                print(f"{i + 1}: {opt}")
            type_index = int(input("Type of the session (number): "))
            data_type = type_options[type_index - 1]

            runs = int(input("Number of runs in the session: "))
            N = int(input("N-back level (input 9 if not N-back): "))
            N_param = None if N == 9 else N

            # labels = get_channel_from_lsl()
            # labels = np.delete(labels, np.arange(32, 39), axis=0)
            # print(labels)
            eeg_runs, marker_runs, labels = load_project_data(
                ID, session, task, runs, data_type, N_param
            )
            do_car = input("Apply CAR? [y/n]: ").strip().lower() == "y"

            session_target, session_nontarget = [], []
            model_input_target, model_input_nontarget = [], []

            for eeg_dict, marker_dict in zip(eeg_runs, marker_runs):
                eeg = eeg_dict["data"]
                timestamps = eeg_dict["timestamps"]
                m_data = marker_dict["values"]
                m_time = marker_dict["timestamps"]
                fs = 512

                filtered_eeg = filtering(
                    eeg,
                    session=session,
                    filter=model_filter,
                    labels=labels,
                    fs=fs,
                    do_car=do_car,
                )
                online_channel_names = [
                    "FZ",
                    "CZ",
                    "PZ",
                    "P3",
                    "P4",
                    "POZ",
                ]  # ordered LIST
                # online_channel_names = config.P300_CHANNEL_NAMES
                # online_channel_names = ["PZ", "FPZ", "FZ", "P3", "P4", "POZ"]

                eeg_channels, kept_labels, kept_idx = select_channels(
                    filtered_eeg, labels, online_channel_names
                )
                segments_target, segments_nontarget = target_epochs(
                    filtered_eeg, m_data, m_time, fs, timestamps, task
                )
                starget, snontarget = target_epochs(
                    eeg_channels, m_data, m_time, fs, timestamps, task
                )

                model_input_target.append(starget)
                model_input_nontarget.append(snontarget)
                session_target.append(segments_target)
                session_nontarget.append(segments_nontarget)

            if (
                task == 1 and data_type == "Nback" and session == "Relaxation"
            ):  # OFFLINE Pre Relaxation
                nback_pre_target = session_target
                nback_pre_nontarget = session_nontarget
                model_pre_target = model_input_target
                model_pre_nontarget = model_input_nontarget
            elif (
                task == 1
                and data_type == "Nback + relax"
                and session == "Relaxation"  # OFFLINE Post Relaxation
            ):
                nback_post_target = session_target
                nback_post_nontarget = session_nontarget
                model_post_target = model_input_target
                model_post_nontarget = model_input_nontarget

            elif (
                task == 2 and data_type == "Nback" and session == "Relaxation"  # ONLINE
            ):
                nback_online_target = session_target
                nback_online_nontarget = session_nontarget

        # Concatenate data
        nback_pre_target_all = (
            np.concatenate(nback_pre_target, axis=2) if nback_pre_target else None
        )
        nback_pre_nontarget_all = (
            np.concatenate(nback_pre_nontarget, axis=2) if nback_pre_nontarget else None
        )
        nback_post_target_all = (
            np.concatenate(nback_post_target, axis=2) if nback_post_target else None
        )
        nback_post_nontarget_all = (
            np.concatenate(nback_post_nontarget, axis=2)
            if nback_post_nontarget
            else None
        )
        online_target_all = (
            np.concatenate(nback_online_target, axis=2) if nback_online_target else None
        )
        online_nontarget_all = (
            np.concatenate(nback_online_nontarget, axis=2)
            if nback_online_nontarget
            else None
        )

        model_pre_target_all = (
            np.concatenate(model_pre_target, axis=2) if model_pre_target else None
        )
        model_pre_nontarget_all = (
            np.concatenate(model_pre_nontarget, axis=2) if model_pre_nontarget else None
        )
        model_post_target_all = (
            np.concatenate(model_post_target, axis=2) if model_post_target else None
        )
        model_post_nontarget_all = (
            np.concatenate(model_post_nontarget, axis=2)
            if model_post_nontarget
            else None
        )

    print("\n✅ Final concatenated shapes:")
    if nback_pre_target_all is not None:
        print("nback_pre_target_all:", nback_pre_target_all.shape)
    if nback_pre_nontarget_all is not None:
        print("nback_pre_nontarget_all:", nback_pre_nontarget_all.shape)
    if nback_post_target_all is not None:
        print("nback_post_target_all:", nback_post_target_all.shape)
    if nback_post_nontarget_all is not None:
        print("nback_post_nontarget_all:", nback_post_nontarget_all.shape)
    if online_target_all is not None:
        print("online_target:", online_target_all.shape)
    if online_nontarget_all is not None:
        print("online_nontarget:", online_nontarget_all.shape)

    print("\n✅ Final concatenated shapes for ML input :")
    if model_pre_target_all is not None:
        print("model_pre_target_all:", model_pre_target_all.shape)
    if model_pre_nontarget_all is not None:
        print("model_pre_nontarget_all:", model_pre_nontarget_all.shape)
    if model_post_target_all is not None:
        print("model_post_target_all:", model_post_target_all.shape)
    if model_post_nontarget_all is not None:
        print("model_post_nontarget_all:", model_post_nontarget_all.shape)

    plot_number = int(input("How many topoplot do y ou want ?"))

    for i in range(0, plot_number):
        plot = int(
            input(
                "Do you want topoplot of : \n [1] Mean P300 Amplitude \n [2] Peak P300 Amplitude \n"
            )
        )

        montage = import_montage("CA-209-dig.fif")
        alias = {"M1": "TP9", "M2": "TP10"}  # if needed
        if plot == 1:  # MEAN AMPLITUDE
            # PRE OFFLINE
            time_ms = np.arange(nback_pre_target_all.shape[0]) * 1000 / fs
            ga_pre = p3_mean_amplitude(
                nback_pre_target_all, time_ms, tmin=0.24, tmax=0.60
            )

            # POST OFFLINE
            time_ms = np.arange(nback_post_target_all.shape[0]) * 1000 / fs
            ga_post = p3_mean_amplitude(
                nback_post_target_all, time_ms, tmin=0.24, tmax=0.60
            )

            # ONLINE
            time_ms = np.arange(online_target_all.shape[0]) * 1000 / fs
            ga_on = p3_mean_amplitude(online_target_all, time_ms, tmin=0.24, tmax=0.60)

            vmin, vmax = common_vlim(
                ga_pre,
                ga_post,
                ga_on,
                labels=labels,
                exclude=("M1", "M2", "T7", "T8"),
                symmetric=True,
            )
            plot_topo(
                ga_pre,
                montage,
                labels,
                1,
                "Pre",
                plot,
                # vlim=(-3, 2.5),
                type="P300",
            )
            plot_topo(
                ga_post,
                montage,
                labels,
                1,
                "Pos",
                plot,
                # vlim=(-3, 2.5),
                type="P300",
            )
            plot_topo(
                ga_on,
                montage,
                labels,
                2,
                "Pos",
                plot,
                # vlim=(-3, 2.5),
                type="P300",
            )

        elif plot == 2:  # Peak Amplitude P300

            # PRE OFFLINE
            time_ms = np.arange(nback_pre_target_all.shape[0]) * 1000 / fs
            ga_pre = p3_peak_amplitude(
                nback_pre_target_all, time_ms, tmin=0.25, tmax=0.60
            )

            time_ms_cpp = np.arange(nback_pre_nontarget_all.shape[0]) * 1000 / fs
            cpp_pre = p3_peak_amplitude(
                nback_pre_nontarget_all, time_ms_cpp, tmin=0.2, tmax=0.50
            )

            # POST OFFLINE
            time_ms = np.arange(nback_post_target_all.shape[0]) * 1000 / fs
            ga_post = p3_peak_amplitude(
                nback_post_target_all, time_ms, tmin=0.25, tmax=0.60
            )

            time_ms_cpp = np.arange(nback_post_nontarget_all.shape[0]) * 1000 / fs
            cpp_post = p3_peak_amplitude(
                nback_post_nontarget_all, time_ms_cpp, tmin=0.2, tmax=0.50
            )

            # ONLINE
            time_ms = np.arange(online_target_all.shape[0]) * 1000 / fs
            ga_on = p3_peak_amplitude(online_target_all, time_ms, tmin=0.25, tmax=0.60)
            time_ms_cpp = np.arange(online_nontarget_all.shape[0]) * 1000 / fs
            cpp_on = p3_peak_amplitude(
                online_nontarget_all, time_ms_cpp, tmin=0.2, tmax=0.50
            )

            plot_topo(
                ga_pre,
                montage,
                labels,
                1,
                "Pre",
                plot,
                type="P300",
            )
            plot_topo(
                ga_post,
                montage,
                labels,
                1,
                "Pos",
                plot,
                type="P300",
            )
            plot_topo(
                ga_on,
                montage,
                labels,
                2,
                "Pos",
                plot,
                type="P300",
            )
            plot_topo(
                cpp_pre,
                montage,
                labels,
                1,
                "Pre",
                plot,
                type="CPP",
            )
            plot_topo(
                cpp_post,
                montage,
                labels,
                1,
                "Pos",
                plot,
                type="CPP",
            )
            plot_topo(
                cpp_on,
                montage,
                labels,
                2,
                "Pos",
                plot,
                type="CPP",
            )

    comparison = int(
        input("Do you want to compare offline to online ERPs?\n [1]Yes\n [2]No\n ")
    )

    if comparison == 1:
        analysis = int(input("How many ERP comparisons do you want to run ?"))
        for i in range(0, analysis):
            run_analysis(
                ID=ID,
                session=session,
                labels=labels,
                p300_pre=nback_pre_target_all,
                p300_post=nback_post_target_all,
                nop300_pre=nback_pre_nontarget_all,
                nop300_post=nback_post_nontarget_all,
                p300_online=online_target_all,
                nop300_online=online_nontarget_all,
                comparison=comparison,
            )
    elif comparison == 2:
        analysis = int(input("How many ERP analyses do you want to run ?"))
        for i in range(0, analysis):
            run_analysis(
                ID=ID,
                session=session,
                labels=labels,
                p300_pre=nback_pre_target_all,
                p300_post=nback_post_target_all,
                nop300_pre=nback_pre_nontarget_all,
                nop300_post=nback_post_nontarget_all,
                p300_online=None,
                nop300_online=None,
                comparison=None,
            )

    training = int(input("Do you want to train a decoder?\n [1]Yes\n [2]No\n"))
    if training == 1:
        # ----------------------------
        # 0) Choose which run lists to use
        #    (swap these to your PRE/POST sets as needed)
        runs_target = model_post_target  # list-like: one element per run
        runs_nontarget = model_post_nontarget  # list-like: one element per run
        # ----------------------------

        tpr_tnr_scorer = make_scorer(tpr_tnr_product, greater_is_better=True)

        # Ask if want to apply CCA
        choice = int(
            input(
                "Feature extractor:\n [1] CCA\n [2] Handcrafted (No CCA)\n [3] Xdawn\n> "
            )
        )

        if choice == 1:  # CCA

            E_target = np.concatenate(runs_target, axis=2)  # -> (S, C, sum_T_pos)
            E_nontarget = np.concatenate(runs_nontarget, axis=2)  # -> (S, C, sum_T_neg)

            # Now combine targets + nontargets
            E = np.concatenate([E_target, E_nontarget], axis=2)
            y = np.hstack(
                [
                    np.ones(E_target.shape[2], dtype=int),
                    np.zeros(E_nontarget.shape[2], dtype=int),
                ]
            )

            S, C, T = E.shape
            X_flat = E.transpose(2, 0, 1).reshape(T, S * C)  # carrier matrix (T, S*C)

            # Groups matching your E order
            t_per_run_target = [rt.shape[2] for rt in runs_target]
            t_per_run_nontarget = [rn.shape[2] for rn in runs_nontarget]
            groups = np.hstack(
                [
                    np.hstack(
                        [np.full(Ti, i + 1) for i, Ti in enumerate(t_per_run_target)]
                    ),
                    np.hstack(
                        [np.full(Ti, i + 1) for i, Ti in enumerate(t_per_run_nontarget)]
                    ),
                ]
            ).astype(int)

            print("Projection done")

            pipe = Pipeline(
                [
                    (
                        "cca_wave",
                        CCAWaveformProjector(
                            samples=S,
                            channels=C,
                            n_components=3,
                            max_iter=5000,
                            flatten=False,
                            feature_fn=feature_adapter,
                        ),
                    ),
                    (
                        "clf",
                        XGBClassifier(
                            eval_metric="logloss",
                            random_state=42,
                            n_jobs=1,
                            tree_method="hist",
                        ),
                    ),
                ]
            )

            # === 2) Class weight ===
            pos = int((y == 1).sum())
            neg = int((y == 0).sum())
            pos_weight = neg / max(pos, 1)

            param_grid = {
                "cca_wave__n_components": [2, 3],
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.03, 0.1],
                "clf__subsample": [0.7, 1.0],
                "clf__colsample_bytree": [0.7, 1.0],
                # "clf__gamma": [0, 1],
                # grid scale_pos_weight
                "clf__scale_pos_weight": [pos_weight],
            }

            logo = LeaveOneGroupOut()
            grid = GridSearchCV(
                pipe,
                param_grid,
                scoring=tpr_tnr_scorer,
                cv=logo,
                n_jobs=1,
                refit=False,
                verbose=1,
            )

            # Fit with groups; do NOT prefit CCA on all data beforehand (avoid leakage)
            grid.fit(X_flat, y, groups=groups)

            # Deterministic tie-breaker over cv_results_
            cv = grid.cv_results_
            idx_best = int(np.argmax(cv["mean_test_score"]))
            # If exact ties occur, np.argmax picks the first occurrence (grid order = deterministic).
            best_params = cv["params"][idx_best]

            # Now build the final, deterministic best model and refit once on ALL data
            best_model = Pipeline(
                [
                    (
                        "cca_wave",
                        CCAWaveformProjector(
                            samples=S,
                            channels=C,
                            n_components=best_params["cca_wave__n_components"],
                            max_iter=5000,
                            flatten=False,
                            feature_fn=feature_adapter,
                        ),
                    ),
                    (
                        "clf",
                        XGBClassifier(
                            eval_metric="logloss",
                            random_state=42,
                            n_jobs=1,
                            tree_method="hist",
                            n_estimators=best_params["clf__n_estimators"],
                            max_depth=best_params["clf__max_depth"],
                            learning_rate=best_params["clf__learning_rate"],
                            subsample=best_params["clf__subsample"],
                            colsample_bytree=best_params["clf__colsample_bytree"],
                            scale_pos_weight=best_params["clf__scale_pos_weight"],
                            # If available in your xgboost, uncomment:
                            # deterministic_histogram=True,
                        ),
                    ),
                ]
            )
            best_model.fit(X_flat, y)
            print("✅ Best params (subject-specific, deterministic):", best_params)
            from sklearn.base import clone

            det_best = clone(best_model)
            # lock down the booster
            det_best.named_steps["clf"].set_params(
                n_jobs=1,  # <- single thread
                random_state=42,
                tree_method="hist",
                subsample=1.0,  # keep 1.0 for strict determinism
                colsample_bytree=1.0,
            )
            try:
                det_best.named_steps["clf"].set_params(deterministic_histogram=True)
            except Exception:
                pass

            det_best.fit(X_flat, y)  # refit once, deterministically

            Wc_refit = det_best.named_steps["cca_wave"].Wc_  # (C, K)
            top = rank_channels_component(
                Wc_refit, ch_names=online_channel_names, component=0
            )
            print("Top channels (comp 0):", top[:10], flush=True)

            # === 5) Run-wise CV score summary ===
            scores = cross_val_score(
                det_best,
                X_flat,
                y,
                groups=groups,
                cv=logo,
                scoring=tpr_tnr_scorer,
                n_jobs=1,  # <- was -1
            )
            print(
                f"Run-wise CV TPR×TNR: mean={scores.mean():.3f}  std={scores.std():.3f}"
            )

            # === 6) Per-run breakdown using CV predictions (each run tested by a model trained on the other runs) ===
            y_pred_cv = cross_val_predict(
                best_model, X_flat, y, groups=groups, cv=logo, n_jobs=1
            )
            for r in np.unique(groups):
                m = groups == r
                print(f"\n--- Run {int(r)} (held out) ---")
                print_metrics(y[m], y_pred_cv[m], label="XGBoost:")

            best_params = grid.best_params_

            # Deterministic OOF probabilities
            oof_true = y.copy()
            oof_proba = np.empty_like(y, dtype=float)

            for tr_idx, te_idx in logo.split(X_flat, y, groups):
                est = clone(det_best)
                est.fit(X_flat[tr_idx], y[tr_idx])
                oof_proba[te_idx] = est.predict_proba(X_flat[te_idx])[:, 1]

            # Choose global θ to maximize TPR×TNR (micro)
            ths = np.unique(np.concatenate(([0.0, 1.0], oof_proba)))
            best_thr, best_prod = 0.5, -1.0
            for thr in ths:
                y_hat = (oof_proba >= thr).astype(int)
                _, _, prod = tpr_tnr_from_labels(oof_true, y_hat)
                if prod > best_prod:
                    best_prod, best_thr = prod, float(thr)

            print(
                f"\nChosen GLOBAL threshold (OOF) = {best_thr:.2f}  (TPR×TNR on OOF={best_prod:.3f})"
            )

            # Per-run metrics at that global θ
            for r in np.unique(groups):
                m = groups == r
                y_hat_r = (oof_proba[m] >= best_thr).astype(int)
                tpr, tnr, prod = tpr_tnr_from_labels(oof_true[m], y_hat_r)
                print(
                    f"  Run {int(r)} @ thr={best_thr:.2f}: TPR={tpr:.3f}  TNR={tnr:.3f}  TPR×TNR={prod:.3f}"
                )

            roc_auc = roc_auc_score(oof_true, oof_proba)
            pr_auc = average_precision_score(oof_true, oof_proba)
            prevalence = float(oof_true.mean())
            print(f"CCA overall ROC-AUC = {roc_auc:.3f}")
            print(f"CCA overall PR-AUC  = {pr_auc:.3f} (baseline={prevalence:.2f})")

            fpr, tpr, _ = roc_curve(oof_true, oof_proba)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC (OOF) with CCA (deterministic)")
            plt.legend()
            plt.tight_layout()
            plt.show()

            prec, rec, _ = precision_recall_curve(oof_true, oof_proba)
            plt.figure(figsize=(5, 4))
            plt.plot(rec, prec, label=f"PR (AP={pr_auc:.3f})")
            plt.hlines(
                prevalence,
                0,
                1,
                colors="gray",
                linestyles="--",
                label=f"Baseline={prevalence:.2f}",
            )
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision–Recall (OOF) with CCA (deterministic)")
            plt.legend()
            plt.tight_layout()
            plt.show()

            model_dir = "/home/alexandra-admin/Documents/saved_models_cca"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"p300_model_cca_sub-{ID}.pkl")

            saved_threshold = float(best_thr) if "best_thr" in globals() else 0.50

            # Pull the fitted CCA step and its weights
            cca_step = None
            Wc = None
            try:
                if (
                    hasattr(best_model, "named_steps")
                    and "cca_wave" in best_model.named_steps
                ):
                    cca_step = best_model.named_steps["cca_wave"]
                    Wc = getattr(cca_step, "Wc_", None)  # (C, K) or None if not set
            except Exception:
                pass

            # Optional: save a channel ranking snapshot (component 0)
            cca_channel_ranking = None
            try:
                if Wc is not None:
                    cca_channel_ranking = rank_channels_component(
                        Wc, ch_names=online_channel_names, component=0
                    )
            except Exception:
                pass

            # Core feature metadata (keep channel order used for training!)
            clf = best_model.named_steps.get("clf", None)
            n_features = getattr(clf, "n_features_in_", None)

            feature_meta = {
                "channels": list(kept_labels),
                "fs": fs,
                "n_features": n_features,  # <- from the classifier
                "features_per_channel": locals().get("features_per_channel", None),
                "cca": {
                    "n_components": getattr(cca_step, "n_components", None),
                    "samples": getattr(cca_step, "samples", None),
                    "channels": getattr(cca_step, "channels", None),
                    "flatten": getattr(cca_step, "flatten", None),
                    "has_feature_fn": getattr(cca_step, "feature_fn", None) is not None,
                    "Wc": Wc,
                    "channel_ranking_comp0": cca_channel_ranking,
                },
            }
            to_save = {
                "model": best_model,
                "scaler": None,
                "threshold": saved_threshold,
                "feature_meta": feature_meta,
                "subject_id": ID,
            }

            with open(model_path, "wb") as f:
                pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"💾 Model saved to {model_path}")
            print(
                f"   threshold={saved_threshold:.2f}, n_features={feature_meta['n_features']}"
            )
            if Wc is not None:
                print(f"   CCA Wc shape: {Wc.shape} (channels x components)")

        elif choice == 2:  # Handcrafted
            # ===== Deterministic subject-specific tuning (LOGO, TPR×TNR) =====

            # 0) Build dataset (no shuffling)
            X, y, groups = build_X_y_groups(runs_target, runs_nontarget, max_runs=None)
            print(X.shape)

            # 1) Class weight (per subject)
            pos = int((y == 1).sum())
            neg = int((y == 0).sum())
            pos_weight = neg / max(pos, 1)

            # 2) Base estimator – single-thread & fixed seed; no stochastic subsampling
            base = XGBClassifier(
                eval_metric="logloss",
                scale_pos_weight=pos_weight,
                random_state=42,
                n_jobs=1,  # <- single thread for determinism
                tree_method="hist",
            )

            # Deterministic, ordered grid
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.03, 0.1],
                "subsample": [1.0],  # <- keep 1.0 for strict determinism
                "colsample_bytree": [1.0],  # <- same
                "gamma": [0, 1],
            }

            logo = LeaveOneGroupOut()
            tpr_tnr_scorer = make_scorer(tpr_tnr_product, greater_is_better=True)

            # 3) GridSearchCV – deterministic: no parallelism, no refit (we refit once explicitly)
            grid = GridSearchCV(
                estimator=base,
                param_grid=param_grid,
                scoring=tpr_tnr_scorer,
                cv=logo,
                n_jobs=1,  # <- deterministic
                refit=False,  # <- we’ll choose + refit once explicitly
                verbose=1,
            )

            # IMPORTANT: pass groups (run-wise splits)
            grid.fit(X, y, groups=groups)

            # Deterministic tie-break (first max by grid order)
            cv = grid.cv_results_
            idx_best = int(np.argmax(cv["mean_test_score"]))
            best_params = cv["params"][idx_best]
            print("✅ Best params (by TPR×TNR):", best_params)

            # 4) Build the final best model and refit once on ALL data (deterministic)
            best_model = XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
                tree_method="hist",
                **best_params,
            )
            best_model.fit(X, y)

            # Optional sanity check: per-run balance
            per_run_balance(y, groups)

            # 5) Leakage-safe OOF probabilities via a single deterministic LOGO loop
            oof_true = y.copy()
            oof_proba = np.empty_like(y, dtype=float)

            for tr_idx, te_idx in logo.split(X, y, groups):
                est = XGBClassifier(
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=1,
                    tree_method="hist",
                    **best_params,
                )
                est.fit(X[tr_idx], y[tr_idx])
                oof_proba[te_idx] = est.predict_proba(X[te_idx])[:, 1]

            # 6) Choose a single global threshold θ to maximize TPR×TNR on OOF
            ths = np.unique(np.concatenate(([0.0, 1.0], oof_proba)))
            best_thr, best_prod = 0.5, -1.0
            for thr in ths:
                y_hat = (oof_proba >= thr).astype(int)
                _, _, prod = tpr_tnr_from_labels(oof_true, y_hat)
                if prod > best_prod:
                    best_prod, best_thr = prod, float(thr)

            print(
                f"\nChosen GLOBAL threshold (OOF) = {best_thr:.2f}  (TPR×TNR on OOF={best_prod:.3f})"
            )

            # 7) Per-run metrics at that GLOBAL threshold (still leakage-safe: OOF)
            per_run_stats = []
            for r in np.unique(groups):
                m = groups == r
                y_true_r = oof_true[m]
                y_hat_r = (oof_proba[m] >= best_thr).astype(int)

                acc = accuracy_score(y_true_r, y_hat_r)
                tpr, tnr, prod = tpr_tnr_from_labels(y_true_r, y_hat_r)

                print(
                    f"  Run {int(r)} @ thr={best_thr:.2f}: "
                    f"Acc={acc:.3f}  TPR={tpr:.3f}  TNR={tnr:.3f}  TPR×TNR={prod:.3f}"
                )
                per_run_stats.append(
                    {
                        "run": int(r),
                        "acc": float(acc),
                        "tpr": float(tpr),
                        "tnr": float(tnr),
                        "prod": float(prod),
                    }
                )
            # 8) ROC / PR from the same OOF vector (deterministic)
            roc_auc = roc_auc_score(oof_true, oof_proba)
            pr_auc = average_precision_score(oof_true, oof_proba)
            prevalence = float(oof_true.mean())
            print(f"Overall ROC-AUC = {roc_auc:.3f}")
            print(f"Overall PR-AUC  = {pr_auc:.3f} (baseline={prevalence:.2f})")

            fpr, tpr_curve, _ = roc_curve(oof_true, oof_proba)
            plt.figure()
            plt.plot(fpr, tpr_curve, label=f"AUC={roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC (OOF, deterministic)")
            plt.legend()
            plt.tight_layout()
            plt.show()

            prec, rec, _ = precision_recall_curve(oof_true, oof_proba)
            plt.figure(figsize=(5, 4))
            plt.plot(rec, prec, label=f"PR (AP={pr_auc:.3f})")
            plt.hlines(
                prevalence,
                0,
                1,
                colors="gray",
                linestyles="--",
                label=f"Baseline={prevalence:.2f}",
            )
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision–Recall (OOF, deterministic)")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # 9) Save exactly what you picked
            model_dir = "/home/alexandra-admin/Documents/saved_models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"p300_model_sub-{ID}.pkl")

            feature_meta = {
                "channels": locals().get("labels", None),
                "fs": locals().get("fs", None),
                "n_features": getattr(best_model, "n_features_in_", None),
            }

            to_save = {
                "model": best_model,
                "scaler": None,
                "threshold": float(best_thr),
                "feature_meta": feature_meta,
                "best_params": best_params,
            }
            with open(model_path, "wb") as f:
                pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"💾 Model saved to {model_path}")
            print(
                f"   threshold={best_thr:.2f}, n_features={feature_meta['n_features']}"
            )

        elif choice == 3:  # XDAWN
            # ===== Deterministic XDAWN (LOGO OOF + single final fit) =====

            # 0) Build epoch cube + labels + groups (no shuffling)
            E, y, groups, S, C = build_epoch_cube_y_groups(
                runs_target, runs_nontarget, max_runs=None, start_gid=1
            )
            # XdawnFeaturizer expects (n_trials, n_channels, n_times)
            X_xdawn = np.transpose(E, (2, 1, 0))  # -> (T, C, S)
            print("\n=== Xdawn (deterministic) ===")

            # 1) Class weight per subject
            pos = int((y == 1).sum())
            neg = int((y == 0).sum())
            pos_weight = neg / max(pos, 1)

            # 2) Deterministic XGB params: no subsampling, single-thread, fixed seed
            xgb_params = dict(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                subsample=1.0,  # <- deterministic
                colsample_bytree=1.0,  # <- deterministic
                reg_lambda=1.0,
                objective="binary:logistic",
                tree_method="hist",
                random_state=42,  # <- deterministic
                n_jobs=1,  # <- deterministic
                scale_pos_weight=pos_weight,
            )

            # 3) Leakage-safe OOF via your helper (sequential LOGO inside)
            res = run_leakage_safe_cv_with_xdawn(
                X=X_xdawn,
                y=y,
                groups=groups,
                sfreq=fs,
                tmin=-0.20,
                p300_window_s=(0.10, 0.80),
                n_components=8,
                xgb_params=xgb_params,  # <- our deterministic params
                feature_fn=extract_features,  # <- your feature extractor
            )

            # 4) Report OOF summary + per-run (deterministic)
            print(f"\nChosen GLOBAL threshold = {res['threshold']:.2f}")
            oof = res["oof_metrics"]
            print(
                f"OOF @thr: ACC={oof['ACC']:.3f}  TPR={oof['TPR']:.3f}  TNR={oof['TNR']:.3f}  "
                f"TPR×TNR={(oof['TPR']*oof['TNR']):.3f}"
            )
            print("\nRun-wise metrics @ global thr:")
            for r, acc, tpr, tnr, prod in res["per_run"]:
                print(
                    f"  Run {r}: Acc={acc:.3f}  TPR={tpr:.3f}  TNR={tnr:.3f}  TPR×TNR={prod:.3f}"
                )
            print(
                f"Run-wise CV TPR×TNR: mean={res['mean_prod']:.3f}  std={res['std_prod']:.3f}"
            )

            # 5) AUCs / curves from the same OOF vector (deterministic)
            oof_true = res["oof_true"]
            oof_proba = res["oof_prob"]

            roc_auc = roc_auc_score(oof_true, oof_proba)
            pr_auc = average_precision_score(oof_true, oof_proba)
            prevalence = float(oof_true.mean())
            print(f"\nXdawn overall ROC-AUC = {roc_auc:.3f}")
            print(f"Xdawn overall PR-AUC  = {pr_auc:.3f} (baseline~{prevalence:.2f})")

            fpr, tpr, _ = roc_curve(oof_true, oof_proba)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], "--", label="Chance")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC (OOF) with XDAWN — deterministic")
            plt.legend()
            plt.tight_layout()
            plt.show()

            prec, rec, _ = precision_recall_curve(oof_true, oof_proba)
            plt.figure()
            plt.plot(rec, prec, label=f"AP={pr_auc:.3f}")
            plt.hlines(
                prevalence,
                0,
                1,
                colors="gray",
                linestyles="--",
                label=f"Baseline={prevalence:.2f}",
            )
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision–Recall (OOF) with XDAWN — deterministic")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # 6) Fit one final end-to-end pipeline on ALL data with the SAME deterministic settings
            xdawn_final = XdawnFeaturizer(
                n_components=8,
                sfreq=fs,
                tmin=-0.20,
                p300_window_s=(0.10, 0.80),
                concat_classes=True,
                feature_fn=extract_features,
            )
            clf_final = XGBClassifier(**xgb_params)
            pipe = Pipeline(
                [
                    ("xdawn", xdawn_final),
                    ("scaler", StandardScaler()),
                    ("clf", clf_final),
                ]
            )
            train_channels = [norm_1020(ch) for ch in config.P300_CHANNEL_NAMES]

            pipe.fit(X_xdawn, y)

            # 7) Package everything for online classification (same as your classify_epoch_once_xdawn bundle)
            bundle = {
                "pipe": pipe,  # sklearn pipeline
                "threshold": float(res["threshold"]),  # θ on P(target)
                "train_channels": list(
                    train_channels
                ),  # preserve exact order if available
                "sfreq": float(fs),
                "featurizer": {
                    "tmin": -0.20,
                    "p300_window_s": (0.10, 0.80),
                    "n_components": 8,
                    "feature_fn_tag": "extract_features@v1",
                },
                "oof_summary": {
                    "oof_metrics": res["oof_metrics"],
                    "per_run": res["per_run"],
                    "mean_prod": res["mean_prod"],
                    "std_prod": res["std_prod"],
                },
                "class_labels": [0, 1],
                "version": "xdawn_pipeline_v1_deterministic",
            }

            model_path = f"/home/alexandra-admin/Documents/saved_models_xdawn/p300_model_xdawn_sub-{ID}.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"💾 Saved Xdawn pipeline to {model_path}")
            print(
                f"   θ={bundle['threshold']:.2f}, channels={bundle['train_channels']}"
            )

    """ finally:
        # Always restore streams and close the file, even if an error occurs
        sys.stdout = old_stdout
        sys.stderr = old_stderr"""

    log_file.close()


if __name__ == "__main__":
    main()
