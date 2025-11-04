# offline_train_compare.py
import numpy as np
import io, sys, os, datetime
from dataclasses import dataclass
from typing import Optional, Callable, Tuple

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
import numpy as np
import matplotlib.pyplot as plt
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
import os
from pylsl import resolve_stream, StreamInlet
from cca import get_cca_spatialfilter, rank_channels_component, load_trained_model_cca
from helpers import (
    build_epochs_flat_y_groups,
    TemporalCCAFeaturizer,
    oof_proba_with_logo_pipeline,
    Tee,
)
from sklearn.cross_decomposition import CCA


# ------------------------
# Small utilities (from your code, kept the same)
# ------------------------


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


def tpr_tnr_product(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return tpr * tnr


def tpr_tnr_from_labels(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return tpr, tnr, tpr * tnr


def per_run_balance(y, groups):
    for r in np.unique(groups):
        m = groups == r
        pos = int((y[m] == 1).sum())
        neg = int((y[m] == 0).sum())
        print(
            f"Run {int(r)}: n={m.sum()}  pos={pos}  neg={neg}  pos_rate={pos/(pos+neg):.2f}"
        )


def extract_features_runwise(runs):
    return [extract_features(r) for r in runs]


def build_X_y_groups(runs_target, runs_nontarget, max_runs=None, run_start_index=1):
    if max_runs is None:
        max_runs = min(len(runs_target), len(runs_nontarget))
    else:
        max_runs = min(max_runs, len(runs_target), len(runs_nontarget))

    X_list, y_list, g_list = [], [], []
    feats_t = extract_features_runwise(runs_target[:max_runs])
    feats_nt = extract_features_runwise(runs_nontarget[:max_runs])

    for i in range(max_runs):
        Xt = feats_t[i]
        Xnt = feats_nt[i]
        X_run = np.vstack([Xt, Xnt])
        y_run = np.hstack([np.ones(Xt.shape[0], int), np.zeros(Xnt.shape[0], int)])
        grp = np.full(y_run.shape[0], run_start_index + i, dtype=int)
        X_list.append(X_run)
        y_list.append(y_run)
        g_list.append(grp)

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    groups = np.hstack(g_list)
    return X, y, groups


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


# ------------------------
# CCA projector (same as your definition)
# ------------------------
class CCAWaveformProjector(BaseEstimator, TransformerMixin):
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
        self.feature_fn = feature_fn
        self.Wc_ = None

    def _to_cube(self, X):
        T = X.shape[0]
        return X.reshape(T, self.samples, self.channels).transpose(1, 2, 0)

    def fit(self, X, y):
        E = self._to_cube(X)
        self.Wc_, _, _ = get_cca_spatialfilter(
            E, y, n_components=self.n_components, max_iter=self.max_iter
        )
        return self

    def transform(self, X):
        E = self._to_cube(X)
        # (S,C,T) @ (C,K)
        S, C, T = E.shape
        K = self.Wc_.shape[1]
        Y = np.empty((S, K, T), dtype=float)
        for t in range(T):
            Y[:, :, t] = E[:, :, t] @ self.Wc_
        if self.feature_fn is not None:
            # feature_fn expects (S,K,T) and returns (T,F)
            return self.feature_fn(Y)
        if self.flatten:
            return Y.reshape(Y.shape[0] * Y.shape[1], Y.shape[2]).T
        raise ValueError("Set flatten=True or provide feature_fn to return (T, F).")


def feature_adapter(Y):
    Z = extract_features(Y)
    if Z.ndim == 3:
        Z = Z.reshape(-1, Z.shape[-1]).T
    return Z


# ------------------------
# Deterministic comparison runners
# ------------------------
from xgboost import XGBClassifier


@dataclass
class ModelOOF:
    name: str
    y: np.ndarray
    groups: np.ndarray
    proba: np.ndarray  # OOF P(class=1)
    theta: float  # global θ picked on OOF


def pick_global_theta(y_true, oof_proba):
    ths = np.unique(np.concatenate(([0.0, 1.0], oof_proba)))
    best_thr, best_prod = 0.5, -1.0
    for thr in ths:
        _, _, prod = tpr_tnr_from_labels(y_true, (oof_proba >= thr).astype(int))
        if prod > best_prod:
            best_thr, best_prod = float(thr), prod
    return best_thr


def run_nocca_oof(runs_target, runs_nontarget) -> ModelOOF:
    X, y, groups = build_X_y_groups(runs_target, runs_nontarget, max_runs=None)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    pos_weight = neg / max(pos, 1)

    base = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
        scale_pos_weight=pos_weight,
        subsample=1.0,
        colsample_bytree=1.0,
    )
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.03, 0.1],
        "subsample": [1.0],
        "colsample_bytree": [1.0],
        "gamma": [0],
    }
    logo = LeaveOneGroupOut()
    scorer = make_scorer(tpr_tnr_product, greater_is_better=True)
    grid = GridSearchCV(
        base, param_grid, scoring=scorer, cv=logo, n_jobs=1, refit=False, verbose=0
    )
    grid.fit(X, y, groups=groups)
    cv = grid.cv_results_
    best = cv["params"][int(np.argmax(cv["mean_test_score"]))]

    # deterministic OOF
    oof = np.empty_like(y, dtype=float)
    for tr, te in logo.split(X, y, groups):
        est = XGBClassifier(
            eval_metric="logloss", random_state=42, n_jobs=1, tree_method="hist", **best
        )
        est.fit(X[tr], y[tr])
        oof[te] = est.predict_proba(X[te])[:, 1]

    theta = pick_global_theta(y, oof)
    return ModelOOF("No CCA", y, groups, oof, theta)


def run_cca_oof(runs_target, runs_nontarget) -> ModelOOF:
    E_t = np.concatenate(runs_target, axis=2)
    E_n = np.concatenate(runs_nontarget, axis=2)
    E = np.concatenate([E_t, E_n], axis=2)
    y = np.hstack([np.ones(E_t.shape[2], int), np.zeros(E_n.shape[2], int)])
    S, C, T = E.shape
    X_flat = E.transpose(2, 0, 1).reshape(T, S * C)

    t_pos = [rt.shape[2] for rt in runs_target]
    t_neg = [rn.shape[2] for rn in runs_nontarget]
    groups = np.hstack(
        [
            np.hstack([np.full(Ti, i + 1) for i, Ti in enumerate(t_pos)]),
            np.hstack([np.full(Ti, i + 1) for i, Ti in enumerate(t_neg)]),
        ]
    ).astype(int)

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    pos_weight = neg / max(pos, 1)

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
                    eval_metric="logloss", random_state=42, n_jobs=1, tree_method="hist"
                ),
            ),
        ]
    )

    param_grid = {
        "cca_wave__n_components": [2, 3],
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.03, 0.1],
        "clf__subsample": [1.0],
        "clf__colsample_bytree": [1.0],
        "clf__scale_pos_weight": [pos_weight],
    }

    logo = LeaveOneGroupOut()
    scorer = make_scorer(tpr_tnr_product, greater_is_better=True)
    grid = GridSearchCV(
        pipe, param_grid, scoring=scorer, cv=logo, n_jobs=1, refit=False, verbose=0
    )
    grid.fit(X_flat, y, groups=groups)
    cv = grid.cv_results_
    best = cv["params"][int(np.argmax(cv["mean_test_score"]))]

    # deterministic OOF with the chosen params
    best_model = Pipeline(
        [
            (
                "cca_wave",
                CCAWaveformProjector(
                    samples=S,
                    channels=C,
                    n_components=best["cca_wave__n_components"],
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
                    n_estimators=best["clf__n_estimators"],
                    max_depth=best["clf__max_depth"],
                    learning_rate=best["clf__learning_rate"],
                    subsample=best["clf__subsample"],
                    colsample_bytree=best["clf__colsample_bytree"],
                    scale_pos_weight=best["clf__scale_pos_weight"],
                ),
            ),
        ]
    )

    oof = np.empty_like(y, dtype=float)
    for tr, te in logo.split(X_flat, y, groups):
        est = clone(best_model)
        est.fit(X_flat[tr], y[tr])
        oof[te] = est.predict_proba(X_flat[te])[:, 1]

    theta = pick_global_theta(y, oof)
    return ModelOOF("CCA", y, groups, oof, theta)


def run_xdawn_oof(runs_target, runs_nontarget, fs) -> ModelOOF:
    E, y, groups, S, C = build_epoch_cube_y_groups(
        runs_target, runs_nontarget, max_runs=None, start_gid=1
    )
    X_xdawn = np.transpose(E, (2, 1, 0))  # (T,C,S)

    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    xgb_params = dict(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        scale_pos_weight=neg / max(pos, 1),
    )

    res = run_leakage_safe_cv_with_xdawn(
        X=X_xdawn,
        y=y,
        groups=groups,
        sfreq=fs,
        tmin=-0.20,
        p300_window_s=(0.10, 0.80),
        n_components=8,
        xgb_params=xgb_params,
        feature_fn=extract_features,
    )
    return ModelOOF(
        "XDawn", res["oof_true"], groups, res["oof_prob"], float(res["threshold"])
    )


# ------------------------
# Plotting helper (TPR blue, TNR red, Accuracy gray line)
# ------------------------
def summarize_and_plot(model: ModelOOF, title_suffix=""):
    runs = np.unique(model.groups)
    accs, tprs, tnrs, names = [], [], [], []
    y_hat = (model.proba >= model.theta).astype(int)

    for r in runs:
        m = model.groups == r
        y_r, yh = model.y[m], y_hat[m]
        tn, fp, fn, tp = confusion_matrix(y_r, yh, labels=[0, 1]).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        names.append(f"Run {int(r)}")
        accs.append(acc)
        tprs.append(tpr)
        tnrs.append(tnr)

    mean_acc = float(np.mean(accs))
    print(f"{model.name}: mean Acc = {mean_acc*100:.1f}%  (θ={model.theta:.2f})")

    x = np.arange(len(names))
    w = 0.35
    plt.figure(figsize=(8, 4.3))
    plt.bar(x - w / 2, tprs, width=w, label="TPR", color="#4c72b0")  # blue
    plt.bar(x + w / 2, tnrs, width=w, label="TNR", color="#c44e52")  # red
    plt.plot(x, accs, marker="o", linestyle="-", color="gray", label="Accuracy")
    plt.xticks(x, names)
    plt.ylim(0, 1.0)
    plt.ylabel("Rate / Score")
    plt.title(f"TPR, TNR, Accuracy per run {title_suffix} — {model.name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return mean_acc


# ------------------------
# ROC & PR plotting
# ------------------------
def plot_roc_pr_comparison(models, title_suffix="(OOF)"):
    """Plot ROC and Precision–Recall curves for multiple models on shared axes."""
    # consistent colors per method
    color_map = {"No CCA": "#4c72b0", "CCA": "#7b6fd0", "XDawn": "#dd8452"}

    # ---- ROC (threshold-free) ----
    plt.figure(figsize=(6.8, 5.2))
    for m in models:
        fpr, tpr, _ = roc_curve(m.y, m.proba)
        auc = roc_auc_score(m.y, m.proba)
        plt.plot(
            fpr,
            tpr,
            label=f"{m.name} (AUC={auc:.3f})",
            linewidth=2,
            color=color_map.get(m.name),
        )
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC {title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Precision–Recall (threshold-free) ----
    # Prevalence is the same across models (same y), so grab from first model.
    prevalence = float(models[0].y.mean())
    plt.figure(figsize=(6.8, 5.2))
    for m in models:
        prec, rec, _ = precision_recall_curve(m.y, m.proba)
        ap = average_precision_score(m.y, m.proba)
        plt.plot(
            rec,
            prec,
            label=f"{m.name} (AP={ap:.3f})",
            linewidth=2,
            color=color_map.get(m.name),
        )
    plt.hlines(
        prevalence,
        0,
        1,
        colors="gray",
        linestyles="--",
        linewidth=1,
        label=f"Baseline={prevalence:.2f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall {title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_roc_pr_single(model, title_suffix="(OOF)"):
    """Plot ROC and PR for one model per figure."""
    # ROC
    fpr, tpr, _ = roc_curve(model.y, model.proba)
    auc = roc_auc_score(model.y, model.proba)
    plt.figure(figsize=(6.2, 4.8))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model.name} — ROC {title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # PR
    prec, rec, _ = precision_recall_curve(model.y, model.proba)
    ap = average_precision_score(model.y, model.proba)
    prevalence = float(model.y.mean())
    plt.figure(figsize=(6.2, 4.8))
    plt.plot(rec, prec, label=f"AP={ap:.3f}", linewidth=2)
    plt.hlines(
        prevalence,
        0,
        1,
        colors="gray",
        linestyles="--",
        linewidth=1,
        label=f"Baseline={prevalence:.2f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model.name} — Precision–Recall {title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------
# MAIN: load data once, run all 3 methods deterministically, compare
# ------------------------
def main():
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

    try:
        n_sessions = int(
            input("How many sessions do you want to load for this subject? ")
        )
        filter = int(input("Do you want to load ?\n [1]Offline [2] Online"))
        model_filter = int(
            input(
                "Do you want to use non-causal(offline) or causal(online) filter ?\n [1]Offline [2] Online"
            )
        )
        nback_pre_target, nback_pre_nontarget = [], []
        nback_post_target, nback_post_nontarget = [], []

        model_pre_target, model_pre_nontarget = [], []
        model_post_target, model_post_nontarget = [], []

        for s in range(n_sessions):
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
                ID, session, filter, runs, data_type, N_param
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
                    filtered_eeg, m_data, m_time, fs, timestamps
                )
                starget, snontarget = target_epochs(
                    eeg_channels, m_data, m_time, fs, timestamps
                )

                model_input_target.append(starget)
                model_input_nontarget.append(snontarget)
                session_target.append(segments_target)
                session_nontarget.append(segments_nontarget)

            if data_type == "Nback" and session == "Relaxation":
                nback_pre_target = session_target
                nback_pre_nontarget = session_nontarget
                model_pre_target = model_input_target
                model_pre_nontarget = model_input_nontarget
            elif data_type == "Nback + relax" and session == "Relaxation":
                nback_post_target = session_target
                nback_post_nontarget = session_nontarget
                model_post_target = model_input_target
                model_post_nontarget = model_input_nontarget

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

        print("\n✅ Final concatenated shapes for ML input :")
        if model_pre_target_all is not None:
            print("model_pre_target_all:", model_pre_target_all.shape)
        if model_pre_nontarget_all is not None:
            print("model_pre_nontarget_all:", model_pre_nontarget_all.shape)
        if model_post_target_all is not None:
            print("model_post_target_all:", model_post_target_all.shape)
        if model_post_nontarget_all is not None:
            print("model_post_nontarget_all:", model_post_nontarget_all.shape)

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
            )

        # ----------------------------
        # 0) Choose which run lists to use
        #    (swap these to your PRE/POST sets as needed)
        runs_target = model_post_target  # list-like: one element per run
        runs_nontarget = model_post_nontarget  # list-like: one element per run
        # ----------------------------

        # =======================
        # Run all three methods deterministically on the SAME data
        # =======================
        print("\n=== Running No-CCA (deterministic) ===")
        m_nocca = run_nocca_oof(runs_target, runs_nontarget)

        print("\n=== Running CCA (deterministic) ===")
        m_cca = run_cca_oof(runs_target, runs_nontarget)

        print("\n=== Running XDawn (deterministic) ===")
        m_xdw = run_xdawn_oof(runs_target, runs_nontarget, fs=512)

        # =======================
        # Compare: per-run bars + mean accuracy for each method
        # =======================
        summarize_and_plot(m_nocca, title_suffix="(Global θ)")
        summarize_and_plot(m_cca, title_suffix="(Global θ)")
        summarize_and_plot(m_xdw, title_suffix="(Global θ)")

        # Optional: print AUC/PR too from OOF
        for m in [m_nocca, m_cca, m_xdw]:
            roc_auc = roc_auc_score(m.y, m.proba)
            pr_auc = average_precision_score(m.y, m.proba)
            prev = float(np.mean(m.y))
            print(
                f"\n{m.name}: ROC-AUC={roc_auc:.3f}  PR-AUC={pr_auc:.3f} (baseline={prev:.2f})"
            )

        # =======================
        # NEW: threshold-free ROC & PR comparisons (same OOF scores)
        # =======================
        plot_roc_pr_comparison(
            [m_nocca, m_cca, m_xdw], title_suffix="(OOF, threshold-free)"
        )

        # (Optional) also see each method separately
        # plot_roc_pr_single(m_nocca, title_suffix="(OOF)")
        # plot_roc_pr_single(m_cca, title_suffix="(OOF)")
        # plot_roc_pr_single(m_xdw, title_suffix="(OOF)")

    finally:
        # restore streams and close log
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        try:
            log_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
