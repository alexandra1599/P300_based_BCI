import numpy as np

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
from cca import get_cca_spatialfilter

import io, sys, os, datetime, atexit


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


def build_epochs_flat_y_groups(runs_target, runs_nontarget, max_runs=None, start_gid=1):
    """
    Flattens each trial (samples x channels) -> (samples*channels,)
    Returns:
      X_flat: (n_trials, S*C)
      y:      (n_trials,)
      groups: (n_trials,)
      S, C:   samples, channels (ints)
    """
    if max_runs is None:
        max_runs = min(len(runs_target), len(runs_nontarget))
    else:
        max_runs = min(max_runs, len(runs_target), len(runs_nontarget))

    S, C, _ = runs_target[0].shape  # assume consistent across runs
    X_list, y_list, g_list = [], [], []
    gid = start_gid
    for i in range(max_runs):
        Et = runs_target[i]  # (S,C,Tt)
        En = runs_nontarget[i]  # (S,C,Tn)
        Xt = Et.transpose(2, 0, 1).reshape(Et.shape[2], S * C)
        Xn = En.transpose(2, 0, 1).reshape(En.shape[2], S * C)
        X_run = np.vstack([Xt, Xn])
        y_run = np.hstack([np.ones(Et.shape[2], int), np.zeros(En.shape[2], int)])
        g_run = np.full(y_run.shape[0], gid, int)
        X_list.append(X_run)
        y_list.append(y_run)
        g_list.append(g_run)
        gid += 1

    X_flat = np.vstack(X_list)
    y = np.hstack(y_list)
    groups = np.hstack(g_list)
    return X_flat, y, groups, S, C


# ----------  B) CCA featurizer (leakage-safe inside CV folds) ----------
class TemporalCCAFeaturizer(BaseEstimator, TransformerMixin):
    """
    Takes flattened epochs per trial (n_trials x S*C), reshapes back to (S,C,T),
    computes:
      - base features via extract_features(E)  -> (n_trials, F)
      - CCA temporal scores (n_trials, K)
    and returns np.hstack([base_features, cca_scores]).
    """

    def __init__(self, samples, channels, n_components=3, max_iter=5000):
        self.samples = int(samples)
        self.channels = int(channels)
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        # fitted artifacts
        self.filters_ = None
        self.x_mean_ = None
        self.base_feat_dim_ = None

    def _to_epoch_cube(self, X):
        # X: (n_trials, S*C) -> E: (S,C,T)
        T = X.shape[0]
        return X.reshape(T, self.samples, self.channels).transpose(1, 2, 0)

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("TemporalCCAFeaturizer requires labels y to fit.")
        E = self._to_epoch_cube(X)  # (S,C,T)
        # Fit temporal CCA filters on TRAIN only
        self.filters_, _ = get_cca_spatialfilter(
            E,
            np.asarray(y).ravel(),
            n_components=self.n_components,
            max_iter=self.max_iter,
        )
        # training mean of trial-averaged time series (for centering)
        trials_avg = E.mean(axis=1)  # (S, T)
        self.x_mean_ = trials_avg.mean(axis=1)  # (S,)
        # learn base feature dimensionality (optional)
        base_feats = extract_features(E)  # your existing function
        self.base_feat_dim_ = base_feats.shape[1]
        return self

    def transform(self, X):
        if self.filters_ is None or self.x_mean_ is None:
            raise RuntimeError("TemporalCCAFeaturizer must be fitted before transform.")
        E = self._to_epoch_cube(X)  # (S,C,T)
        # base features from your pipeline
        X_base = extract_features(E)  # (T, F)

        # CCA temporal scores per trial
        trials_avg = E.mean(axis=1)  # (S, T)
        X_time = trials_avg.T  # (T, S)
        Xc = X_time - self.x_mean_[None, :]  # center using TRAIN mean
        X_cca = Xc @ self.filters_  # (T, K)

        return np.hstack([X_base, X_cca])


# ----------  C) OOF proba helper for any estimator/pipeline ----------
def oof_proba_with_logo_pipeline(X, y, groups, estimator):
    logo = LeaveOneGroupOut()
    oof = np.zeros_like(y, dtype=float)
    for tr_idx, te_idx in logo.split(X, y, groups):
        est = clone(estimator)
        est.fit(X[tr_idx], y[tr_idx])
        oof[te_idx] = est.predict_proba(X[te_idx])[:, 1]
    return oof


def reorder_epoch_to_training(eeg_window, online_names, train_order):
    """
    eeg_window: (n_channels, n_samples)   [your current buffer orientation]
    online_names: list of those channel names (same order as eeg_window rows)
    train_order: ordered list of names used in training

    Returns: E_epoch (S, C) in training order
    """
    # we need (S, C); incoming is (C, S) → transpose then index columns
    if eeg_window.ndim != 2:
        raise ValueError(f"eeg_window must be 2D, got {eeg_window.shape}")
    S = eeg_window.shape[1]
    name_to_idx = {nm: i for i, nm in enumerate(online_names)}

    idx = []
    missing = []
    for nm in train_order:
        if nm in name_to_idx:
            idx.append(name_to_idx[nm])
        else:
            missing.append(nm)
            idx.append(None)

    # build (S, C) matrix
    E = np.zeros((S, len(train_order)), dtype=float)
    for j, k in enumerate(idx):
        if k is not None:
            # take that online channel’s samples
            E[:, j] = eeg_window[k, :]

    if missing:
        print(f"WARNING: Missing online chans padded with zeros: {missing}")

    return E  # (S, C)


def predict_one_epoch_cca(pipe, threshold, E_epoch_SxC):
    """
    E_epoch_SxC: (S, C) in TRAINING order
    """
    X_row = E_epoch_SxC.reshape(1, -1)  # (1, S*C)
    proba = pipe.predict_proba(X_row)[0, 1]  # prob of class 1 (target)
    yhat = int(proba >= threshold)
    return float(proba), yhat
