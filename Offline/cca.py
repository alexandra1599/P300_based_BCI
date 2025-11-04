import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import CCA


import numpy as np
from sklearn.cross_decomposition import CCA


import numpy as np
from sklearn.cross_decomposition import CCA

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import CCA
import numpy as np


class SpatialCCATimeExpanded(BaseEstimator, TransformerMixin):
    """
    MATLAB-style time-expanded spatial CCA:
      - Observations = (samples * trials), features = channels
      - For each class, build GA (C x S) template and repeat it
      - transform() -> per-trial CCA scores by projecting each time sample
        and averaging over time (ERP-like summary)
    """

    def __init__(self, samples, channels, n_components=3, max_iter=5000):
        self.samples = int(samples)
        self.channels = int(channels)
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.cca_ = None
        self.channel_weights_ = None  # (C, K)

    def _to_cube(self, X):
        # X: (T, S*C) -> (S, C, T)
        T = X.shape[0]
        return X.reshape(T, self.samples, self.channels).transpose(1, 2, 0)

    def fit(self, X, y):
        E = self._to_cube(X)  # (S,C,T)
        y = np.asarray(y).ravel()
        S, C, T = E.shape

        X_all, Y_all = [], []
        for cls in np.unique(y):
            m = y == cls
            if not m.any():
                continue
            Ex = E[:, :, m]  # (S,C,Tc)
            Ex_cst = np.transpose(Ex, (1, 0, 2))  # (C,S,Tc)
            GA = Ex_cst.mean(axis=2)  # (C,S)
            X_cls = Ex_cst.reshape(C, S * Ex.shape[2]).T  # (S*Tc, C)
            Y_cls = np.repeat(GA, Ex.shape[2], axis=1).T  # (S*Tc, C)
            X_all.append(X_cls)
            Y_all.append(Y_cls)

        X_te = np.concatenate(X_all, axis=0)
        Y_te = np.concatenate(Y_all, axis=0)

        cca = CCA(n_components=self.n_components, max_iter=self.max_iter, scale=True)
        cca.fit(X_te, Y_te)
        self.cca_ = cca
        self.channel_weights_ = np.asarray(cca.x_weights_)  # (C, K)
        return self

    def transform(self, X):
        if self.cca_ is None:
            raise RuntimeError("Fit before transform.")
        E = self._to_cube(X)  # (S,C,T)
        W = self.channel_weights_  # (C,K)
        T = E.shape[2]
        scores = np.empty((T, W.shape[1]), dtype=float)
        for t in range(T):
            SK = E[:, :, t] @ W  # (S,K)
            scores[t, :] = SK.mean(axis=0)
        return scores

    # Optional: channel importance
    def channel_importance(self, ch_names=None, component=0, normalize=True):
        w = self.channel_weights_[:, component]
        imp = np.abs(w)
        if normalize:
            n = np.linalg.norm(imp) or 1.0
            imp = imp / n
        if ch_names is None:
            return imp
        return sorted(zip(ch_names, imp), key=lambda z: z[1], reverse=True)


def get_cca_spatialfilter(
    dataEpochs: np.ndarray,
    dataLabels: np.ndarray,
    n_components: int = 1,
    max_iter: int = 5000,
):
    """
    Python equivalent of:
      spatialFilter = get_cca_spatialfilter(dataEpochs, dataLabels)

    Args:
        dataEpochs: (S, C, T)  - samples x channels x trials
        dataLabels: (T,)       - trial labels (binary or multiclass)
        n_components: number of canonical components
        max_iter: CCA max iterations

    Returns:
        spatialFilter: (C, n_components)  - channel-space weights (matches MATLAB's output)
        canonical_corrs: (n_components,)  - per-component canonical correlations
        cca_obj: fitted sklearn CCA object (for downstream transforms if needed)
    """
    E = np.asarray(dataEpochs)
    y = np.asarray(dataLabels).ravel()

    if E.ndim != 3:
        raise ValueError("dataEpochs must be (samples, channels, trials).")
    S, C, T = E.shape
    if y.shape[0] != T:
        raise ValueError("dataLabels length must match dataEpochs.shape[2] (trials).")

    classes = np.unique(y)
    concat_data = []  # (sum over classes of S*T_cls, C)
    concat_ga = []  # same shape

    for cls in classes:
        mask = y == cls
        if not np.any(mask):
            continue
        Ex = E[:, :, mask]  # (S, C, T_cls)
        # MATLAB: exEpochs = permute(exEpochs,[2 1 3]) -> (C, S, T)
        Ex_cst = np.transpose(Ex, (1, 0, 2))  # (C, S, T_cls)

        # grand-average over trials: (C, S)
        GA = Ex_cst.mean(axis=2)  # (C, S)

        # ex_epochs = reshape(Ex_cst, [C, S*T_cls])
        ex_epochs = Ex_cst.reshape(C, Ex_cst.shape[1] * Ex_cst.shape[2])  # (C, S*T_cls)

        # ga_data = repmat(GA, [1, T_cls]) along time*trial axis -> (C, S*T_cls)
        ga_rep = np.repeat(GA, repeats=Ex_cst.shape[2], axis=1)

        # Stack (observations = rows)
        concat_data.append(ex_epochs.T)  # (S*T_cls, C)
        concat_ga.append(ga_rep.T)  # (S*T_cls, C)

    if not concat_data:
        raise ValueError("No trials for any class.")

    X_all = np.concatenate(concat_data, axis=0)  # (N_obs, C)
    Y_all = np.concatenate(concat_ga, axis=0)  # (N_obs, C)

    max_comps = min(X_all.shape[0], X_all.shape[1], Y_all.shape[1])
    n_components = int(max(1, min(n_components, max_comps)))

    cca = CCA(n_components=n_components, max_iter=max_iter, scale=True)
    Xs, Ys = cca.fit_transform(X_all, Y_all)

    # Canonical correlations component-wise
    canonical_corrs = np.array(
        [np.corrcoef(Xs[:, k], Ys[:, k])[0, 1] for k in range(n_components)],
        dtype=float,
    )

    # Spatial filter over channels (same as MATLAB's canoncorr Wx for X variables)
    spatialFilter = np.asarray(cca.x_weights_, dtype=float)  # (C, n_components)
    return spatialFilter, canonical_corrs, cca


def spatial_cca_trial_scores(E: np.ndarray, Wc: np.ndarray) -> np.ndarray:
    """
    E: (S, C, T)
    Wc: (C, K) spatial filter from get_cca_spatialfilter
    Returns:
        scores: (T, K) per-trial component scores
    """
    S, C, T = E.shape
    K = Wc.shape[1]
    scores = np.empty((T, K), dtype=float)
    for t in range(T):
        # (S,C) @ (C,K) -> (S,K), then average over time S
        SK = E[:, :, t] @ Wc
        scores[t, :] = SK.mean(axis=0)
    return scores


import numpy as np


def rank_channels_component(Wc, ch_names=None, component=0, normalize=True):
    # Wc: (C, K)
    Wc = np.asarray(Wc)
    C, K = Wc.shape

    if component < 0 or component >= K:
        raise ValueError(f"component={component} out of range for Wc with K={K}")

    w = Wc[:, component].astype(float)
    # Replace NaNs/inf if any
    if not np.isfinite(w).all():
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    if normalize:
        n = np.linalg.norm(w)
        if n > 0:
            w = w / n

    imp = np.abs(w)

    # Ensure ch_names is an indexable, correctly-sized, ordered list
    if ch_names is None:
        names = [f"ch{c}" for c in range(C)]
    else:
        names = list(ch_names)  # accept set/tuple/etc.
        if len(names) != C:
            raise ValueError(f"len(ch_names)={len(names)} != number of channels C={C}")

    order = np.argsort(imp)[::-1]
    return [(names[i], float(imp[i])) for i in order]


import os, pickle


def load_trained_model_cca(path):
    with open(path, "rb") as f:
        saved = pickle.load(f)

    pipe = saved["model"]  # the fitted Pipeline
    thr = saved["threshold"]  # global decision threshold
    meta = saved["feature_meta"]  # dict with channels, fs, cca, etc.

    # Convenience fields
    ch_order = list(meta.get("channels", []))  # ordered list used for training
    fs = meta.get("fs", None)
    cca_info = meta.get("cca", {})
    S_tr = cca_info.get("samples", None)  # samples per epoch used in training
    C_tr = cca_info.get("channels", None)  # channels count used in training
    Wc = cca_info.get("Wc", None)  # (C, K) weights (optional debug)

    return pipe, thr, ch_order, fs, (S_tr, C_tr), meta
