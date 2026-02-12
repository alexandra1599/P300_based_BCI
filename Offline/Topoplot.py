"""
© 2026 Alexandra Mikhael. All Rights Reserved.
"""
import config
import numpy as np
import matplotlib.pyplot as plt
import mne
from matplotlib.cm import ScalarMappable  # Add colorbar explicitly
from matplotlib.colors import Normalize  # Add colorbar explicitly
import os


def common_vlim(
    *vectors, labels=None, exclude=("M1", "M2", "T7", "T8"), symmetric=True
):
    """
    Compute shared (vmin, vmax) across multiple 1D channel vectors,
    excluding specified channels from the colorbar scaling.
    """
    if labels is not None:
        # Make a mask that keeps only non-excluded channels
        ex = {e.upper() for e in exclude}
        mask = np.array([lbl.upper() not in ex for lbl in labels])
    else:
        mask = slice(None)

    # Collect all included channel values
    vals = []
    for v in vectors:
        v = np.asarray(v).ravel()
        if isinstance(mask, np.ndarray):
            v = v[mask]
        vals.append(v)
    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]

    # Compute symmetric or full-range limits
    if symmetric:
        vmax = float(np.nanmax(np.abs(vals)))
        vmin = -vmax
    else:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))

    return vmin, vmax


def p3_mean_amplitude(
    A,
    times,
    tmin=0.25,
    tmax=0.50,
):
    """
    Function to extrat the mean amplitude in the P300 window per channel.

    Inputs :
        - A (ndarray) : EEG matrix (time x ch x trials)
        - times (vector) : Time vector in ms
        - tmin (int) : Start index of P300 window to consider (in seconds)
        - tmax (int) : End index of P300 window to consider (in seconds)

    Return a 1D vector (n_channels,) with the mean amplitude in [tmin, tmax].
    """
    A = np.asarray(A)
    t = np.asarray(times, float)

    # convert ms -> s if needed
    if t.max() > 10:
        t = t / 1000.0

    if A.ndim == 3:
        # try to figure out which axis is time
        if A.shape[0] == len(t):
            # A: (time, chan, trial)
            A_tc = A.mean(axis=2)  # -> (time, chan)
        elif A.shape[1] == len(t):
            # A: (chan, time, trial)
            A_tc = A.transpose(1, 0, 2).mean(axis=2)  # -> (time, chan)
        else:
            raise ValueError(
                f"Cannot match time axis: A.shape={A.shape}, len(times)={len(t)}"
            )
    elif A.ndim == 2:
        if A.shape[0] == len(t):
            # (time, chan)
            A_tc = A
        elif A.shape[1] == len(t):
            # (chan, time)
            A_tc = A.T  # -> (time, chan)
        else:
            raise ValueError(
                f"Cannot match time axis: A.shape={A.shape}, len(times)={len(t)}"
            )
    else:
        raise ValueError("A must be 2D or 3D (time x chan [x trial])")

    # time window mask
    w = (t >= tmin) & (t <= tmax)
    if not np.any(w):
        raise ValueError("No samples in the requested time window")

    windowed = A_tc[w, :]  # (n_win_times, n_channels)
    ga_1d = windowed.mean(axis=0)  # mean amplitude per channel in the P300 window
    ga_1d = np.where(np.isfinite(ga_1d), ga_1d, 0.0)

    return ga_1d  # shape (n_channels,)


def p3_peak_amplitude(
    A_txc_xt, times, tmin=0.25, tmax=0.50, positive=True, use_abs=False
):
    """
    Function to extrat the peak amplitude in the P300 window.

    Inputs :
        - A_txc_xt (ndarray) : EEG matrix (times x ch x trials)
        - times (vector) : Time points vector (in ms)
        - tmin (int) : Start index of P300 window to consider (in seconds)
        - tmax (int) : End index of P300 window to consider (in seconds)
        - positive (bool) : Extracted peak should be positive peak
        - use_abs (bool) : Consider both positive and negative peaks in extraction

    Return a 1D vector (n_channels,) with the peak amplitude in [tmin, tmax].
    """

    A = np.asarray(A_txc_xt)
    if A.ndim == 3:
        A = A.mean(axis=2)  # (n_times, n_channels)

    t = np.asarray(times, float)
    if t.max() > 10:  # ms -> s
        t = t / 1000.0
    w = (t >= tmin) & (t <= tmax)
    if not np.any(w):
        w = slice(None)

    windowed = A[w, :]  # (n_win_times, n_channels)
    if use_abs:
        ga_1d = np.max(np.abs(windowed), axis=0)
    else:
        ga_1d = np.max(windowed, axis=0) if positive else np.min(windowed, axis=0)
    # ga_1d = np.where(np.isfinite(ga_1d), ga_1d, 0.0)
    return ga_1d


def plot_topo(
    grand_average,
    montage,
    labels,
    task,
    data_type,
    topo,
    times=None,
    t_window=None,
    vlim=None,
    symmetric=False,
    type=None,
):
    """
    Function to get the topoplot excluding channels M1,M2,T7,T8 where noise is high
    """
    zero_channels = ("M1", "M2", "T7", "T8")

    # --- reduce to 1D per channel ---
    ga = np.asarray(grand_average)
    if t_window is not None and times is not None and ga.ndim >= 2:
        tmin, tmax = t_window
        mask = (times >= tmin) & (times <= tmax)
        if ga.ndim == 2:  # (n_ch, n_times)
            ga = ga[:, mask].mean(axis=1)
        elif ga.ndim == 3:  # (n_ch, n_times, n_trials)
            ga = ga[:, mask, :].mean(axis=(1, 2))
        else:
            raise ValueError("Unsupported shape for time-window averaging.")
    else:
        while ga.ndim > 1:
            ga = ga.mean(axis=-1)
    if ga.ndim != 1:
        raise ValueError(f"Expected 1D per-channel vector, got shape {ga.shape}")

    aliases = {"M1": "TP9", "M2": "TP10"}
    zset = {s.upper() for s in zero_channels}
    zset |= {aliases.get(s, s).upper() for s in zset}  # add TP9/TP10 if needed

    labels_up = [str(x).strip().upper() for x in labels]
    ga = ga.copy()

    for i, lab in enumerate(labels_up):
        if lab in zset:
            ga[i] = 0.0

    # --- Match data to montage ---
    ch_positions = montage.get_positions()["ch_pos"]
    montage_ch_names = list(ch_positions.keys())

    # Map data by label order
    labels_norm = [str(x).strip().upper() for x in labels]
    idx_map = {name: i for i, name in enumerate(labels_norm)}
    updated = np.array(
        [
            ga[idx_map[ch.upper()]] if ch.upper() in idx_map else 0
            for ch in montage_ch_names
        ]
    )

    # --- also zero in the montage-ordered vector (just in case names differ) ---
    mont_up = [ch.upper() for ch in montage_ch_names]
    mask_zero = np.array([m in zset for m in mont_up])
    updated[mask_zero] = 0.0

    # --- Plotting ---
    pos = np.array([ch_positions[ch][:2] for ch in montage_ch_names])
    data = updated

    if vlim is None:
        if symmetric:
            vmax = float(np.nanmax(np.abs(data)))
            vmin = -vmax
        else:
            vmin = float(np.nanmin(data))
            vmax = float(np.nanmax(data))
    else:
        vmin, vmax = vlim

    fig, ax = plt.subplots(figsize=(8, 8))
    im, _ = mne.viz.plot_topomap(
        data,
        pos,
        axes=ax,
        show=False,
        names=montage_ch_names,
        contours=0,
        cmap="coolwarm",
        extrapolate="head",
        outlines="head",
        vlim=(vmin, vmax),
    )

    # Colorbar
    norm_ = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap="coolwarm", norm=norm_)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

    label = (
        f"{'Mean' if topo == 1 else 'Peak'} amplitude in {type or 'P300'} window (µV)"
    )
    cbar.set_label(label, fontsize=12)

    # Title
    title = f"Topoplot ({type or 'P300'} Grand Average) {'Online' if task == 2 else 'Offline ' + data_type}"
    ax.set_title(title, fontsize=16)

    plt.show()

    """
    # --- name normalization + aliasing ---
    def norm(s):
        return str(s).strip().upper().replace(" ", "")

    aliases = {"M1": "TP9", "M2": "TP10"}  # extend if needed

    labels_norm = [aliases.get(norm(x), norm(x)) for x in labels]
    idx_map = {name: i for i, name in enumerate(labels_norm)}

    ch_positions = montage.get_positions()["ch_pos"]
    montage_ch_names = list(ch_positions.keys())

    updated = np.zeros(len(montage_ch_names))
    for i, ch in enumerate(montage_ch_names):
        key = aliases.get(norm(ch), norm(ch))
        updated[i] = ga[idx_map[key]] if key in idx_map else 0.0

    # --- optionally re-reference (excluding mastoids/edges) ---
    if reref:
        edge_norm = {norm(e) for e in edge_set}
        keep_for_ref = [
            i for i, ch in enumerate(montage_ch_names) if norm(ch) not in edge_norm
        ]
        if keep_for_ref:  # avoid empty slice
            ref = updated[keep_for_ref].mean()
            updated = updated - ref

    # --- optionally drop mastoids/edge channels from plotting ---
    keep_idx = list(range(len(montage_ch_names)))
    if drop_edge:
        edge_norm = {norm(e) for e in edge_set}
        keep_idx = [
            i for i, ch in enumerate(montage_ch_names) if norm(ch) not in edge_norm
        ]

    # positions (x, y) and names after dropping edge chans
    pos = np.array([ch_positions[montage_ch_names[i]][:2] for i in keep_idx])
    names = [montage_ch_names[i] for i in keep_idx]
    data = updated[keep_idx]

    # plotting
    # --- fixed color scale for comparability ---
    if vlim is None:
        if symmetric:
            vmax = float(np.nanmax(np.abs(data)))
            vmin = -vmax
        else:
            vmin = float(np.nanmin(data))
            vmax = float(np.nanmax(data))
    else:
        vmin, vmax = vlim

    fig, ax = plt.subplots(figsize=(8, 8))
    im, _ = mne.viz.plot_topomap(
        data,
        pos,
        axes=ax,
        show=False,
        names=names,
        contours=0,
        cmap="coolwarm",
        extrapolate="local",
        vlim=(vmin, vmax),  # <— key line: lock the scale
    )

    norm_ = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap="coolwarm", norm=norm_)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    if type == "P300":
        if topo == 1:  # MEAN
            cbar.set_label("Mean amplitude in P300 window (µV)", fontsize=12)
        elif topo == 2:  # PEAK
            cbar.set_label("Peak amplitude in P300 window (µV)", fontsize=12)

    elif type == "CPP":
        if topo == 1:  # MEAN
            cbar.set_label("Mean amplitude in CPP window (µV)", fontsize=12)
        elif topo == 2:  # PEAK
            cbar.set_label("Peak amplitude in CPP window (µV)", fontsize=12)

    fig.subplots_adjust(left=0.2, right=0.6, top=0.85, bottom=0.15)
    if task == 1 and data_type == "Pre":
        ax.set_title("Topoplot (P300 Grand Average) Offline Pre", fontsize=16)
    elif task == 1 and data_type == "Pos":
        ax.set_title("Topoplot (P300 Grand Average) Offline Post", fontsize=16)
    elif task == 2:
        ax.set_title("Topoplot (P300 Grand Average) Online", fontsize=16)
    plt.show()"""


def import_montage(montage_name):

    # Define the path to the .fif montage file
    montage_path = os.path.join(os.path.dirname(__file__), montage_name)

    # Load the montage
    if os.path.exists(montage_path):
        montage = mne.channels.read_dig_fif(montage_path)
        print(f"Loaded montage from: {montage_path}")
    else:
        raise FileNotFoundError(f"Montage file not found at: {montage_path}")
    return montage
