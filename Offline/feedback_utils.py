import time
import config
from experiment_utils import LeakyIntegrator
import socket
import pygame
import pyautogui
from stream_utils import get_channel_from_lsl
import numpy as np
from preprocessing import (
    extract_P300_features,
)


udp_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = "127.0.0.1"
port = 12345
# Screen settings
screen_tmp = pyautogui.size()
screen_width = screen_tmp[0]
screen_height = screen_tmp[1]
screen = pygame.display.set_mode((screen_width, screen_height))


def predict_one_epoch_cca(pipe, threshold, E_epoch_SxC):
    """
    E_epoch_SxC: (S, C) in TRAINING order
    """
    X_row = E_epoch_SxC.reshape(1, -1)  # (1, S*C)
    proba = pipe.predict_proba(X_row)[0, 1]  # prob of class 1 (target)
    yhat = int(proba >= threshold)
    return float(proba), yhat


def reorder_epoch_to_training(eeg_window, live_names_for_window, train_order):
    """
    eeg_window: (n_channels, n_samples)  [rows align to live_names_for_window]
    live_names_for_window: list[str], len == eeg_window.shape[0]
    train_order: ordered list[str] used during training

    Returns: E (S, C) where S = n_samples, C = len(train_order)
             Columns are in the exact training order.
             Raises if any training channel is missing.
    """
    if eeg_window.ndim != 2:
        raise ValueError(f"eeg_window must be 2D, got {eeg_window.shape}")

    n_rows, S = eeg_window.shape[0], eeg_window.shape[1]
    if len(live_names_for_window) != n_rows:
        raise ValueError(
            f"Names length ({len(live_names_for_window)}) must match eeg_window rows ({n_rows})."
        )

    name_to_idx = {nm: i for i, nm in enumerate(live_names_for_window)}

    missing = [nm for nm in train_order if nm not in name_to_idx]
    if missing:
        raise ValueError(
            f"Missing required training channels in live window: {missing}"
        )

    idx = [name_to_idx[nm] for nm in train_order]  # exact training order

    # Build (S, C)
    E = np.empty((S, len(train_order)), dtype=float)
    for j, k in enumerate(idx):
        E[:, j] = eeg_window[k, :]

    return E


import numpy as np


def build_trial_features(
    eeg_window, online_channel_names, train_channels, fs, expected_feats=None
):
    """
    Build a single-trial feature row for online inference.

    Args
    ----
    eeg_window : np.ndarray
        Shape (n_channels, n_samples) for THIS window. Row order must match online_channel_names.
    online_channel_names : list[str]
        Names for the rows in eeg_window, in order.
    train_channels : list[str] or None
        Ordered list of channel names used during training. If provided, we reorder to this list
        and zero-pad any missing channels. If None, we fallback to expected_feats logic.
    fs : float
        Sampling rate used by the feature extractor.
    expected_feats : int or None
        Total feature length the model expects (e.g., model.n_features_in_).
        Used only when train_channels is None.

    Returns
    -------
    feats_row : np.ndarray
        Shape (1, N * Fpc), where:
          - N = len(train_channels) if provided, else inferred from expected_feats
          - Fpc = features-per-channel from the extractor
    """
    # --- 0) sanity checks ---
    if eeg_window.ndim != 2:
        raise ValueError(
            f"eeg_window must be 2D (n_channels, n_samples), got {eeg_window.shape}"
        )
    if not isinstance(online_channel_names, (list, tuple)):
        online_channel_names = list(online_channel_names)

    n_ch, _ = eeg_window.shape
    if len(online_channel_names) != n_ch:
        print(
            f"WARNING: online_channel_names ({len(online_channel_names)}) "
            f"!= eeg_window rows ({n_ch}). Using min() and proceeding."
        )

    # --- 1) per-channel features ( extractor expects (samples, channels)) ---
    feats_per_ch = extract_P300_features(eeg_window.T, fs=fs)  # (n_ch, Fpc)
    if feats_per_ch.ndim != 2:
        raise ValueError(
            f"Extractor must return 2D (n_ch, Fpc), got {feats_per_ch.shape}"
        )
    Fpc = feats_per_ch.shape[1]

    # --- 2) helper: canonicalize names for robust matching ---
    def canon(name: str) -> str:
        return name.replace(" ", "").upper()

    # --- 3) If we have training channels, reorder to that exact list (pad missing with zeros) ---
    if train_channels is not None and len(train_channels) > 0:
        # map live channel name -> row index in feats_per_ch
        live_map = {canon(ch): i for i, ch in enumerate(online_channel_names)}

        # build stacked features in the TRAIN order
        stacked = np.zeros((len(train_channels), Fpc), dtype=float)
        missing = []
        for out_i, ch in enumerate(train_channels):
            key = canon(ch)
            idx = live_map.get(key, None)
            # guard against None and out-of-range indices
            if idx is None or not (0 <= idx < feats_per_ch.shape[0]):
                missing.append(ch)
                # keep zeros for this channel
            else:
                stacked[out_i, :] = feats_per_ch[idx, :]
        if missing:
            print(f"WARNING: Missing channels in live stream (zero-padded): {missing}")
        return stacked.reshape(1, -1)

    # --- 4) No training channel list: fall back to expected_feats (trim/pad) ---
    if expected_feats is None:
        # last resort: return everything we have (may mismatch model)
        return feats_per_ch.reshape(1, -1)

    # Determine how many channels the model expects given Fpc
    if expected_feats % Fpc != 0:
        print(
            f"WARNING: expected_feats={expected_feats} not divisible by Fpc={Fpc}. "
            f"Proceeding by truncation/padding; verify your extractor matches training."
        )
    expected_ch = expected_feats // Fpc

    if n_ch >= expected_ch:
        stacked = feats_per_ch[:expected_ch, :]
    else:
        pad = np.zeros((expected_ch - n_ch, Fpc), dtype=float)
        stacked = np.vstack([feats_per_ch, pad])
        print(f"WARNING: {expected_ch - n_ch} channels missing; zero-padded.")

    return stacked.reshape(1, -1)


# Function to draw fixation cross
def draw_fixation_cross(duration, color):
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < duration:
        screen.fill(config.BLACK)
        center = (screen_width // 2, screen_height // 2)
        line_width = 5
        pygame.draw.line(
            screen,
            color,
            (center[0] - 20, center[1]),
            (center[0] + 20, center[1]),
            line_width,
        )
        pygame.draw.line(
            screen,
            color,
            (center[0], center[1] - 20),
            (center[0], center[1] + 20),
            line_width,
        )
        pygame.display.flip()


def send_udp_message(socket, ip, port, message):
    """
    Send a UDP message to the specified IP and port.
    Parameters:
        socket (socket.socket): The socket object for communication.
        ip (str): The target IP address.
        port (int): The target port.
        message (str): The message to send.
    """
    socket.sendto(message.encode("utf-8"), (ip, port))
    print(f"Sent UDP message to {ip}:{port}: {message}")


def draw_fixation_cross_frame(color):
    screen.fill(config.BLACK)
    cx, cy = screen_width // 2, screen_height // 2
    pygame.draw.line(screen, color, (cx - 20, cy), (cx + 20, cy), 5)
    pygame.draw.line(screen, color, (cx, cy - 20), (cx, cy + 20), 5)
    pygame.display.flip()


def _canon(name: str) -> str:
    # Robust canonical form: case-insensitive, remove spaces
    return name.replace(" ", "").upper()


def restrict_eeg_to_channels(eeg_window, online_names_for_window, want_names):
    """
    eeg_window: (n_ch, n_samples)
    online_names_for_window: list[str] for those n_ch rows
    want_names: list[str] target order (e.g., ["Pz","Cz","Fz","P3","P4","POz"])
    """
    if not isinstance(online_names_for_window, (list, tuple)):
        raise ValueError(
            "online_names_for_window must be a list aligned to eeg_window rows."
        )

    n_rows = eeg_window.shape[0]
    if len(online_names_for_window) != n_rows:
        print(
            f"WARNING: row/name length mismatch: rows={n_rows}, names={len(online_names_for_window)}. "
            "Adjusting names to match rows."
        )
        # heuristic: if more names than rows, take the first rows; if fewer, pad dummy names
        if len(online_names_for_window) > n_rows:
            online_names_for_window = list(online_names_for_window[:n_rows])
        else:
            online_names_for_window = list(online_names_for_window) + [
                f"ch{i}" for i in range(n_rows - len(online_names_for_window))
            ]

    S = eeg_window.shape[1]

    def _canon(name: str) -> str:
        return name.replace(" ", "").upper()

    name2idx = {_canon(nm): i for i, nm in enumerate(online_names_for_window)}

    eeg_sel = np.zeros((len(want_names), S), dtype=eeg_window.dtype)
    missing = []
    for out_i, nm in enumerate(want_names):
        k = name2idx.get(_canon(nm))
        if k is None:
            missing.append(nm)  # keep zeros for this channel
        else:
            eeg_sel[out_i, :] = eeg_window[k, :]
    if missing:
        print(f"WARNING: Missing channels (zero-padded): {missing}")
    return eeg_sel, list(want_names)


def classify_epoch_once(
    eeg_state,
    window_size_samples,
    xgb_model,
    expected_feats,
    mode,
    train_channels,
    online_channel_names,  # MUST be the names aligned to eeg_window rows
):
    # 1) pull the most recent window
    eeg_window_all, _ = eeg_state.get_baseline_corrected_window(
        window_size_samples
    )  # (n_ch, n_samples)

    # 2) select only the P300 channels in the target order
    want = config.P300_CHANNEL_NAMES
    eeg_window, sel_names = restrict_eeg_to_channels(
        eeg_window_all, online_channel_names, want
    )

    # 3) features (stacks by training order if provided; else auto-trim/pad to expected_feats)
    feats_flat = build_trial_features(
        eeg_window,  # (6, samples)
        sel_names,  # ["Pz","Cz","Fz","P3","P4","POz"]
        train_channels,  # pass SAME list if model was trained on these 6
        fs=config.FS,
        expected_feats=expected_feats,
    )
    # sanity: model expects exactly expected_feats (e.g., 48 = 6*Fpc)
    if feats_flat.shape[1] != expected_feats:
        raise ValueError(
            f"Feature shape mismatch: got {feats_flat.shape[1]}, expected {expected_feats}."
        )

    # 4) predict and threshold P(target) once
    probs = xgb_model.predict_proba(feats_flat)[0]
    classes = np.asarray(xgb_model.classes_)  # expect [0,1]
    idx_non = int(np.where(classes == 0)[0][0])
    idx_tar = int(np.where(classes == 1)[0][0])

    p_non, p_tar = float(probs[idx_non]), float(probs[idx_tar])
    prediction = int(p_tar >= config.THRESHOLD)  # 1=Target, 0=Non-target
    conf = p_tar
    all_probs = [[time.time(), p_non, p_tar]]
    return prediction, conf, all_probs


def get_baseline_fixation(inlet, countdown_start, countdown_dur, baseline_buffer):
    """
    Monitors countdown time and collects baseline EEG data during last 0.5s

    Parameters:
            inlet: LSL EEG stream inlet
            countdown_start: Start time of countdown (pygame ticks)
            countdown_dur: Total countdown duration (ms)
            baseline_buffer: List to store baseline EEG data

    Returns:
            Updated baseline_buffer containing baseline samples
    """

    pygame.display.flip()
    current_time = pygame.time.get_ticks()
    remaining_time = countdown_dur - (current_time - countdown_start)

    if remaining_time <= 1000 and not baseline_buffer:
        print("Flushing buffer and collecting baseline data . . . ")
        inlet.flush()  # Remove old EEG data

    if remaining_time <= 1000:  # Collect EEG data continuously
        new_data, _ = inlet.pull_chunk(
            timeout=0.1, max_samples=config.FS // 2
        )  # 0.5s worth of samples
        if new_data:
            baseline_buffer.extend(new_data)

    return baseline_buffer
