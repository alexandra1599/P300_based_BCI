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


def build_trial_features(
    eeg_window, online_channel_names, train_channels, fs, expected_feats=None
):
    """
    eeg_window: (n_channels, n_samples) for THIS window (rows must match online_channel_names)
    online_channel_names: list[str] length == n_channels
    train_channels: ordered list[str] used at training; may be None
    expected_feats: model.n_features_in_ (e.g., 312). Used when train_channels is None.

    Returns (1, N*8) where N = len(train_channels) if provided, else expected_feats//8.
    """
    if eeg_window.ndim != 2:
        raise ValueError(
            f"eeg_window must be 2D (n_channels, n_samples), got {eeg_window.shape}"
        )
    n_ch, _ = eeg_window.shape
    if online_channel_names is None or len(online_channel_names) != n_ch:
        raise ValueError(
            f"online_channel_names length ({0 if online_channel_names is None else len(online_channel_names)}) "
            f"!= eeg_window rows ({n_ch}). Use eeg_state.channel_names."
        )

    # 1) per-channel features (extractor expects (samples, channels))
    feats_per_ch = extract_P300_features(eeg_window.T, fs=fs)  # -> (n_ch, 8)
    if feats_per_ch.shape != (n_ch, 8):
        raise ValueError(
            f"Unexpected feature shape {feats_per_ch.shape} for {n_ch} channels"
        )

    # 2) if we have training channels, reorder to that exact list (and pad missing with zeros)
    def canon(name: str) -> str:
        return name.replace(" ", "").upper()

    if train_channels.all():
        online_map = {
            canon(ch): feats_per_ch[i] for i, ch in enumerate(online_channel_names)
        }
        stacked = []
        missing = []
        for ch in train_channels:
            v = online_map.get(canon(ch))
            if v is None:
                missing.append(ch)
                v = np.zeros(8, dtype=float)
            stacked.append(v)
        if missing:
            print(f"WARNING: Missing channels (padded with zeros): {missing}")
        return np.vstack(stacked).reshape(1, -1)

    # 3) no training metadata -> trim/pad to model’s expected size
    if expected_feats is None:
        # last resort: use what we have (may still mismatch the model)
        return feats_per_ch.reshape(1, -1)

    expected_ch = expected_feats // 8
    if n_ch >= expected_ch:
        stacked = feats_per_ch[:expected_ch]
    else:
        pad = np.zeros((expected_ch - n_ch, 8), dtype=float)
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


def show_feedback(duration, expected_feats, mode=0, eeg_state=None):
    """
    Displays feedback animation, collects EEG data, performs real-time classification using a sliding window approach
    """

    start_time = time.time()
    step_size = config.STEP_SIZE  # Slisding window step size (sec)
    window_size = config.CLASSIFY_WINDOW / 1000  # Convert ms to s
    window_size_samples = int(window_size * config.FS)
    step_size_samples = int(step_size * config.FS)
    global filter_states

    all_probs = []
    preds = []
    data_buffer = []
    leaky_integrator = LeakyIntegrator(
        alpha=config.INTEGRATOR_ALPHA
    )  # Confidence smoothing
    min_preds = config.MIN_PREDICTIONS

    classification_results = []
    correct_class = 1 if mode == 11 else 0
    incorrect_class = 0 if mode == 11 else 1

    # accuracy threshold
    accuracy_threshold = (
        config.THRESHOLD_TARGET if mode == 11 else config.THRESHOLD_NONTARGET
    )

    # Send UDP triggers
    if mode == 11:  # target trial
        send_udp_message(
            udp_marker,
            config.UDP_MARKER["IP"],
            config.UDP_MARKER["PORT"],
            config.TRIGGERS["MATCH"],
        )
    else:
        send_udp_message(
            udp_marker,
            config.UDP_MARKER["IP"],
            config.UDP_MARKER["PORT"],
            config.TRIGGERS["NON_MATCH"],
        )

    clock = pygame.time.Clock()
    running_avg_confidence = 0.5  # Initial placeholder

    while time.time() - start_time < duration:

        eeg_state.update()
        current_confidence, preds, all_probs = classify_real_time(
            eeg_state,
            window_size_samples,
            step_size_samples,
            all_probs,
            preds,
            mode,  # <-- correct
            leaky_integrator,  # <-- correct
            baseline_mean=None,  # or your computed baseline
            expected_feats=expected_feats,
        )

        if all_probs:
            _, p_non, p_tar = all_probs[-1]
            send_udp_message(
                udp_marker,
                ip,
                port,
                f"{config.TRIGGERS['TARGET_PROBS' if mode == 11 else 'NON_TARGET_PROBS']}, {p_tar:.5f},{p_non:.5f}",
            )

        running_avg_confidence = leaky_integrator.update(current_confidence)

        screen.fill(config.BLACK)

        if (
            mode == 11 and running_avg_confidence > config.THRESHOLD_TARGET
        ):  # target trial and decoder got P300
            draw_fixation_cross(1.5, config.GREEN)  # P300 detected during target trial
            # display_text(f'P300 detected ! Correct!', font, config.green, (screen_width // 2, screen_height // 2), 3)
        elif mode == 11 and running_avg_confidence < config.THRESHOLD_TARGET:
            draw_fixation_cross(
                1.5, config.red
            )  # P300 not detected during target trial
            # display_text(f'P300 detected ! Incorrect!', font, config.red, (screen_width // 2, screen_height // 2), 3)

        if (
            mode == 12 and running_avg_confidence > config.THRESHOLD_NONTARGET
        ):  # non-target trial and decoder did not detect P300
            draw_fixation_cross(
                1.5, config.GREEN
            )  # P300 not detected during non target trial
            # display_text(f'P300 not detected ! Correct!', font, config.green, (screen_width // 2, screen_height // 2), 3)
        elif (
            mode == 12 and running_avg_confidence < config.THRESHOLD_NONTARGET
        ):  # non-target trial and decoder did detect P300
            draw_fixation_cross(
                1.5, config.red
            )  # P300 detected during non target trial
            # display_text(f'P300 detected ! Incorrect!', font, config.red, (screen_width // 2, screen_height // 2), 3)

        pygame.display.flip()

    # Final Decision
    if (mode == 11 and running_avg_confidence > config.THRESHOLD_TARGET) or (
        mode == 12 and running_avg_confidence > config.THRESHOLD_NONTARGET
    ):
        final_class = correct_class
    else:
        final_class = incorrect_class

    if final_class is not None:
        print(
            f"Final decision: {final_class}, Confidence for correct({correct_class}) class :"
            f"{running_avg_confidence:.2f}, at sample size {len(preds)}"
        )

    else:
        print(f"No threshold met. Confidence: {running_avg_confidence:.2f} ")

    return (
        final_class,
        running_avg_confidence,
        leaky_integrator,
        data_buffer,
        all_probs,
    )


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


def classify_real_time(
    eeg_state,
    window_size,
    step_size,
    all_probs,
    preds,
    mode,
    leaky_integrator,
    baseline_mean,
    expected_feats,
    xgb_model,
):
    try:
        eeg_window, _ = eeg_state.get_baseline_corrected_window(window_size)
    except ValueError:
        return mode, preds, all_probs

    # new_data, _ = inlet.pull_chunk(timeout=0.1, max_samples=int(step_size))

    pygame.display.flip()
    pygame.event.get()

    # if eeg_window:
    """new_data_np = np.array(new_data)
    data_buffer.extend(new_data_np)

    if len(data_buffer) < config.FS:
            return (
                leaky_integrator.accumulated_probabilities,
                preds,
                all_probs,
                data_buffer,
            )

        eeg_window = np.array(data_buffer[-window_size:]).T"""
    sfreq = config.FS
    channel_names = get_channel_from_lsl("EEG")

    online_channel_names = {"Pz", "P1", "P2", "CPz"}
    """
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq)
    raw = mne.io.RawArray(eeg_window, info)

    aux_channels = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
    existing_aux = [ch for ch in aux_channels if ch in raw.ch_names]
    if existing_aux:
        raw.drop_channels(existing_aux)
        print("Dropped AUX channels")

    rename_dict = {
        "FP1": "Fp1",
        "FPZ": "Fpz",
        "FP2": "Fp2",
        "FZ": "Fz",
        "CZ": "Cz",
        "PZ": "Pz",
        "POZ": "POz",
        "OZ": "Oz",
    }
    raw.rename_channels(rename_dict)

    mastoid_channels = ["M1", "M2"]
    existing_mastoids = [ch for ch in mastoid_channels if ch in raw.ch_names]
    if existing_mastoids:
        raw.drop_channels(existing_mastoids)
        print("Dropped M1,M2 channels")

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=True, on_missing="warn")

    for ch in raw.info["chs"]:
        ch["unit"] = 201  # uV

    raw.data = filtering(raw, session, labels=None, fs=sfreq, do_car=False)
    print("BP filtered in [{config.LOWCUT},{config.HIGHCUT}]")

    # Baseline Correction
    raw._data -= baseline_mean
    print("Baseline Correction applied")

    eeg_data = raw.get_data()"""
    channel_names = np.delete(channel_names, np.arange(32, 39), axis=0)

    feats_flat = build_trial_features(
        eeg_window,
        channel_names,
        channel_names,  # this may be None
        fs=config.FS,
        expected_feats=expected_feats,
    )

    print("Feature shape : ", {feats_flat.shape})
    print("n_online_ch:", 0 if channel_names is None else len(channel_names))
    print("expected_feats:", expected_feats)
    print("feats_flat.shape:", feats_flat.shape)

    # Classification
    probs = xgb_model.predict_proba(feats_flat)[0]
    classes = np.asarray(xgb_model.classes_)  # e.g., array([0, 1]) or array([1, 0])

    # Map indices
    try:
        idx_non = int(np.where(classes == 0)[0][0])
        idx_tar = int(np.where(classes == 1)[0][0])
    except IndexError:
        raise ValueError(f"Model classes {classes.tolist()} must contain 0 and 1.")

    # Optional: model’s predicted label (not strictly needed for thresholding path)
    predicted = int(classes[np.argmax(probs)])

    # Log and slide buffer  (always store probs with consistent meaning)
    preds.append(predicted)
    all_probs.append(
        [time.time(), probs[idx_non], probs[idx_tar]]
    )  # [ts, P(non), P(tar)]

    # Confidence of the “correct” class for this trial mode
    correct_class = 1 if mode == 11 else 0
    idx_correct = idx_tar if correct_class == 1 else idx_non
    current_confidence = float(probs[idx_correct])

    print(
        f"classes={classes.tolist()} | P(non)={probs[idx_non]:.3f} | P(tar)={probs[idx_tar]:.3f} | "
        f"mode={'Target' if mode==11 else 'Non-target'} | conf(correct)={current_confidence:.3f}"
    )

    return current_confidence, preds, all_probs
