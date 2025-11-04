import pygame
import socket
import time
import pickle
import datetime
import os
import random
import pyautogui
from pylsl import StreamInlet, resolve_stream
import numpy as np
from collections import deque
from sklearn.base import BaseEstimator, TransformerMixin, clone
from extract_features import extract_features

from feedback_utils import (
    draw_fixation_cross,  # keep for the post-decision 1.5s display
    draw_fixation_cross_frame,  # NEW (non-blocking frame)
    classify_epoch_once,  # NEW (one-shot classifier)
)

from logger import log_trial_prediction

# Stream Utilites
from stream_utils import get_channel_from_lsl

# Configuration Parameters
import config

# Performance Evaluation (Classification Metrics)
from sklearn.metrics import confusion_matrix


from pathlib import Path
from load_model import (
    get_n_features_in,
    load_trained_model,
)
from display import (
    display_text,
)

from XDawn import (
    classify_epoch_once_xdawn,
)

# MNE for real time EEG processing
import mne

mne.set_log_level("WARNING")

from EEGStreamState import EEGStreamState
from cca import get_cca_spatialfilter, rank_channels_component, load_trained_model_cca


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


def feature_adapter(Y):
    # If extract_features returns 3D, force (T, F):
    Z = extract_features(Y)  # must produce per-trial features
    if Z.ndim == 3:  # e.g., (S', C', T)
        Z = Z.reshape(-1, Z.shape[-1]).T  # (T, S'*C')
    return Z  # must be (T, F)


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


def load_trained_model_cca_with_expected_feats(path):
    pipe, thr, ch_order, fs, (S_tr, C_tr), meta = load_trained_model_cca(path)

    expected_feats = meta.get("n_features")

    if expected_feats is None:
        # try classifier
        clf = getattr(pipe, "named_steps", {}).get("clf", None)
        if clf is not None and hasattr(clf, "n_features_in_"):
            expected_feats = int(clf.n_features_in_)
        elif hasattr(pipe, "n_features_in_"):
            expected_feats = int(pipe.n_features_in_)

    # FINAL fallback for CCA pipelines: just use input size S*C
    if expected_feats is None and S_tr and C_tr:
        expected_feats = int(S_tr) * int(C_tr)

    return pipe, thr, ch_order, fs, (S_tr, C_tr), meta, expected_feats


# Networking Utilities
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


# What session is it ? Relax or tACS
session_input = input("Session (1 = Relaxation, 2= tACS_: ").strip()
session_number = input("Session number :")
session_map = {"1": "Relaxation", "2": "tACS"}
session = session_map.get(session_input, session_input)
choice = int(
    input("Feature extractor:\n [1] CCA\n [2] Handcrafted (No CCA)\n [3] Xdawn\n> ")
)

pygame.init()

# Screen settings
screen_tmp = pyautogui.size()
screen_width = screen_tmp[0]
screen_height = screen_tmp[1]


screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Online N-back Task")
info = pygame.display.Info()
print("Pygame initialized and display configured.")

# Set up fonts
font = pygame.font.SysFont("Arial", 100)
letter_font = pygame.font.SysFont("Arial", 150)
small_font = pygame.font.SysFont("Arial", 100)

# Setup UDP
udp_marker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
message1 = "0"  # "Trial Start"
message2 = "100"  # "Button press match"
message3 = "200"  # "Button press no-match"
message4 = "300"  # "Timeout"
message5 = "400"  # "Trial End"
message6 = "1"
message7 = "2"
message8 = "11"  # match
message9 = "12"  # non match
message10 = "33"  # Correct Prediction
message11 = "44"  # "Incorrect Prediction"
ip = "127.0.0.1"
port = 12345


def reorder_to_trained(eeg_window_CxS, online_names, trained_order):
    name2i = {ch: i for i, ch in enumerate(online_names)}
    try:
        idx = [name2i[ch] for ch in trained_order]  # KeyError if any missing
    except KeyError as e:
        missing = [ch for ch in trained_order if ch not in name2i]
        raise ValueError(f"Missing channels online: {missing}") from e
    return eeg_window_CxS[idx, :]


# global pipe, threshold, train_channels, fs_tr

# ===Load model===
if choice == 1:
    subject_model_dir = "/home/alexandra-admin/Documents/saved_models_cca"
    subject_model_path = os.path.join(
        subject_model_dir, f"p300_model_cca_sub-{config.TRAINING_SUBJECT}.pkl"
    )

    print("Loading:", subject_model_path)
    pipe, threshold, ch_order, fs, (S_tr, C_tr), meta, expected_feats = (
        load_trained_model_cca_with_expected_feats(subject_model_path)
    )

    # ✅ Use the EXACT saved training order (LIST), do NOT overwrite with a set
    train_channels = list(ch_order)  # ensure it's a list

    if not train_channels:
        # try feature_meta channels
        fm = meta.get("feature_meta") if isinstance(meta, dict) else None
        ch_meta = None if fm is None else fm.get("channels")
        if ch_meta:
            train_channels = list(ch_meta)
        else:
            raise RuntimeError("No saved channel order found in the model bundle.")

    """
    # Ensure correct shape
    if X_live.ndim == 1:
        X_live = X_live.reshape(1, -1)
    if X_live.shape[1] != expected_feats:
        raise ValueError(f"Feature length mismatch: got {X_live.shape[1]}, expected {expected_feats}")

    p = pipe.predict_proba(X_live)[:, 1]
    y_hat = (p >= thr).astype(int)

    print(
        "Loaded CCA pipeline. Expected epoch:",
        (S_tr, C_tr),
        "Train channels:",
        train_channels,
    )"""


elif choice == 2:
    subject_model_dir = "/home/alexandra-admin/Documents/saved_models/"

    subject_model_path = os.path.join(
        subject_model_dir, f"p300_model_sub-{config.TRAINING_SUBJECT}.pkl"
    )

    print("Loading:", subject_model_path)
    try:
        xgb_model, threshold, scaler, train_channels, feature_meta = load_trained_model(
            subject_model_path
        )
        print(f"Model successfully loaded from: {subject_model_path}")
    except FileNotFoundError:
        print("ERROR: Model file not found. Ensure the model has been trained.")
        raise

    expected_feats = get_n_features_in(xgb_model)
    print("This model was trained on . . .")

elif choice == 3:  # XDAWN
    subject_model_dir = "/home/alexandra-admin/Documents/saved_models_xdawn/"
    subject_model_path = os.path.join(
        subject_model_dir, f"p300_model_xdawn_sub-{config.TRAINING_SUBJECT}.pkl"
    )

    print("Loading:", subject_model_path)

    with open(subject_model_path, "rb") as f:
        bundle = pickle.load(f)

    # Optional: unpack if you still want local vars, but always keep `bundle`
    pipe = bundle["pipe"]
    threshold = float(bundle["threshold"])
    train_channels = bundle.get("train_channels", [])
    sfreq = int(round(bundle["sfreq"]))
    featurizer = bundle.get("featurizer", {})


eeg_dir = os.path.join(
    config.DATA_DIR, f"sub-P{config.TRAINING_SUBJECT}", "training_data"
)


print(f"Script is looking for XDF files in: {eeg_dir}")

print("This model was trained on . . .")
eeg_dir = os.path.join(
    config.DATA_DIR, f"sub-P{config.TRAINING_SUBJECT}", "training_data"
)
print(f"Script is looking for XDF files in: {eeg_dir}")

xdf_files = [
    os.path.join(eeg_dir, f)
    for f in os.listdir(eeg_dir)
    if f.endswith(".xdf") and "OBS" not in f
]

if not xdf_files:
    raise FileNotFoundError(f"No XDF files found in: {eeg_dir}")

print(f"training data: {xdf_files}")

# Initialize runtime structures
pred_list = []
true_list = []

fs = config.FS
filtered_buffer = deque(maxlen=2048)


SID = config.TRAINING_SUBJECT
SESSION_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(f"Session timestamp set: {SESSION_TIMESTAMP}")

# --- where to store logs ---
log_dir = Path(config.OUTPUT_ONLINE_DIR) / f"sub-{SID}" / "predictions" / session_number
log_dir.mkdir(parents=True, exist_ok=True)

log_path = log_dir / f"predictions_sub-{SID}_{session_number}.txt"

# --- create file with header (once) ---
with log_path.open("w", encoding="utf-8") as f:
    f.write(SESSION_TIMESTAMP)
    f.write("timestamp,run,trial,label,prob,pred,threshold\n")

print(f"🔎 Logging predictions to: {log_path}")


# List of possible letters
letters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# N-back level
N = 2  # Change to your desired N level


# Function to generate N-back sequence correctly
def generate_nback_sequence(total_trials, target_ratio, N):
    numTargets = round(target_ratio * total_trials)
    sequence = []
    target_indices = set(
        random.sample(range(N, total_trials), numTargets)
    )  # Pick where matches should be

    for i in range(total_trials):
        if i in target_indices:
            letter = sequence[i - N]  # Match N-back letter
        else:
            # Ensure no accidental match
            letter = random.choice(
                [l for l in letters if i < N or l != sequence[i - N]]
            )

        sequence.append(letter)

    print("\nGenerated Sequence:", sequence)
    print("Target Indices:", target_indices)  # Debug: See where matches occur
    return sequence, target_indices


def main():

    # MAIN GAME LOOP
    global mode
    mode = 12
    print("Resolving EEG data stream via LSL ...")
    streams = resolve_stream("type", "EEG")
    inlet = StreamInlet(streams[0])
    print("✅ EEG stream detected and inlet established.")

    # Initialize EEG handler
    eeg_h = EEGStreamState(inlet=inlet, config=config, mode="p300", logger=None)
    print("✅ EEGStreamState object created - ready to pull and process data")

    # Generate and log trial sequence
    global total_trials
    total_trials = 50
    target_ratio = 0.3  # 30% of trials should be matches
    sequence, target_indices = generate_nback_sequence(total_trials, target_ratio, N)
    print(f"Trial sequence generated: {sequence}")
    target_idx = sorted(target_indices)
    mode_labels = [
        "Target" if t in target_idx else "NonTarget" for t in range(len(sequence))
    ]
    print(f"Trial Sequence (labeled): {mode_labels}")
    global correct_responses
    correct_responses = 0
    trial = 0

    # Fetch and log channel names from stream
    channel_names = get_channel_from_lsl()
    print(f"Channel names detected in LSL stream: {channel_names}")

    # Load 10-20 montage and rename for MNE
    montage = mne.channels.make_standard_montage("standard_1020")
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

    # Filter for EEG-only channels
    non_eeg_channels = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
    valid_eeg_channels = [ch for ch in channel_names if ch not in non_eeg_channels]
    valid_indices = [channel_names.index(ch) for ch in valid_eeg_channels]
    print(f"Got EEG channels (excluding AUX/Trigger): {valid_eeg_channels}")

    # Initialize MNE Raw object for online data structure
    sfreq = config.FS
    info = mne.create_info(ch_names=valid_eeg_channels, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((len(valid_eeg_channels), 1)), info)

    # Apply montage and unit conversion
    raw.rename_channels(rename_dict)
    raw.set_montage(montage, match_case=True, on_missing="warn")
    for ch in raw.info["chs"]:
        ch["unit"] = 201  # Set unit to uV

    print(
        f"✅Applied 10-20 montage and prepared Raw Object. Final Channels: {raw.ch_names}"
    )

    # initialize data buffer and experiment state
    all_results = []
    running = True
    clock = pygame.time.Clock()

    # --- Prefill EEG buffer so Trial 1 has enough samples ---
    print("Prefilling EEG buffer…")
    prefill_sec = max(1.0, (config.CLASSIFY_WINDOW / 1000.0))
    prefill_samples = int(prefill_sec * config.FS)
    t0 = time.perf_counter()
    while True:
        eeg_h.update()
        try:
            eeg_h.get_baseline_corrected_window(prefill_samples)
            break
        except ValueError:
            if time.perf_counter() - t0 > 2.0:
                break
            time.sleep(0.01)

    # Display "Press any key to start"
    display_text(
        "Press any key to start",
        font,
        config.WHITE,
        (screen_width // 2, screen_height // 2),
        0,
    )
    waiting_for_start = True
    while waiting_for_start:
        eeg_h.update()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                waiting_for_start = False
        time.sleep(0.01)

    print("✅Started the task")

    while running and trial < total_trials:
        print(f"--- Trial {trial+1}/{len(sequence)} START ---")
        letter = sequence[trial]
        if trial >= 0:  # Ensure at least one previous trial exists

            # Mark trial type and start
            is_match = trial in target_indices
            if is_match:
                send_udp_message(udp_marker, ip, port, message8)
                mode = 11
            else:
                send_udp_message(udp_marker, ip, port, message9)
                mode = 12
                send_udp_message(udp_marker, ip, port, message1)  # "Trial Start"

            # Show digit and record onset
            letter_onset = time.perf_counter()
            global response
            response = None

            while True:
                now = time.perf_counter()
                elapsed = now - letter_onset

                # Show digit for 0.5 s, then fixation (single frame; no blocking)
                if elapsed < 0.5:
                    screen.fill(config.BLACK)
                    text_surface = letter_font.render(letter, True, config.WHITE)
                    text_rect = text_surface.get_rect(
                        center=(screen_width // 2, screen_height // 2)
                    )
                    screen.blit(text_surface, text_rect)
                    pygame.display.flip()
                else:
                    draw_fixation_cross_frame(config.WHITE)

                # Keep EEG buffer fresh
                eeg_h.update()

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                    if (
                        event.type == pygame.MOUSEBUTTONDOWN
                        and response is None
                        # and elapsed > 0.3
                    ):
                        """ "
                        if event.key == pygame.K_e:
                            pygame.quit()
                            exit()"""
                        if event.button in (3, 2):
                            response = "y"
                            send_udp_message(udp_marker, ip, port, message2)
                        elif event.button == 1:
                            response = "n"
                            send_udp_message(udp_marker, ip, port, message3)

                # Stop at key or timeout (1.1 s)
                if response is not None or elapsed >= 1.1:
                    break

                time.sleep(0.01)  # reduce CPU

            press_ts = time.perf_counter()
            if response is None:
                response = "timeout"
                send_udp_message(udp_marker, ip, port, message4)

            # Score the *behavioral* response (independent of EEG)
            if response != "timeout":
                if (response == "y" and is_match) or (response == "n" and not is_match):
                    correct_responses += 1
                    send_udp_message(udp_marker, ip, port, message6)  # "correct key"
                else:
                    send_udp_message(udp_marker, ip, port, message7)  # "incorrect key"

        trial += 1

        # Show feedback and perform classification
        print(
            f"Starting feedback classification - Mode: {'Target' if mode == 11 else 'Non-target'}"
        )
        if choice == 2:  # Handcrafted

            prediction = []
            duration_sec = press_ts - letter_onset if response != "timeout" else 1.1
            duration_sec = min(
                max(duration_sec, 0.20), 1.20
            )  # clamp [0.20s, 1.20s] if you like
            window_size_samples = int(round(duration_sec * config.FS))

            # Use the channel names fetched earlier; ensure it's a LIST
            online_channel_names = channel_names

            prediction, confidence, trial_probs = classify_epoch_once(
                eeg_state=eeg_h,
                window_size_samples=window_size_samples,
                xgb_model=xgb_model,
                expected_feats=expected_feats,
                mode=mode,
                train_channels=config.P300_CHANNEL_NAMES,  # from loaded model
                online_channel_names=config.P300_CHANNEL_NAMES,
            )

            true_label = 1 if mode == 11 else 0  # 1=Target, 0=Non-target
            is_correct = prediction == true_label

            pred_list.append(prediction)
            true_list.append(true_label)

        elif choice == 3:  # XDAWN
            prediction, confidence, trial_probs = classify_epoch_once_xdawn(
                eeg_state=eeg_h,
                bundle=bundle,
                online_channel_names=bundle["train_channels"],
                mode=mode,
            )

            true_label = 1 if mode == 11 else 0
            is_correct = prediction == true_label
            pred_list.append(prediction)
            true_list.append(true_label)

        # Log and end-of-trial marker
        print(
            f"Classification result - Predicted: {prediction}, Ground Truth: {1 if mode == 11 else 0}"
        )
        # Pull the probabilities from this trial (we logged [ts, P(non), P(tar)])
        _, p_non, p_tar = trial_probs[-1]

        # Print the prob that matches the trial type
        if true_label == 1:
            print(
                f"[Trial {trial}] P(target) = {p_tar:.3f}  (θ on P(target) = {config.THRESHOLD_TARGET:.2f})"
            )
        else:
            print(
                f"[Trial {trial}] P(non-target) = {p_non:.3f}  (θ on P(target) = {config.THRESHOLD_TARGET:.2f})"
            )

        send_udp_message(udp_marker, ip, port, message5)  # "Trial End"

        log_trial_prediction(
            log_path=log_path,
            run_idx=session_number,
            trial_idx=trial,
            mode=mode,
            prob=trial_probs,  # [[ts, p_non, p_tar]]
            pred=prediction,
            threshold=(config.THRESHOLD_TARGET),
        )

        send_udp_message(
            udp_marker, ip, port, message10 if is_correct else message11
        )  # 33/44
        draw_fixation_cross(1.5, config.GREEN if is_correct else config.red)
        pygame.display.flip()
        pygame.event.get()

        print(
            f"Stored decoder output for trial {trial}: {len(trial_probs)} timepoints."
        )

        # Green cross (TARGET)
        if mode == 11:  # TARGET
            if prediction == 1:  # CORRECT
                """messages = ["Correct", "P300 detected"]
                colors = [config.green, config.green]
                offsets = [-100, 100]
                duration = 1.5
                should_hold_and_classify = True"""
                print("Prediction correct for Target")
                send_udp_message(udp_marker, ip, port, message10)
            else:
                """messages = ["Incorrect", "P300 not detected"]
                colors = [config.red, config.red]
                offsets = [-100, 100]
                duration = 1.5
                should_hold_and_classify = False"""

                print("Prediction Incorrect for Target")
                send_udp_message(udp_marker, ip, port, message11)

        else:  # NON TARGET
            if prediction == 0:  # CORRECT
                """messages = ["Correct", "No P300 detected"]
                colors = [config.green, config.green]
                offsets = [-100, 100]
                duration = 1.5"""

                print("Prediction correct for non target")
                send_udp_message(udp_marker, ip, port, message10)

            else:
                """messages = ["Incorrect", "P300 detected"]
                colors = [config.red, config.red]
                offsets = [-100, 100]
                duration = 1.5"""

                print("Prediction Incorrect for non Target")
                send_udp_message(udp_marker, ip, port, message11)

        draw_fixation_cross(0.8, config.WHITE)

        print(f"Trial {trial} complete. Proceeding to next.")
        pygame.display.flip()

    print(f"Run complete")
    display_text(
        f"You got {correct_responses} out of {total_trials} correct!",
        font,
        config.WHITE,
        (screen_width // 2, screen_height // 2),
        3,
    )

    # ===== Decoder metrics (terminal) =====
    from sklearn.metrics import confusion_matrix

    y_true = np.array(true_list, dtype=int)
    y_pred = np.array(pred_list, dtype=int)

    if y_true.size > 0:
        # Confusion: rows = true, cols = pred; labels ensure [0,1] order
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            acc = (tp + tn) / total if total > 0 else float("nan")
            tpr = (
                tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            )  # sensitivity/recall for Target=1
            tnr = (
                tn / (tn + fp) if (tn + fp) > 0 else float("nan")
            )  # specificity for Non-target=0

            print("\n=== Decoder Summary ===")
            print(
                f"Trials: {total} | Double threshold on P(target): θ = {config.THRESHOLD_TARGET:.2f} and P(non): θ = {config.THRESHOLD_NONTARGET:.2f}"
            )
            print(f"Confusion matrix (true rows x pred cols) with labels [0,1]:\n{cm}")
            print(f"Accuracy: {acc*100:.1f}%")
            print(f"TPR (Target recall): {tpr*100:.1f}%   (TP={tp}, FN={fn})")
            print(f"TNR (Non-target specificity): {tnr*100:.1f}%   (TN={tn}, FP={fp})")
        else:
            print("Not enough class variety to compute full metrics.")
    else:
        print("No predictions collected; cannot compute metrics.")

    pygame.quit()


if __name__ == "__main__":
    main()
