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
from Utils.feedback_utils import (
    draw_fixation_cross,
    show_feedback,
)
from logger import log_trial_prediction

# Stream Utilites
from Utils.stream_utils import get_channel_from_lsl

# Configuration Parameters
from Utils import config

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

# MNE for real time EEG processing
import mne

mne.set_log_level("WARNING")

from EEGStreamState import EEGStreamState


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
do_cca = input("CCA? [1]Y [2]N")

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

# ===Load model===
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
eeg_dir = os.path.join(
    config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "training_data"
)
print(f"Script is looking for XDF files in: {eeg_dir}")

print("This model was trained on . . .")
eeg_dir = os.path.join(
    config.DATA_DIR, f"sub-{config.TRAINING_SUBJECT}", "training_data"
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
"""filter_state_tracker = initialize_filter_bank(
    fs=config.FS,
    lowcut=config.LOWCUT,
    highcut=config.HIGHCUT,
    notch_freqs=[60],
    notch_q=30,
)"""

SID = config.TRAINING_SUBJECT
SESSION_TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(f"Session timestamp set: {SESSION_TIMESTAMP}")

# --- where to store logs ---
log_dir = Path(config.OUTPUT_ONLINE_DIR) / f"sub-{SID}" / "predictions" / session_input
log_dir.mkdir(parents=True, exist_ok=True)

log_path = log_dir / f"predictions_sub-{SID}_{session_input}.txt"

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
    eeg_h = EEGStreamState(inlet=inlet, config=config)
    print("✅ EEGStreamState object created - ready to pull and process data")

    # Generate and log trial sequence
    global total_trials
    total_trials = 50
    target_ratio = 0.3  # 20% of trials should be matches
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
        # "FP1": "Fp1",
        # "FPZ": "Fpz",
        # "FP2": "Fp2",
        "FZ": "Fz",
        "CZ": "Cz",
        "PZ": "Pz",
        "POZ": "POz",
        # "OZ": "Oz",
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
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                waiting_for_start = False

    print("✅Started the task")

    while running and trial < total_trials:
        print(f"--- Trial {trial+1}/{len(sequence)} START ---")
        letter = sequence[trial]
        if trial > 0:  # Ensure at least one previous trial exists
            # Replace the failing line with this block (keep same indent level):
            if N == 0:
                correct_letter = sequence[trial - 1]
            else:
                correct_letter = sequence[trial - N]

            is_match = trial in target_indices  # Check if this trial is a match
            if is_match:
                send_udp_message(udp_marker, ip, port, message8)
                mode = 11
            elif not is_match:
                send_udp_message(udp_marker, ip, port, message9)
                mode = 12

            start_time = time.time()
            response_time = start_time + 1.1
            response = None
            send_udp_message(udp_marker, ip, port, message1)
            baseline_buffer = []

            while time.time() - start_time < 1.1:
                # Only show the number for 500 ms
                if time.time() - start_time < 0.5:
                    screen.fill(config.BLACK)
                    text_surface = letter_font.render(letter, True, config.WHITE)
                    text_rect = text_surface.get_rect(
                        center=(screen_width // 2, screen_height // 2)
                    )
                    screen.blit(text_surface, text_rect)
                    pygame.display.flip()
                else:
                    # Clear screen after 600ms for duration
                    duration = 0.8
                    draw_fixation_cross(duration, config.WHITE)
                    pygame.display.flip()

                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            exit()

                        if (
                            event.type == pygame.KEYDOWN
                            and response is None
                            and time.time() - start_time > 0.3
                        ):
                            if event.key == pygame.K_e:
                                pygame.quit()
                                exit()
                            if event.key == pygame.K_m:
                                response = "y"  # Yes (match)
                                send_udp_message(udp_marker, ip, port, message2)
                            elif event.key == pygame.K_z:
                                response = "n"  # No (no match)
                                send_udp_message(udp_marker, ip, port, message3)

                if response is None:
                    response = "timeout"
                    send_udp_message(udp_marker, ip, port, message4)

                if response != "timeout":
                    if (response == "y" and is_match) or (
                        response == "n" and not is_match
                    ):
                        correct_responses += 1
                        send_udp_message(udp_marker, ip, port, message6)
                    else:
                        send_udp_message(udp_marker, ip, port, message7)

        trial += 1
        # basline_buffer = get_baseline_fixation(inlet, None, 1500, baseline_buffer)
        next_trial_mode = sequence[trial]

        # baseline_data = np.array(basline_buffer)

        # Show feedback and perform classification
        print(
            f"Starting feedback classification - Mode: {'Target' if mode == 11 else 'Non-target'}"
        )
        (
            prediction,
            confidence,
            leaky_integrator,
            data_buffer,
            trial_probs,
        ) = show_feedback(
            duration=0.8,
            expected_feats=expected_feats,
            mode=mode,
            eeg_state=eeg_h,
        )
        pygame.display.flip()
        pygame.event.get()

        # log classification results
        print(
            f"Classification result - Predicted: {prediction}, Ground Truth: {1 if mode == 1 else 0}"
        )
        send_udp_message(udp_marker, ip, port, message5)

        # Store classification outcome

        log_trial_prediction(
            log_path=log_path,
            run_idx=session_number,
            trial_idx=trial,
            mode=mode,  # 1 target, 0 non-target
            prob=trial_probs,
            pred=prediction,
            threshold=config.THRESHOLD_TARGET,
        )

        print(
            f"Stored decoder output for trial {trial}: {len(trial_probs)} timepoints."
        )

        pred_list.append(prediction)
        true_list.append(1 if mode == 11 else 0)

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

            should_hold_and_classify = False

        # Display feedback messages
        """print(
            f"Displaying feedback: '{messages[0]}' | Meaning: '{messages[1]}' | Duration: {duration}s"
        )
        display_multiple_mess_udp(
            messages=messages,
            colors=colors,
            offsets=offsets,
            duration=duration,
        )"""
        cross_col = (
            config.GREEN
            if (mode == 11 and prediction == 1) or (mode == 12 and prediction == 0)
            else config.red
        )
        """display_cross_with_messages(
            messages, colors, offsets, duration, cross_color=cross_col, font_obj=font
        )"""

        # draw_fixation_cross(0.8, config.WHITE)
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
    pygame.quit()


if __name__ == "__main__":
    main()
