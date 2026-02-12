"""
© 2026 Alexandra Mikhael. All Rights Reserved.
"""

import numpy as np


def target_epochs(eeg, marker_values, marker_timestamps, fs, time):
    """
    Extracts baseline-corrected epochs for correct target and non-target trials.

    Parameters:
        eeg (np.ndarray): EEG data (samples x channels)
        marker_values (np.ndarray): Marker codes
        marker_timestamps (np.ndarray): Corresponding timestamps (in seconds)
        fs (int): Sampling frequency
        time (np.ndarray): EEG time vector (in seconds)

    Returns:
        epochs (np.ndarray): Target epochs, shape (samples, channels, trials)
        epochs_nt (np.ndarray): Non-target epochs, shape (samples, channels, trials)
    """
    b = int(round(0.5 * fs))  # 500 ms pre-stim
    window_size_samples = int(round(1.0 * fs))  # 1000 ms post-stim

    marker_values = np.array(marker_values).flatten()
    marker_timestamps = np.array(marker_timestamps).flatten()

    epochs = []
    epochs_nt = []

    for i in range(len(marker_values) - 5):
        # Check full trial structure: [0, key, correct, 400, stim_type]
        if (
            marker_values[i] == 0
            and marker_values[i + 3] == 400
            and marker_values[i + 4] in [11, 12]
            and marker_values[i + 2] == 1  # only correct trials
        ):
            stim_time = marker_timestamps[i + 4]
            stim_idx = np.argmin(np.abs(time - stim_time))
            start_idx = stim_idx
            """
            if start_idx - b >= 0 and start_idx + window_size_samples <= eeg.shape[0]:
                segment = eeg[start_idx - b : start_idx + window_size_samples, :]
                baseline = np.mean(segment[:b, :], axis=0)
                data = segment[b:, :] - baseline

                # Append to target or non-target
                if marker_values[i + 4] == 11:
                    epochs.append(data)
                elif marker_values[i + 4] == 12:
                    epochs_nt.append(data)
                    """
            # Ensure we have a full post-stim window
            if stim_idx + window_size_samples <= eeg.shape[0]:
                # Post-stim segment only
                segment = eeg[
                    stim_idx : stim_idx + window_size_samples, :
                ]  # (samples, channels)

                # Baseline = value at start of epoch (time 0)
                baseline = segment[0, :]  # (channels,)

                # Baseline-correct entire epoch by subtracting the first sample
                data = segment - baseline  # broadcasts to (samples, channels)

                if marker_values[i + 4] == 11:
                    epochs.append(data)
                else:  # 12
                    epochs_nt.append(data)

    # Final formatting
    if len(epochs) == 0:
        print("⚠️ No valid target epochs extracted.")
        epochs = np.empty((window_size_samples, eeg.shape[1], 0))
    else:
        epochs = np.stack(epochs, axis=2)

    if len(epochs_nt) == 0:
        print("⚠️ No valid non-target epochs extracted.")
        epochs_nt = np.empty((window_size_samples, eeg.shape[1], 0))
    else:
        epochs_nt = np.stack(epochs_nt, axis=2)

    return epochs, epochs_nt
