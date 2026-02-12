"""
© 2026 Alexandra Mikhael. All Rights Reserved.
"""
import numpy as np


"""def extract_features(epochs, fs=512):
    features = []
    for trial in epochs.transpose(2, 0, 1):  # trial shape: (samples, channels)
        # Use only Pz (e.g., channel 25)
        signal = trial[:, 25]
        time_window = signal[int(0.3 * fs) : int(0.6 * fs)]

        mean_amp = np.mean(time_window)
        peak_amp = np.max(time_window)
        peak_latency = np.argmax(time_window) / fs

        features.append([mean_amp, peak_amp, peak_latency])
    return np.array(features)"""


import numpy as np
from scipy.signal import welch


def extract_features(epochs, fs=512):
    """
    Extract features from all EEG channels:
    - Mean amplitude
    - Peak amplitude
    - Peak latency
    - Min before peak
    - Min after peak
    - Peak-to-peak amplitude
    - Dominant frequency in 0.3–0.6s window

    Args:
        epochs (np.ndarray): Shape (n_samples, n_channels, n_trials)
        fs (int): Sampling rate in Hz

    Returns:
        features (np.ndarray): Shape (n_trials, n_channels * 7)
    """
    n_samples, n_channels, n_trials = epochs.shape

    features = []

    for trial in epochs.transpose(2, 0, 1):  # (samples, channels)
        trial_feats = []
        for ch in range(n_channels):
            signal = trial[:, ch]
            window = signal[int(0.25 * fs) : int(0.6 * fs)]

            # Basic features
            mean_amp = np.mean(window)
            peak_amp = np.max(window)
            peak_idx = np.argmax(window)
            peak_latency = peak_idx / fs
            variance = np.var(window)

            # Min before and after peak
            min_before = np.min(window[:peak_idx]) if peak_idx > 0 else 0
            min_after = (
                np.min(window[peak_idx + 1 :]) if peak_idx + 1 < len(window) else 0
            )

            # Peak-to-peak amplitude
            peak_to_peak = peak_amp - min(window)

            # Dominant frequency via Welch's method
            f, Pxx = welch(window, fs=fs, nperseg=len(window))
            dom_freq = f[np.argmax(Pxx)]

            # Append all features
            trial_feats.extend(
                [
                    mean_amp,
                    peak_amp,
                    peak_latency,
                    min_before,
                    min_after,
                    peak_to_peak,
                    dom_freq,
                    variance,
                ]
            )
        features.append(trial_feats)

    return np.array(features)
