from scipy.signal import iirnotch, butter, filtfilt, lfilter, firwin
import numpy as np


def real_time_bandpass(eeg, fs=512, lowcut=1, highcut=40, order=101):
    nyq = 0.5 * fs
    taps = firwin(order, [lowcut / nyq, highcut / nyq], pass_zero=False)
    return lfilter(taps, 1.0, eeg, axis=0)


import numpy as np
from scipy.signal import butter, sosfiltfilt


def zero_phase_bandpass(x, fs, low, high, order=4, axis=0, padlen=None):
    """
    x: array, can be (time, channels) or (samples,) etc.
    fs: Hz
    low, high: Hz
    order: IIR order (per direction)
    axis: time axis in x
    padlen: override default padding to control edge behavior (int or None)
    """
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=axis, padlen=padlen)


def filtering(eeg_data, session, filter, labels=None, fs=512, do_car=False):
    """
    Apply notch filter, optional CAR, and bandpass filtering.

    Parameters:
        eeg_data (np.ndarray): EEG data (time x channels)
        session (str): 'tACS' or 'Relaxation'
        filter (str) : online/offline, causal/non-causal
        labels (list): Optional channel labels (will be modified if M1/M2 removed)
        fs (int): Sampling frequency
        do_car (bool): Apply CAR (default: False)


    Returns:
        filtered_signal (np.ndarray): EEG after preprocessing
        labels (list): Updated labels if M1/M2 were removed
    """
    eeg = eeg_data.copy()

    # === Remove EOG/AUX channels ===
    eeg = np.delete(eeg, np.arange(32, 39), axis=1)

    # === Notch filter at 60 Hz ===
    print("🔧 Applying 60 Hz notch filter...")
    harmonics = 2
    line_freq = 60
    quality_factor = 30
    for harmonic in range(1, harmonics + 1):
        target_freq = line_freq * harmonic
        b, a = iirnotch(target_freq, quality_factor, fs)
        filtered_data = filtfilt(b, a, eeg, axis=0)

    # === Remove M1 and M2 if tACS ===
    if session.lower() == "tacs":
        print("🧹 Removing M1 (ch 13) and M2 (ch 19)...")
        eeg = np.delete(eeg, [18, 12], axis=1)  # 0-indexed
        if labels:
            del labels[18]
            del labels[12]

    # === Optional CAR ===
    if do_car:
        print("🧠 Applying CAR...")
        # filtered_data = np.delete(filtered_data, [18, 12], axis=1)  # 0-indexed
        car_data = filtered_data
        car_data = np.delete(car_data, [18, 17, 13, 12], axis=1)
        avg = np.mean(car_data, axis=1, keepdims=True)
        eeg = filtered_data - avg
        # === Bandpass filter ===
        print("📡 Applying bandpass filter: 1–40 Hz...")
        low, high = 1.0, 12.0
        nyquist = 0.5 * fs
        b, a = butter(4, [low / nyquist, high / nyquist], btype="band")
        if filter == 1:  # Offline
            eeg = zero_phase_bandpass(eeg, fs, low, high, order=4, axis=0)
        else:
            eeg = real_time_bandpass(eeg, fs=fs, lowcut=1, highcut=40, order=101)
        print("✅ Filtering complete.")

    else:
        print("⚠️ Skipping CAR.")
        # === Bandpass filter ===
        print("📡 Applying bandpass filter: 1–12 Hz...")

        low, high = 1.0, 12.0
        nyquist = 0.5 * fs
        b, a = butter(4, [low / nyquist, high / nyquist], btype="band")
        if filter == 1:  # Offline
            eeg = zero_phase_bandpass(eeg, fs, low, high, order=4, axis=0)
        else:
            eeg = real_time_bandpass(eeg, fs=fs, lowcut=1, highcut=40, order=101)
        print("✅ Filtering complete.")

    return eeg
