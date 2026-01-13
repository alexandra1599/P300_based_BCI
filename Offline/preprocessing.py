from scipy.signal import welch, butter, filtfilt, iirnotch, lfilter
import numpy as np
from sklearn.linear_model import LinearRegression
import config
import mne


def concatenate_streams(eeg, marker):
    """
    Concatenate multiple EEG and marker streams into a single dataset with preserved metadata
    """

    eeg_all = {"time_series": [], "time_stamps": [], "info": eeg[0].get("info", {})}
    markers_all = {"time_series": [], "time_stamps": []}

    offset = 0
    for i, (data, m) in enumerate(zip(eeg, marker)):
        if i == 0:
            eeg_all["time_series"] = data["time_series"]
            eeg_all["time_stamps"] = data["time_stamps"]
            markers_all["time_series"] = m["time_series"]
            markers_all["time_stamps"] = m["time_stamps"]
        else:
            offset = (
                eeg_all["time_stamps"][-1]
                - data["time_stamps"][0]
                + np.mean(np.diff(data["time_stamps"]))
            )
            eeg_all["time_series"] = np.vstack(
                [eeg_all["time_series"], data["time_series"]]
            )
            eeg_all["time_stamps"] = np.concatenate(
                [eeg_all["time_stamps"], data["time_stamps"] + offset]
            )
            markers_all["time_series"] = np.vstack(
                [markers_all["time_series"], m["time_series"]]
            )
            markers_all["time_stamps"] = np.concatenate(
                [markers_all["time_stamps"], m["time_stamps"] + offset]
            )

    return eeg_all, markers_all


def apply_streaming_filters(data, filter_bank, filter_state=None):
    """
    Applies notch and bandpass filters in sequence using lfilter with state.

    Parameters:
        data (ndarray): EEG data (samples x ch)
        filter_bank (dict): Output from initialize_filter_bank
        filter_state (dict or None): Previous zi values for filters (optional)

    Returns:
        filtered (ndarray): Filtered data
        updated_state (dict): Updated filter state for streaming
    """
    if not isinstance(filter_bank, dict) or "bandpass" not in filter_bank:
        raise ValueError(
            "Invalid filter_bank: must be a dict with at least 'bandpass' key."
        )
    if "notch" not in filter_bank:
        print("⚠️ No notch filters found in filter_bank — skipping notch filtering.")

    if filter_state is None:
        filter_state = {}

    filtered = data
    updated_state = {}

    # --- Apply notch filters (if any) ---
    for idx, notch_coeffs in enumerate(filter_bank.get("notch", [])):
        b, a = notch_coeffs
        key = f"notch_{idx}"
        zi = filter_state.get(key)
        if zi is None:
            zi = np.zeros((data.shape[0], len(a) - 1))
        filtered, zf = lfilter(b, a, filtered, axis=1, zi=zi)
        updated_state[key] = zf

    avg = np.mean(filtered, axis=0, keepdims=True)
    filtered = filtered - avg

    # --- Apply bandpass filter ---
    bp_b, bp_a = filter_bank["bandpass"]
    key = "bandpass"
    zi = filter_state.get(key)
    if zi is None:
        zi = np.zeros((data.shape[0], len(bp_a) - 1))
    filtered, zf = lfilter(bp_b, bp_a, filtered, axis=1, zi=zi)
    updated_state[key] = zf

    return filtered, updated_state


def initialize_filter_bank(fs, lowcut, highcut, notch_freqs=[60], notch_q=30, order=4):
    """
    Initializes causal filter coefficients for bandpass and notch filters.

    Parameters:
        fs (float): Sampling rate in Hz
        lowcut (float): Low cutoff for bandpass
        highcut (float): High cutoff for bandpass
        notch_freqs (list): Frequencies to notch filter (e.g., [60])
        notch_q (float): Q factor for notch filters
        order (int): Order for bandpass filter

    Returns:
        dict: {'bandpass': (b, a), 'notch': [(b, a), ...]}
    """
    # Bandpass filter
    nyq = 0.5 * fs
    bp_b, bp_a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")

    # Notch filters
    notch_filters = []
    for freq in notch_freqs:
        b, a = iirnotch(freq / nyq, notch_q)
        notch_filters.append((b, a))

    return {"bandpass": (bp_b, bp_a), "notch": notch_filters}


def apply_notch(eeg, fs, line_freq=60, quality_factor=30, harmonics=2):
    """
    Apply notch filter to EEG data to remove line noise

    Parameters :
            eeg (np.ndarray): EEG data of shape (samples,channels)
            fs (int): Sampling Rate in Hz
            line_freq (float): Line noise frequency (default: 60 Hz)
            quality_factor (float): Quality factor of the notch filter (default: 30)
            harmonics (int): Number of harmonics to filter (default: 1)

    Returns : np.ndarray : Filtered EEG data with same size and shape as input
    """

    filtered_data = eeg.copy()

    for harmonic in range(1, harmonics + 1):
        target_freq = line_freq * harmonic
        b, a = iirnotch(target_freq, quality_factor, fs)
        filtered_data = filtfilt(b, a, filtered_data, axis=0)

    return filtered_data


def butter_bandpass(lowcut, highcut, fs, order=4):

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")

    return b, a


def apply_car(data):

    avg = np.mean(data, axis=0)
    return data - avg


def compute_grand_avg(data, fs, mode="trials"):
    """
    Compute the grand average of EEG data in specific time window

    Parameters :
            data (np.ndarray) : Input data (samples, channels, trials)
            fs (int): Sampling rate of EEG data in Hz
            mode (str): Average mode
                    "trials" - Average across trials, retaining temporal resolution
                    "trials_and_timepoints" - Average across trials and timepoints, one value per channel

    Returns:
            np.ndarray : Grand Average data
                         Shape : (channels, window_samples) for "trials"
                         Shape : (channels,) for "trials_adn_timepoints"
    """

    if mode == "trials":
        g_avg = np.mean(data, axis=2)
        return g_avg  # Shape: (channels, window_samples)

    elif mode == "trials_and_timepoints":
        g_avg = np.mean(data, axis=(0, 2))
        return g_avg  # Shape: (channels,)

    else:
        raise ValueError(
            f"Invalid mode: {mode}. Choose 'trials' or 'trials_and_timepoints'."
        )


def extract_segments(
    eeg_data, eeg_timestamps, m_timestamps, m_values, window_size_ms, fs, offset=0
):
    """
    Extract EEG segments based on marker timestamps and time window

    Parameters :
            eeg_data (np.ndarray): 2D array of EEG data (samples, channels)
            eeg_timestamps (np.ndarray): 1D array of timestamps for EEG data
            m_timestamps (np.ndarray): 1D array of timestamps for marker data
            m_values (np.ndarray): 1D array of marker values
            sindow_size_ms (int): Time window for segment extraction in ms
            fs (float): Sampling rate of EEG data in Hz
            offset (int): Time offset after the marker in millisecond (default: 0)

    Returns:
            np.ndarray: 3D array of segments with shape (time, trials, channels)
            np.ndarray: 1D array of labels corresponding to the trials
    """

    window_samples = int((window_size_ms / 1000) * fs)
    offset_samples = int((offset / 1000) * fs)
    segments = []
    labels = []

    for marker_time, marker_value in zip(m_timestamps, m_values):
        if marker_value not in [100, 200]:
            continue

        # Find marker index in EEG timestamps
        closest_idx = np.searchsorted(eeg_timestamps, marker_time)
        start_idx = closest_idx + offset_samples
        end_idx = start_idx + window_samples

        # Ensure indices are within bounds of EEG data
        if start_idx < 0 or end_idx > len(eeg_data):
            print(f"Skipping marker at {marker_time:.2f}s: Out of bound.")
            continue

        # Extract Segment
        segment = eeg_data[start_idx:end_idx, :]  # shape (time, channels)

        if segment.shape[0] == window_samples:
            segments.append(segment)
            labels.append(marker_value)
        else:
            print(f"Skipping marker at {marker_time: .2f}s : Segment size mismatch.")

    # Convert to numpy arrays
    if segments:
        segments = np.stack(segments, axis=2)  # Stack trials on 3rd axis

    else:
        segments = np.empty((window_samples, eeg_data.shape[1], 0))

    labels = np.array(labels)

    return segments, labels


def flatten_segments(segments):
    """
    Flatten segmented EEG data from 3D to 2D

    Parameter:
            segments (n.ndarray): Input data of shape (time,channels,trials)

    Returns:
            np.ndarray: Flattened data of shape (trials, time*channels)
    """

    if segments.ndim != 3:
        raise ValueError("Input data must ne a 3D array.")

    trials_first = np.transpose(segments, (2, 0, 1))  # Shape (trials,time,channels)

    # Flatten time and channels dimensions
    flattened = trials_first.reshape(
        trials_first.shape[0], -1
    )  # Shape (trials, time*channels)

    return flattened


def flatten_single_segment(segment):
    """
    Flatten segmented EEG data from 2D to 1D. This is used for prediction during test

    Parameters:
            segments (np.ndarray): Input data shape (time,channels)

    Returns:
            np.ndarray: Flattened data of shape (time * channels)
    """

    if segment.ndim != 2:
        raise ValueError("Input data must be a 2D array (time, channels).")

    flattened = segment.flatten()  # Shape (time*channels,)

    # Reshape into a row vector
    flattened_row = flattened.reshape(1, -1)  # Shape : (1, time*channels)

    return flattened_row


def extract_flatten_segment(eeg, start, fs, window_size, offset=0):
    """
    Extract and flatten a single EEG segment from the raw data

    Parameters :
            eeg (np.ndarray): 2D array of EEG data (samples x channels)
            start (float): Start time of the segment in seconds
            fs (float): Sampling rate of EEG data in Hz
            window_size (int): Time window for segment in ms
            offset (int): Time offset after start in ms (default: 0)

    Returns :
            np.ndarray: Flattened segment of shape (1, time*channels)
    """

    window_samples = int((window_size / 1000) * fs)
    offset_samples = int((offset / 1000) * fs)

    # Calculate start and end indices
    start_idx = int(start * fs) + offset_samples
    end_idx = start_idx + window_samples

    # Ensure indices are within bounds
    if start_idx < 0 or end_idx > eeg.shape[0]:
        raise ValueError(
            f"Segment indices out of bounds: start_idx={start_idx}, end_idx={end_idx}"
        )

    # Extract segment
    segment = eeg[start_idx:end_idx, :]

    # Ensure segment has correct shape
    if segment.shape[0] != window_samples:
        raise ValueError(
            f"Segment size mismatch: expected {window_samples} samples, got {segment.shape[0]}"
        )

    # Flatten segment into row vector
    flattened_segment = segment.flatten().reshape(1, -1)  # Shape (1,timexchannels)

    return flattened_segment


def separate_classes(segments, labels, class1=100, class2=200):
    """
    Separate EEG segments and labels into classes

    Parameters :
            segments (np.ndarray): Input data of shape (time,channels,trials)
            labels (np.ndarray): Corresponding labels for the trials
            class1 (int): Marker value for class 1 (100: Button press match)
            class2 (int): Marker value for class 2 (200: Button press non-match)

    Returns:
            tuple: Two dictionaries, each containing 'data' (segments) and 'labels' for the two classes
    """

    # Ensure label match the trials in the segments
    if segments.shape[2] != len(labels):
        raise ValueError(f"Mismatch between number of trials in segments and labels.")

    # Boolean masks for class separation
    mask_class1 = labels == class1
    mask_class2 = labels == class2

    # Separate the segments and labels
    class1_data = segments[:, :, mask_class1]
    class1_labels = labels[mask_class1]

    class2_data = segments[:, :, mask_class2]
    class2_labels = labels[mask_class2]

    return (
        {"data": class1_data, "labels": class1_labels},
        {"data": class2_data, "labels": class2_labels},
    )


def remove_eog(eeg, eog):
    """
    Remove EOG artifacts from EEG data using regression

    Parameters:
            eeg (np.ndarray): 2D array of EEG data (samples,channels)
            eog (np.ndarray): 2D array of EOG data (samples,EOGchannels)

    Returns:
            np.ndarray: Cleaned EEG data
    """

    if eog.ndim == 1:
        eog = eog[:, np.newaxis]  # Convert to (samples x 1)

    eeg_cleaned = np.zeros_like(eeg)

    # Perform regression for each channel
    for channel in range(eeg.shape[1]):
        # Extract current EEG channel
        eeg_ch = eeg[:, channel]

        # Fit linear regression model
        regressor = LinearRegression()
        regressor.fit(eog, eeg_ch)

        # Predict EOG contribution to EEG channel
        eog_pred = regressor.predict(eog)

        # Substract EOG contribution from EEG
        eeg_cleaned[:, channel] = eeg_ch - eog_pred

    return eeg_cleaned


def get_valid_channel_mask_and_metadata(
    eeg_data, channel_names, fs, drop_mastoids=True
):
    # 1) Normalize names here (single source of truth)
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
    norm_names = [rename_dict.get(ch, ch) for ch in channel_names]

    # 2) Drop non-EEG
    non_eeg = {"AUX1", "AUX2", "AUX3", "AUX7", "AUX8", "AUX9", "TRIGGER"}
    valid_mask = [nm not in non_eeg for nm in norm_names]
    valid_indices = [i for i, keep in enumerate(valid_mask) if keep]
    valid_channels = [norm_names[i] for i in valid_indices]

    # 3) Build Raw for metadata/selection
    info = mne.create_info(ch_names=valid_channels, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(eeg_data[valid_indices, :].copy(), info)

    # 4) Optionally drop M1/M2 AND update mapping accordingly
    if drop_mastoids and "M1" in raw.ch_names and "M2" in raw.ch_names:
        raw.drop_channels(["M1", "M2"])
        valid_channels = raw.ch_names
        # recompute mapping from original channel_names → valid_channels after drop
        valid_indices = [
            i
            for i, nm in enumerate(norm_names)
            if nm in valid_channels and nm not in non_eeg
        ]

    return valid_channels, raw, valid_indices


def get_channel_names_from_xdf(eeg_stream):
    """
    Extract channel names from an EEG stream in a pyxdf file.

    Parameters:
        eeg_stream (dict): EEG stream from the loaded pyxdf file.

    Returns:
        list: A list of channel names.
    """
    if "desc" in eeg_stream["info"] and "channels" in eeg_stream["info"]["desc"][0]:
        channel_desc = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
        channel_names = [channel["label"][0] for channel in channel_desc]
        return channel_names
    else:
        raise ValueError("Channel names not found in EEG stream metadata.")


def select_channels(raw, keep_channels=None):
    """
    Filters the MNE Raw object to keep only explicitly specified channels.

    Parameters:
    - raw (mne.io.Raw): The MNE Raw object containing EEG data.
    - keep_channels (list[str]): Exact channel names to keep (e.g., ["C3", "Cz", "C4"]).
                                If None, keeps all channels.

    Returns:
    - raw (mne.io.Raw): The modified MNE Raw object with only the selected channels.
    """

    channel_names = raw.info["ch_names"]

    if keep_channels is not None:
        # Ensure only channels that exist in the data are selected
        selected_channels = [ch for ch in keep_channels if ch in channel_names]
        missing = [ch for ch in keep_channels if ch not in channel_names]
        if missing:
            raise ValueError(f"Requested channels not found in Raw: {missing}")
    else:
        selected_channels = channel_names

    raw.pick_channels(selected_channels)
    return raw


def parse_eeg_eog(eeg, channel):
    """
    Parse EEG and EOG data from given EEG stream

    Parameters:
            eeg (dict): EEG streams containing 'time_series' and 'time_stamps'
            channel (list): List of channel names

    Returns:
            tuple:
                    -np.ndarray: EEG data from specified channel
                    -np.ndarray or None: EOG data from specified channel (if EOG_TOGGLE enabled)
    """

    # Extract EEG data
    eeg_data = np.array(eeg["time_series"])  # Shape: (N_samples, N_channels)

    # Handle "ALL" keyword for EEG channels
    if config.EEG_CHANNEL_NAMES == ["ALL"]:
        eeg_selected = eeg_data[
            :, : config.CAP_TYPE
        ]  # Assume first 32 channels are EEG
    else:
        # Identify indices for EEG channels based on configuration
        eeg_indices = [
            channel.index(ch) for ch in config.EEG_CHANNEL_NAMES if ch in channel
        ]
        if not eeg_indices:
            raise ValueError(f"No matching EEG channels found in the provided stream.")

        # Extract EEG data
        eeg_selected = eeg[:, eeg_indices]

    # Handle EOG data
    if config.EOG_TOGGLE:
        # Identify indices for EOG channels
        eog_indices = [
            channel.index(ch) for ch in config.EOG_CHANNEL_NAMES if ch in channel
        ]
        if not eog_indices:
            print("Warning: No matching EOG channels found in stream.")
            eog_selected = None
        else:
            # Extracted EOG data
            eog_selected = eeg_data[:, eog_indices]

    else:
        eog_selected = None

    return eeg_selected, eog_selected


def extract_P300_features(epochs, fs=512):
    """
    epochs: 2D array shaped (n_samples, n_channels)
    returns: (n_channels, 8) features per channel
    """
    epochs = np.asarray(epochs, dtype=float)
    if epochs.ndim != 2:
        raise ValueError(f"epochs must be 2D (samples x channels), got {epochs.shape}")

    n_samples, n_channels = epochs.shape  # <-- do NOT transpose here

    # 0.25–0.60 s window (clamped to available samples)
    start = int(0.25 * fs)
    end = int(0.60 * fs)
    start = max(0, min(start, max(0, n_samples - 2)))
    end = max(start + 1, min(end, n_samples))

    feats = np.empty((n_channels, 8), dtype=float)

    for ch in range(n_channels):
        signal = epochs[:, ch]
        window = signal[start:end]
        if window.size < 4:
            window = signal

        mean_amp = float(np.mean(window))
        peak_amp = float(np.max(window))
        peak_idx = int(np.argmax(window))
        peak_lat = float((start + peak_idx) / fs)  # seconds
        variance = float(np.var(window))
        min_before = float(np.min(window[:peak_idx])) if peak_idx > 0 else 0.0
        min_after = (
            float(np.min(window[peak_idx + 1 :])) if peak_idx + 1 < window.size else 0.0
        )
        peak2peak = float(peak_amp - np.min(window))

        nperseg = max(8, min(window.size, 256))
        if window.size >= 8:
            f, Pxx = welch(window, fs=fs, nperseg=nperseg)
            dom_freq = float(f[int(np.argmax(Pxx))]) if f.size else 0.0
        else:
            dom_freq = 0.0

        feats[ch] = [
            mean_amp,
            peak_amp,
            peak_lat,
            min_before,
            min_after,
            peak2peak,
            dom_freq,
            variance,
        ]

    return feats
