def extract_data(streams):
    import numpy as np

    eeg_data = None
    markers = []
    timestamps = None
    fs = None

    for stream in streams:
        # Some streams are tuples (stream_dict, metadata), we take the first element
        if isinstance(stream, (list, tuple)) and isinstance(stream[0], dict):
            stream = stream[0]

        try:
            stream_type = stream["info"]["type"][0]
        except Exception as e:
            print(f"⚠️ failed to get type ({e})")
            continue

        if stream_type == "EEG":
            eeg_data = np.array(stream["time_series"])
            timestamps = np.array(stream["time_stamps"])
            fs = float(stream["info"]["nominal_srate"][0])
        elif stream_type == "Markers":
            markers = np.array(stream["time_series"])

    if eeg_data is None or timestamps is None or fs is None:
        raise ValueError("❌ Failed to extract EEG data")

    return eeg_data, timestamps, fs, markers
