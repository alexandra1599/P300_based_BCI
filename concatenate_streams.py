import numpy as np


def concatenate_streams(all_streams):
    """
    Adapted function to concatenate EEG and marker streams from your loaded project data.
    Input:
        all_streams: List of lists of streams (one list per run, each containing multiple streams)
    Output:
        List of merged streams [merged_marker_stream, merged_eeg_stream]
    """
    eeg_streams = []
    marker_streams = []

    for run_streams in all_streams:
        for stream in run_streams:
            try:
                stream_type = stream["info"]["type"][0]
                if stream_type == "EEG":
                    eeg_streams.append(stream)
                elif stream_type == "Markers":
                    marker_streams.append(stream)
            except Exception as e:
                print(f"Skipping stream due to error: {e}")

    if not eeg_streams or not marker_streams:
        raise ValueError("No valid EEG or Marker streams found.")

    merged_eeg = {
        "time_series": eeg_streams[0]["time_series"],
        "time_stamps": eeg_streams[0]["time_stamps"],
        "info": eeg_streams[0].get("info", {}),
    }

    merged_markers = {
        "time_series": marker_streams[0]["time_series"],
        "time_stamps": marker_streams[0]["time_stamps"],
        "info": marker_streams[0].get("info", {}),
    }

    for i in range(1, len(eeg_streams)):
        eeg_offset = (
            merged_eeg["time_stamps"][-1]
            - eeg_streams[i]["time_stamps"][0]
            + np.mean(np.diff(eeg_streams[i]["time_stamps"]))
        )
        merged_eeg["time_series"] = np.vstack(
            [merged_eeg["time_series"], eeg_streams[i]["time_series"]]
        )
        merged_eeg["time_stamps"] = np.concatenate(
            [merged_eeg["time_stamps"], eeg_streams[i]["time_stamps"] + eeg_offset]
        )

        marker_offset = (
            merged_markers["time_stamps"][-1]
            - marker_streams[i]["time_stamps"][0]
            + np.mean(np.diff(marker_streams[i]["time_stamps"]))
        )
        merged_markers["time_series"] = np.vstack(
            [merged_markers["time_series"], marker_streams[i]["time_series"]]
        )
        merged_markers["time_stamps"] = np.concatenate(
            [
                merged_markers["time_stamps"],
                marker_streams[i]["time_stamps"] + marker_offset,
            ]
        )

    return [merged_markers, merged_eeg]
