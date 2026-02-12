"""
© 2026 Alexandra Mikhael. All Rights Reserved.
"""

import os
import pyxdf
import numpy as np


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


def load_project_data(ID, session, filter, runs, data_type, N=None):
    subject_ids = {
        100: "sub-P100",
        9: "sub-P009",
        1: "sub-P001",
        2: "sub-P002",
        3: "sub-P003",
        101: "sub-P101",
        102: "sub-P102",
        103: "sub-P103",
        164: "sub-P164",
        203: "sub-P203",
        367: "sub-P367",
        321: "sub-P321",
        322: "sub-P322",
        323: "sub-P323",
        401: "sub-P401",
        621: "sub-P621",
        622: "sub-P622",
        623: "sub-P623",
        921: "sub-P921",
        922: "sub-P922",
        923: "sub-P923",
        1221: "sub-P1221",
        1222: "sub-P1222",
        1223: "sub-P1223",
        1224: "sub-P1224",
        1321: "sub-P1321",
        1322: "sub-P1322",
        1323: "sub-P1323",
    }
    ID_str = subject_ids.get(ID, f"sub-P{ID}")

    if filter == 1:
        type_session = "Offline"
    else:
        type_session = "Online"

    base_path = os.path.join(
        "/home/alexandra-admin/Documents/CurrentStudy/", ID_str, session, type_session
    )
    all_streams = []

    def load_run_file(run_path, filename_fmt):
        for i in range(1, runs + 1):
            filename = filename_fmt % i
            file_path = os.path.join(run_path, filename)
            print(f"Loading file: {file_path}")
            if not os.path.exists(file_path):
                print("❌ File not found.")
                continue
            try:
                run_streams, _ = pyxdf.load_xdf(file_path)
                for stream in run_streams:
                    print(f"Stream Name: {stream['info']['name'][0]}")
                    print(f"Stream Type: {stream['info']['type'][0]}")
                    print(f"Number of samples: {len(stream['time_series'])}")
                    print("-----")
                all_streams.append(run_streams)
            except Exception as e:
                print(f"⚠️ Failed to load or parse XDF: {e}")

    # === Path logic ===
    if session == "Relaxation":
        if data_type == "EOG":
            run_path = os.path.join(base_path, data_type)
            load_run_file(run_path, "EOG_run-%03d_eeg.xdf")
        elif data_type == "Eyes Closed pre":
            run_path = os.path.join(base_path, data_type)
            load_run_file(run_path, "ECpre_run-%03d_eeg.xdf")
        elif data_type == "Eyes Closed post":
            run_path = os.path.join(base_path, data_type)
            load_run_file(run_path, "ECpost_run-%03d_eeg.xdf")
        elif data_type == "Relax":
            run_path = os.path.join(base_path, data_type)
            load_run_file(run_path, "relax_run-%03d_eeg.xdf")
        elif data_type in ["Nback", "Nback + relax"]:
            pathn = f"N{N}"
            run_path = os.path.join(base_path, data_type, pathn)
            load_run_file(run_path, "nb_run-%03d_eeg.xdf")
    elif session == "tACS":
        # Add corresponding logic for tACS session if needed
        pass

    # === Separate EEG and Marker streams per run ===
    eeg_runs = []
    marker_runs = []

    for run_streams in all_streams:
        eeg_stream = None
        marker_stream = None
        for stream in run_streams:
            if "type" in stream["info"]:
                if stream["info"]["type"][0] == "EEG":
                    eeg_stream = stream
                # elif stream["info"]["type"][0] == "Markers":
                # marker_stream = stream

                elif (
                    stream["info"]["type"][0] == "Markers"
                    and stream["info"]["name"][0] == "MarkerStream"
                ):
                    marker_stream = stream

        if eeg_stream is None or marker_stream is None:
            print("⚠️ Missing EEG or Marker stream in one run — skipping.")
            continue

        eeg_data = np.array(eeg_stream["time_series"])
        eeg_ts = np.array(eeg_stream["time_stamps"])
        marker_data = [int(m[0]) for m in marker_stream["time_series"]]
        marker_ts = np.array(marker_stream["time_stamps"])

        eeg_runs.append({"data": eeg_data, "timestamps": eeg_ts})
        marker_runs.append({"values": np.array(marker_data), "timestamps": marker_ts})
        print(f"✅ Loaded {len(eeg_runs)} runs.")

        labels = get_channel_names_from_xdf(eeg_stream)
        if labels != None:
            labels = np.delete(labels, np.arange(32, 39), axis=0)
        else:
            labels = []

        print(f"Channels", labels)

    return eeg_runs, marker_runs, labels
