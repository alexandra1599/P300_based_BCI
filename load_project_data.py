import os
import pyxdf
import numpy as np


def load_project_data(ID, session, runs, data_type, N=None):
    subject_ids = {
        9: "Subject 9",
        101: "Subject 101",
        102: "Subject 102",
        367: "Subject 367",
        321: "Subject 321",
        322: "Subject 322",
        323: "Subject 323",
        621: "Subject 621",
        622: "Subject 622",
        623: "Subject 623",
        921: "Subject 921",
        922: "Subject 922",
        923: "Subject 923",
        1221: "Subject 1221",
        1222: "Subject 1222",
        1223: "Subject 1223",
        1224: "Subject 1224",
        1321: "Subject 1321",
        1322: "Subject 1322",
        1323: "Subject 1323",
    }
    ID_str = subject_ids.get(ID, f"Subject {ID}")

    base_path = os.path.join(
        "/Users/alexandra/Desktop/PhD/Project/Experiment/Data", ID_str, session
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
                elif stream["info"]["type"][0] == "Markers":
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

    # === Extract labels once from the first valid EEG stream ===
    labels = None
    try:
        first_eeg = eeg_runs[0]
        desc = all_streams[0][0]["info"]["desc"][0]
        if "channels" in desc and "channel" in desc["channels"][0]:
            labels = [ch["label"][0] for ch in desc["channels"][0]["channel"]]
    except Exception as e:
        print(f"⚠️ Failed to extract channel labels: {e}")

    print(f"✅ Loaded {len(eeg_runs)} runs.")
    return eeg_runs, marker_runs, labels
