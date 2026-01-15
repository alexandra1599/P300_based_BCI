import os
import pyxdf
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class SessionConfig:
    """Configuration for a single session"""

    session_id: str  # e.g., "ses-S001"
    task_name: str  # e.g., "Nbackpre"
    n_runs: int
    filter_type: int  # 1 = offline, 2 = online
    n_level: Optional[int] = None  # For N-back tasks


# Define session configurations
OFFLINE_SESSIONS = [
    SessionConfig("ses-S001", "Nbackpre", 4, 1, 2),
    # SessionConfig("ses-S002", "ecpre", 1, 1),
    # SessionConfig("ses-S003", "relax", 1, 1),
    # SessionConfig("ses-S004", "ecpost", 1, 1),
    SessionConfig("ses-S005", "Nbackpost", 8, 1, 2),
]

ONLINE_SESSIONS = [
    # SessionConfig("ses-S012", "ecpre", 1, 2),
    # SessionConfig("ses-S013", "relax", 1, 2),
    # SessionConfig("ses-S014", "ecpost", 1, 2),
    SessionConfig("ses-S015", "Nbackonline", 8, 2, 2),
]


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


def get_available_subjects(
    base_path: str = "/home/alexandra-admin/Documents/CurrentStudy",
) -> List[int]:
    """
    Automatically detect all available subject IDs from the directory structure.

    Parameters:
        base_path (str): Base directory containing subject folders

    Returns:
        List[int]: Sorted list of subject IDs found in the directory
    """
    base_path = Path(base_path)
    subject_ids = []

    if not base_path.exists():
        print(f"⚠️ Warning: Base path {base_path} does not exist!")
        return []

    for subject_dir in base_path.iterdir():
        if subject_dir.is_dir() and subject_dir.name.startswith("sub-P"):
            try:
                # Extract numeric ID from "sub-P###" format
                subject_id = int(subject_dir.name.replace("sub-P", ""))
                subject_ids.append(subject_id)
            except ValueError:
                print(f"⚠️ Skipping invalid subject directory: {subject_dir.name}")
                continue

    return sorted(subject_ids)


def load_single_run(file_path: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Load a single XDF file and extract EEG and marker streams.

    Parameters:
        file_path (str): Path to the XDF file

    Returns:
        Tuple[Optional[Dict], Optional[Dict]]: EEG data dict and marker data dict, or (None, None) if failed
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None, None

    try:
        print(f"Loading file: {file_path}")
        streams, _ = pyxdf.load_xdf(file_path)

        eeg_stream = None
        marker_stream = None

        for stream in streams:
            stream_type = stream["info"].get("type", [None])[0]
            stream_name = stream["info"].get("name", [None])[0]

            print(
                f"  Stream Name: {stream_name}, Type: {stream_type}, Samples: {len(stream['time_series'])}"
            )

            if stream_type == "EEG":
                eeg_stream = stream
            elif stream_type == "Markers" and stream_name == "MarkerStream":
                marker_stream = stream

        if eeg_stream is None:
            print("  ⚠️ No EEG stream found")
            return None, None

        if marker_stream is None:
            print("  ⚠️ No marker stream found")
            return None, None

        # Extract data
        eeg_data = {
            "data": np.array(eeg_stream["time_series"]),
            "timestamps": np.array(eeg_stream["time_stamps"]),
            "stream_info": eeg_stream["info"],
        }

        marker_data = {
            "values": np.array([int(m[0]) for m in marker_stream["time_series"]]),
            "timestamps": np.array(marker_stream["time_stamps"]),
        }

        print("  ✅ Successfully loaded")
        return eeg_data, marker_data

    except Exception as e:
        print(f"  ⚠️ Failed to load XDF: {e}")
        return None, None


def load_session_data(
    subject_id: int,
    session_config: SessionConfig,
    base_path: str = "/home/alexandra-admin/Documents/CurrentStudy",
) -> Tuple[List[Dict], List[Dict], List[str]]:
    """
    Load all runs for a specific session.

    Parameters:
        subject_id (int): Subject ID (e.g., 401)
        session_config (SessionConfig): Configuration for the session to load
        base_path (str): Base directory for data

    Returns:
        Tuple containing:
            - List of EEG run data dictionaries
            - List of marker run data dictionaries
            - List of channel labels
    """
    subject_str = f"sub-P{subject_id:03d}"
    session_path = Path(base_path) / subject_str / session_config.session_id / "eeg"

    print(f"\n{'='*60}")
    print(
        f"Loading {subject_str} - {session_config.session_id} ({session_config.task_name})"
    )
    print(f"Expected runs: {session_config.n_runs}")
    print(f"Path: {session_path}")
    print(f"{'='*60}")

    if not session_path.exists():
        print(f"❌ Session path does not exist: {session_path}")
        return [], [], []

    eeg_runs = []
    marker_runs = []
    labels = []

    # Load each run
    for run_num in range(1, session_config.n_runs + 1):
        filename = f"{subject_str}_{session_config.session_id}_task-Default_run-{run_num:03d}_eeg.xdf"
        file_path = session_path / filename

        eeg_data, marker_data = load_single_run(str(file_path))

        if eeg_data is not None and marker_data is not None:
            eeg_runs.append(eeg_data)
            marker_runs.append(marker_data)

            # Extract channel labels from first successful run
            if not labels:
                try:
                    labels = get_channel_names_from_xdf(
                        {"info": eeg_data["stream_info"]}
                    )
                    # Remove channels 32-38 (indices 32-39) as in original code
                    if len(labels) > 39:
                        labels = [
                            labels[i] for i in range(len(labels)) if i < 32 or i >= 39
                        ]
                    print(f"\n📊 Channel labels: {labels}")
                except Exception as e:
                    print(f"⚠️ Could not extract channel labels: {e}")
                    labels = []

    print(f"\n✅ Successfully loaded {len(eeg_runs)}/{session_config.n_runs} runs")
    return eeg_runs, marker_runs, labels


def load_subject_data(
    subject_id: int,
    mode: str = "offline",
    base_path: str = "/home/alexandra-admin/Documents/CurrentStudy",
) -> Dict[str, Tuple[List[Dict], List[Dict], List[str]]]:
    """
    Load all sessions for a subject.

    Parameters:
        subject_id (int): Subject ID
        mode (str): "offline" or "online"
        base_path (str): Base directory for data

    Returns:
        Dict mapping session names to (eeg_runs, marker_runs, labels) tuples
    """
    sessions = OFFLINE_SESSIONS if mode == "offline" else ONLINE_SESSIONS
    subject_data = {}

    print(f"\n{'#'*60}")
    print(f"# Loading Subject {subject_id} - {mode.upper()} mode")
    print(f"{'#'*60}")

    for session_config in sessions:
        eeg_runs, marker_runs, labels = load_session_data(
            subject_id, session_config, base_path
        )
        subject_data[session_config.task_name] = (eeg_runs, marker_runs, labels)

    return subject_data


def load_all_subjects(
    subject_ids: Optional[List[int]] = None,
    mode: str = "offline",
    base_path: str = "/home/alexandra-admin/Documents/CurrentStudy",
) -> Dict[int, Dict]:
    """
    Load data for multiple subjects.

    Parameters:
        subject_ids (Optional[List[int]]): List of subject IDs. If None, auto-detect all subjects.
        mode (str): "offline" or "online"
        base_path (str): Base directory for data

    Returns:
        Dict mapping subject IDs to their session data
    """
    if subject_ids is None:
        subject_ids = get_available_subjects(base_path)
        print(f"📁 Auto-detected subjects: {subject_ids}")

    all_subjects_data = {}

    for subject_id in subject_ids:
        subject_data = load_subject_data(subject_id, mode, base_path)
        all_subjects_data[subject_id] = subject_data

    return all_subjects_data


if __name__ == "__main__":
    # Demo: Show available subjects
    print("Available subjects:")
    subjects = get_available_subjects()
    print(subjects)

    # Demo: Load one subject
    if subjects:
        print(f"\nLoading first subject ({subjects[0]})...")
        data = load_subject_data(subjects[0], mode="offline")
        print("\nLoaded sessions:")
        for session_name, (eeg_runs, marker_runs, labels) in data.items():
            print(f"  {session_name}: {len(eeg_runs)} runs, {len(labels)} channels")
