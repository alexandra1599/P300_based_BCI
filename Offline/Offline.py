import os
import sys
import datetime
import numpy as np
from load_data import (
    get_available_subjects,
    load_subject_data,
    OFFLINE_SESSIONS,
    ONLINE_SESSIONS,
)

# Import all your processing functions
from filtering import filtering
from target_epochs import target_epochs
from extract_features import extract_features
from helpers import Tee

import numpy as np
import sys, os, datetime
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    cross_val_score,
    cross_val_predict,
)
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
)
import config
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from xgboost import XGBClassifier
from filtering import filtering
from target_epochs import target_epochs
from run_analysis import run_analysis
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    cross_val_predict,
    LeaveOneGroupOut,
)
from XDawn import (
    XdawnFeaturizer,
    run_leakage_safe_cv_with_xdawn,
)
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from extract_features import extract_features
from xgboost import XGBClassifier
import pickle
from cca import rank_channels_component
from helpers import (
    select_channels,
    print_metrics,
    build_X_y_groups,
    tpr_tnr_product,
    per_run_balance,
    tpr_tnr_from_labels,
    CCAWaveformProjector,
    feature_adapter,
    build_epoch_cube_y_groups,
    norm_1020,
    Tee,
)

import matplotlib.pyplot as plt
from Topoplot import (
    common_vlim,
    p3_mean_amplitude,
    p3_peak_amplitude,
    plot_topo,
    import_montage,
)


def process_subject_data(subject_id: int, mode: str = "offline"):
    """
    Process data for a single subject using new loader.
    Matches the structure of your original main() function.
    """
    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_dir = os.path.join(
        "/home/alexandra-admin/Documents/Offline/offline_logs", f"sub-{subject_id}"
    )
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"offline_train_{timestamp}.txt")

    with open(log_path, "w", buffering=1, encoding="utf-8") as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)

        try:
            np.set_printoptions(precision=3, suppress=True)
            print(f"📄 Logging to: {log_path}")
            print(f"\n{'='*60}")
            print(f"Processing Subject {subject_id}")
            print(f"{'='*60}\n")

            # ===== RESET EVERYTHING PER SUBJECT =====
            nback_pre_target, nback_pre_nontarget = [], []
            nback_post_target, nback_post_nontarget = [], []
            nback_online_target, nback_online_nontarget = [], []

            model_pre_target, model_pre_nontarget = [], []
            model_post_target, model_post_nontarget = [], []

            connectivity_pre_target, connectivity_pre_nontarget = [], []
            connectivity_post_target, connectivity_post_nontarget = [], []
            connectivity_online_target, connectivity_online_nontarget = [], []

            # Load all sessions for this subject
            subject_data = load_subject_data(subject_id, mode=mode)

            # Processing parameters
            fs = 512
            do_car = 1  # Apply CAR
            model_filter = 1  # Offline filter
            session = "Relaxation"  # Session type

            online_channel_names = ["FZ", "CZ", "PZ", "P3", "P4", "POZ"]

            # Get labels from first available session
            labels = None
            for sess_name, (eeg_runs, marker_runs, sess_labels) in subject_data.items():
                if sess_labels:
                    labels = sess_labels
                    break

            if labels is None:
                print("❌ No labels found in any session!")
                return None

            # ===== Process each session type =====
            # Session mapping based on your original code:
            # s=0: Nbackpre (4 runs)
            # s=1: Nback + relax (8 runs)
            # s=2: Nback online (8 runs)

            # Process Nbackpre (ses-S001)
            if "Nbackpre" in subject_data:
                print("\n" + "=" * 60)
                print("Processing Nbackpre (Pre-training)")
                print("=" * 60)

                eeg_runs, marker_runs, _ = subject_data["Nbackpre"]
                task = 1

                for run_idx, (eeg_dict, marker_dict) in enumerate(
                    zip(eeg_runs, marker_runs)
                ):
                    print(f"\n  Run {run_idx + 1}/{len(eeg_runs)}:")

                    eeg = eeg_dict["data"]
                    timestamps = eeg_dict["timestamps"]
                    m_data = marker_dict["values"]
                    m_time = marker_dict["timestamps"]

                    print(f"    EEG shape: {eeg.shape}")
                    print(
                        f"    Markers: {len(m_data)} events, unique: {np.unique(m_data)}"
                    )

                    # Apply filtering - NOW RETURNS UPDATED LABELS
                    filtered_eeg, filtered_labels = filtering(
                        eeg,
                        session=session,
                        filter=model_filter,
                        labels=labels,
                        fs=fs,
                        do_car=do_car,
                    )

                    # Use filtered_labels for channel selection
                    eeg_channels, kept_labels, kept_idx = select_channels(
                        filtered_eeg, filtered_labels, online_channel_names
                    )

                    # Extract epochs
                    segments_target, segments_nontarget = target_epochs(
                        filtered_eeg, m_data, m_time, fs, timestamps, task
                    )

                    # Connectivity (unfiltered)
                    con_target, con_nontarget = target_epochs(
                        eeg, m_data, m_time, fs, timestamps, task
                    )
                    con_target = np.delete(con_target, np.arange(32, 39), axis=1)
                    con_nontarget = np.delete(con_nontarget, np.arange(32, 39), axis=1)

                    # Model input (selected channels)
                    starget, snontarget = target_epochs(
                        eeg_channels, m_data, m_time, fs, timestamps, task
                    )

                    # Store
                    nback_pre_target.append(segments_target)
                    nback_pre_nontarget.append(segments_nontarget)
                    connectivity_pre_target.append(con_target)
                    connectivity_pre_nontarget.append(con_nontarget)
                    model_pre_target.append(starget)
                    model_pre_nontarget.append(snontarget)

                    print(
                        f"    Target epochs: {segments_target.shape}, Non-target: {segments_nontarget.shape}"
                    )

            # Process Nbackpost (ses-S005) - this is "Nback + relax" in original
            if "Nbackpost" in subject_data:
                print("\n" + "=" * 60)
                print("Processing Nbackpost (Post-training)")
                print("=" * 60)

                eeg_runs, marker_runs, _ = subject_data["Nbackpost"]
                task = 1

                for run_idx, (eeg_dict, marker_dict) in enumerate(
                    zip(eeg_runs, marker_runs)
                ):
                    print(f"\n  Run {run_idx + 1}/{len(eeg_runs)}:")

                    eeg = eeg_dict["data"]
                    timestamps = eeg_dict["timestamps"]
                    m_data = marker_dict["values"]
                    m_time = marker_dict["timestamps"]

                    print(f"    EEG shape: {eeg.shape}")
                    print(
                        f"    Markers: {len(m_data)} events, unique: {np.unique(m_data)}"
                    )

                    filtered_eeg, filtered_labels = filtering(
                        eeg,
                        session=session,
                        filter=model_filter,
                        labels=labels,
                        fs=fs,
                        do_car=do_car,
                    )

                    eeg_channels, kept_labels, kept_idx = select_channels(
                        filtered_eeg, filtered_labels, online_channel_names
                    )

                    segments_target, segments_nontarget = target_epochs(
                        filtered_eeg, m_data, m_time, fs, timestamps, task
                    )

                    con_target, con_nontarget = target_epochs(
                        eeg, m_data, m_time, fs, timestamps, task
                    )
                    con_target = np.delete(con_target, np.arange(32, 39), axis=1)
                    con_nontarget = np.delete(con_nontarget, np.arange(32, 39), axis=1)

                    starget, snontarget = target_epochs(
                        eeg_channels, m_data, m_time, fs, timestamps, task
                    )

                    nback_post_target.append(segments_target)
                    nback_post_nontarget.append(segments_nontarget)
                    connectivity_post_target.append(con_target)
                    connectivity_post_nontarget.append(con_nontarget)
                    model_post_target.append(starget)
                    model_post_nontarget.append(snontarget)

                    print(
                        f"    Target epochs: {segments_target.shape}, Non-target: {segments_nontarget.shape}"
                    )

            if "Nbackonline" in subject_data:
                print("\n" + "=" * 60)
                print("Processing Nbackonline (Online)")
                print("=" * 60)

                eeg_runs, marker_runs, _ = subject_data["Nbackonline"]
                task = 2  # Online task

                for run_idx, (eeg_dict, marker_dict) in enumerate(
                    zip(eeg_runs, marker_runs)
                ):
                    print(f"\n  Run {run_idx + 1}/{len(eeg_runs)}:")

                    eeg = eeg_dict["data"]
                    timestamps = eeg_dict["timestamps"]
                    m_data = marker_dict["values"]
                    m_time = marker_dict["timestamps"]

                    print(f"    EEG shape: {eeg.shape}")
                    print(
                        f"    Markers: {len(m_data)} events, unique: {np.unique(m_data)}"
                    )

                    # Apply filtering
                    filtered_eeg, filtered_labels = filtering(
                        eeg,
                        session=session,
                        filter=model_filter,
                        labels=labels,
                        fs=fs,
                        do_car=do_car,
                    )

                    # Extract epochs
                    segments_target, segments_nontarget = target_epochs(
                        filtered_eeg, m_data, m_time, fs, timestamps, task
                    )

                    # Connectivity (unfiltered)
                    con_target, con_nontarget = target_epochs(
                        eeg, m_data, m_time, fs, timestamps, task
                    )
                    con_target = np.delete(con_target, np.arange(32, 39), axis=1)
                    con_nontarget = np.delete(con_nontarget, np.arange(32, 39), axis=1)

                    # Store online data
                    nback_online_target.append(segments_target)
                    nback_online_nontarget.append(segments_nontarget)
                    connectivity_online_target.append(con_target)
                    connectivity_online_nontarget.append(con_nontarget)

                    print(
                        f"    Target epochs: {segments_target.shape}, Non-target: {segments_nontarget.shape}"
                    )

            # ===== CONCATENATE PER SUBJECT =====
            print("\n" + "=" * 60)
            print("Concatenating all runs...")
            print("=" * 60)

            nback_pre_target_all = (
                np.concatenate(nback_pre_target, axis=2) if nback_pre_target else None
            )
            nback_pre_nontarget_all = (
                np.concatenate(nback_pre_nontarget, axis=2)
                if nback_pre_nontarget
                else None
            )
            nback_post_target_all = (
                np.concatenate(nback_post_target, axis=2) if nback_post_target else None
            )
            nback_post_nontarget_all = (
                np.concatenate(nback_post_nontarget, axis=2)
                if nback_post_nontarget
                else None
            )

            connectivity_pre_target_all = (
                np.concatenate(connectivity_pre_target, axis=2)
                if connectivity_pre_target
                else None
            )
            connectivity_pre_nontarget_all = (
                np.concatenate(connectivity_pre_nontarget, axis=2)
                if connectivity_pre_nontarget
                else None
            )
            connectivity_post_target_all = (
                np.concatenate(connectivity_post_target, axis=2)
                if connectivity_post_target
                else None
            )
            connectivity_post_nontarget_all = (
                np.concatenate(connectivity_post_nontarget, axis=2)
                if connectivity_post_nontarget
                else None
            )
            nback_online_target_all = (
                np.concatenate(nback_online_target, axis=2)
                if nback_online_target
                else None
            )
            nback_online_nontarget_all = (
                np.concatenate(nback_online_nontarget, axis=2)
                if nback_online_nontarget
                else None
            )

            connectivity_online_target_all = (
                np.concatenate(connectivity_online_target, axis=2)
                if connectivity_online_target
                else None
            )
            connectivity_online_nontarget_all = (
                np.concatenate(connectivity_online_nontarget, axis=2)
                if connectivity_online_nontarget
                else None
            )

            model_pre_target_all = (
                np.concatenate(model_pre_target, axis=2) if model_pre_target else None
            )
            model_pre_nontarget_all = (
                np.concatenate(model_pre_nontarget, axis=2)
                if model_pre_nontarget
                else None
            )
            model_post_target_all = (
                np.concatenate(model_post_target, axis=2) if model_post_target else None
            )
            model_post_nontarget_all = (
                np.concatenate(model_post_nontarget, axis=2)
                if model_post_nontarget
                else None
            )

            print(f"\n✅ Subject {subject_id} concatenated shapes:")
            if nback_pre_target_all is not None:
                print(f"  nback_pre_target_all: {nback_pre_target_all.shape}")
            if nback_pre_nontarget_all is not None:
                print(f"  nback_pre_nontarget_all: {nback_pre_nontarget_all.shape}")
            if nback_post_target_all is not None:
                print(f"  nback_post_target_all: {nback_post_target_all.shape}")
            if nback_post_nontarget_all is not None:
                print(f"  nback_post_nontarget_all: {nback_post_nontarget_all.shape}")
            if nback_online_target_all is not None:
                print(f"  nback_online_target_all: {nback_online_target_all.shape}")
            if nback_online_nontarget_all is not None:
                print(
                    f"  nback_online_nontarget_all: {nback_online_nontarget_all.shape}"
                )

            print(f"\n✅ Subject {subject_id} ML input shapes:")
            if model_pre_target_all is not None:
                print(f"  model_pre_target_all: {model_pre_target_all.shape}")
            if model_pre_nontarget_all is not None:
                print(f"  model_pre_nontarget_all: {model_pre_nontarget_all.shape}")
            if model_post_target_all is not None:
                print(f"  model_post_target_all: {model_post_target_all.shape}")
            if model_post_nontarget_all is not None:
                print(f"  model_post_nontarget_all: {model_post_nontarget_all.shape}")

            results = {
                "nback_pre_target_all": nback_pre_target_all,
                "nback_pre_nontarget_all": nback_pre_nontarget_all,
                "nback_post_target_all": nback_post_target_all,
                "nback_post_nontarget_all": nback_post_nontarget_all,
                "nback_online_target_all": nback_online_target_all,
                "nback_online_nontarget_all": nback_online_nontarget_all,
                "model_pre_target_all": model_pre_target_all,
                "model_pre_nontarget_all": model_pre_nontarget_all,
                "model_post_target_all": model_post_target_all,
                "model_post_nontarget_all": model_post_nontarget_all,
                "connectivity_pre_target_all": connectivity_pre_target_all,
                "connectivity_pre_nontarget_all": connectivity_pre_nontarget_all,
                "connectivity_post_target_all": connectivity_post_target_all,
                "connectivity_post_nontarget_all": connectivity_post_nontarget_all,
                "connectivity_online_target_all": connectivity_online_target_all,
                "connectivity_online_nontarget_all": connectivity_online_nontarget_all,
                "labels": labels,
                "session": session,
                "kept_labels": kept_labels,
                "fs": fs,
                "model_pre_target": model_pre_target_all,
                "model_pre_nontarget": model_pre_nontarget_all,
                "model_post_target": model_post_target_all,
                "model_post_nontarget": model_post_nontarget_all,
            }

            print(f"\n✅ Subject {subject_id} complete!")
            return results

        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            log_file.close()


def main():
    print("=" * 60)
    print("OFFLINE P300 ANALYSIS")
    print("=" * 60)

    # Specify subjects manually
    SUBJECT_IDS = [401, 402, 202, 312]  # , 403, 203]

    print(f"\n📋 Processing subjects: {SUBJECT_IDS}")

    # Storage for all subjects
    all_subjects_data = {}
    all_subjects_data_online = {}

    # Process each subject
    for subject_id in SUBJECT_IDS:
        try:
            results = process_subject_data(subject_id, mode="offline")
            if results is not None:
                all_subjects_data[subject_id] = results
            results_online = process_subject_data(subject_id, mode="online")
            if results_online is not None:
                all_subjects_data_online[subject_id] = results_online
        except Exception as e:
            print(f"❌ Error processing subject {subject_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print("ALL SUBJECTS LOADED - COMBINING DATA")
    print("=" * 70)

    # Collect data from all subjects
    all_pre_target = []
    all_pre_nontarget = []
    all_post_target = []
    all_post_nontarget = []
    all_online_target = []
    all_online_nontarget = []
    all_connect_pre_target = []
    all_connect_pre_nontarget = []
    all_connect_post_target = []
    all_connect_post_nontarget = []
    all_connect_online_target = []
    all_connect_online_nontarget = []
    all_model_pre_target = []
    all_model_pre_nontarget = []
    all_model_post_target = []
    all_model_post_nontarget = []

    for subject_id, data in all_subjects_data.items():
        if data["nback_pre_target_all"] is not None:
            all_pre_target.append(data["nback_pre_target_all"])
        if data["nback_pre_nontarget_all"] is not None:
            all_pre_nontarget.append(data["nback_pre_nontarget_all"])
        if data["nback_post_target_all"] is not None:
            all_post_target.append(data["nback_post_target_all"])
        if data["nback_post_nontarget_all"] is not None:
            all_post_nontarget.append(data["nback_post_nontarget_all"])
        if data.get("nback_online_target_all") is not None:
            all_online_target.append(data["nback_online_target_all"])
        if data.get("nback_online_nontarget_all") is not None:
            all_online_nontarget.append(data["nback_online_nontarget_all"])
        if data["connectivity_pre_target_all"] is not None:
            all_connect_pre_target.append(data["connectivity_pre_target_all"])
        if data["connectivity_pre_nontarget_all"] is not None:
            all_connect_pre_nontarget.append(data["connectivity_pre_nontarget_all"])
        if data["connectivity_post_target_all"] is not None:
            all_connect_post_target.append(data["connectivity_post_target_all"])
        if data["connectivity_post_nontarget_all"] is not None:
            all_connect_post_nontarget.append(data["connectivity_post_nontarget_all"])
        if data.get("connectivity_online_target_all") is not None:
            all_connect_online_target.append(data["connectivity_online_target_all"])
        if data.get("connectivity_online_nontarget_all") is not None:
            all_connect_online_nontarget.append(
                data["connectivity_online_nontarget_all"]
            )
        if data.get("model_pre_target") is not None:
            all_model_pre_target.append(data["model_pre_target"])
        if data.get("model_pre_nontarget") is not None:
            all_model_pre_nontarget.append(data["model_pre_nontarget"])
        if data.get("model_post_target") is not None:
            all_model_post_target.append(data["model_post_target"])
        if data.get("model_post_nontarget") is not None:
            all_model_post_nontarget.append(data["model_post_nontarget"])

    # Concatenate across ALL subjects (axis=2 = trials)
    nback_pre_target_all = (
        np.concatenate(all_pre_target, axis=2) if all_pre_target else None
    )
    nback_pre_nontarget_all = (
        np.concatenate(all_pre_nontarget, axis=2) if all_pre_nontarget else None
    )
    nback_post_target_all = (
        np.concatenate(all_post_target, axis=2) if all_post_target else None
    )
    nback_post_nontarget_all = (
        np.concatenate(all_post_nontarget, axis=2) if all_post_nontarget else None
    )
    # ADD THESE:
    nback_online_target_all = (
        np.concatenate(all_online_target, axis=2) if all_online_target else None
    )
    nback_online_nontarget_all = (
        np.concatenate(all_online_nontarget, axis=2) if all_online_nontarget else None
    )
    connectivity_pre_target_all = (
        np.concatenate(all_connect_pre_target, axis=2)
        if all_connect_pre_target
        else None
    )
    connectivity_pre_nontarget_all = (
        np.concatenate(all_connect_pre_nontarget, axis=2)
        if all_connect_pre_nontarget
        else None
    )
    connectivity_post_target_all = (
        np.concatenate(all_connect_post_target, axis=2)
        if all_connect_post_target
        else None
    )
    connectivity_post_nontarget_all = (
        np.concatenate(all_connect_post_nontarget, axis=2)
        if all_connect_post_nontarget
        else None
    )
    connectivity_online_target_all = (
        np.concatenate(all_connect_online_target, axis=2)
        if all_connect_online_target
        else None
    )
    connectivity_online_nontarget_all = (
        np.concatenate(all_connect_online_nontarget, axis=2)
        if all_connect_online_nontarget
        else None
    )
    model_pre_target_all = (
        np.concatenate(all_model_pre_target, axis=2) if all_model_pre_target else None
    )
    model_pre_nontarget_all = (
        np.concatenate(all_model_pre_nontarget, axis=2)
        if all_model_pre_nontarget
        else None
    )
    model_post_target_all = (
        np.concatenate(all_model_post_target, axis=2) if all_model_post_target else None
    )
    model_post_nontarget_all = (
        np.concatenate(all_model_post_nontarget, axis=2)
        if all_model_post_nontarget
        else None
    )

    print("\n✅ COMBINED data from all subjects:")
    if nback_pre_target_all is not None:
        print(f"nback_pre_target_all: {nback_pre_target_all.shape}")
    if nback_pre_nontarget_all is not None:
        print(f"nback_pre_nontarget_all: {nback_pre_nontarget_all.shape}")
    if nback_post_target_all is not None:
        print(f"nback_post_target_all: {nback_post_target_all.shape}")
    if nback_post_nontarget_all is not None:
        print(f"nback_post_nontarget_all: {nback_post_nontarget_all.shape}")
    if nback_online_target_all is not None:
        print(f"nback_online_target_all: {nback_online_target_all.shape}")
    if nback_online_nontarget_all is not None:
        print(f"nback_online_nontarget_all: {nback_online_nontarget_all.shape}")

    if model_pre_target_all is not None:
        print("  model_pre_target_all:", model_pre_target_all.shape)
    if model_pre_nontarget_all is not None:
        print("  model_pre_nontarget_all:", model_pre_nontarget_all.shape)
    if model_post_target_all is not None:
        print("  model_post_target_all:", model_post_target_all.shape)
    if model_post_nontarget_all is not None:
        print("  model_post_nontarget_all:", model_post_nontarget_all.shape)

    if connectivity_pre_target_all is not None:
        print(" Connectivity_pre_target_all:", connectivity_pre_target_all.shape)
    if connectivity_pre_nontarget_all is not None:
        print(
            "  connectivity_pre_nontarget_all:",
            connectivity_pre_nontarget_all.shape,
        )
    if connectivity_post_target_all is not None:
        print("  connectivity_post_target_all:", connectivity_post_target_all.shape)
    if connectivity_post_nontarget_all is not None:
        print(
            "  connectivity_post_nontarget_all:",
            connectivity_post_nontarget_all.shape,
        )
    if connectivity_online_target_all is not None:
        print(
            "  connectivity_online_target_all:",
            connectivity_online_target_all.shape,
        )
    if connectivity_online_nontarget_all is not None:
        print(
            "  connectivity_online_nontarget_all:",
            connectivity_online_nontarget_all.shape,
        )

    fs = 512

    print("\n" + "=" * 70)
    print(f"✅ Finished processing {len(all_subjects_data)} subjects")
    print("=" * 70)

    # Get labels from first subject's data
    first_subject_data = next(iter(all_subjects_data.values()))
    labels = first_subject_data["labels"]  # Original 39 labels

    # Remove EOG/AUX channels from labels to match filtered data (32 channels)
    if len(labels) == 39:
        labels = [labels[i] for i in range(len(labels)) if i not in range(32, 39)]

    fs = first_subject_data["fs"]
    session = first_subject_data["session"]
    kept_labels = first_subject_data.get("kept_labels")

    print(f"\n📊 Using labels from first subject:")
    print(f"  Total channels: {len(labels)}")
    print(f"  Labels: {labels}")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Session: {session}")

    plot_number = int(input("How many topoplot do y ou want ?"))

    for i in range(0, plot_number):
        plot = int(
            input(
                "Do you want topoplot of : \n [1] Mean P300 Amplitude \n [2] Peak P300 Amplitude \n"
            )
        )

        montage = import_montage("CA-209-dig.fif")
        alias = {"M1": "TP9", "M2": "TP10"}  # if needed
        if plot == 1:  # MEAN AMPLITUDE
            # PRE OFFLINE
            time_ms = np.arange(nback_pre_target_all.shape[0]) * 1000 / fs
            ga_pre = p3_mean_amplitude(
                nback_pre_target_all, time_ms, tmin=0.24, tmax=0.60
            )

            # POST OFFLINE
            time_ms = np.arange(nback_post_target_all.shape[0]) * 1000 / fs
            ga_post = p3_mean_amplitude(
                nback_post_target_all, time_ms, tmin=0.24, tmax=0.60
            )

            # ONLINE
            time_ms = np.arange(nback_online_target_all.shape[0]) * 1000 / fs
            ga_on = p3_mean_amplitude(
                nback_online_target_all, time_ms, tmin=0.24, tmax=0.60
            )

            vmin, vmax = common_vlim(
                ga_pre,
                ga_post,
                ga_on,
                labels=labels,
                exclude=("M1", "M2", "T7", "T8"),
                symmetric=True,
            )
            plot_topo(
                ga_pre,
                montage,
                labels,
                1,
                "Pre",
                plot,
                # vlim=(-3, 2.5),
                type="P300",
            )
            plot_topo(
                ga_post,
                montage,
                labels,
                1,
                "Pos",
                plot,
                # vlim=(-3, 2.5),
                type="P300",
            )
            plot_topo(
                ga_on,
                montage,
                labels,
                2,
                "Pos",
                plot,
                # vlim=(-3, 2.5),
                type="P300",
            )

        elif plot == 2:  # Peak Amplitude P300

            # PRE OFFLINE
            time_ms = np.arange(nback_pre_target_all.shape[0]) * 1000 / fs
            ga_pre = p3_peak_amplitude(
                nback_pre_target_all, time_ms, tmin=0.25, tmax=0.60
            )

            time_ms_cpp = np.arange(nback_pre_nontarget_all.shape[0]) * 1000 / fs
            cpp_pre = p3_peak_amplitude(
                nback_pre_nontarget_all, time_ms_cpp, tmin=0.2, tmax=0.50
            )

            # POST OFFLINE
            time_ms = np.arange(nback_post_target_all.shape[0]) * 1000 / fs
            ga_post = p3_peak_amplitude(
                nback_post_target_all, time_ms, tmin=0.25, tmax=0.60
            )

            time_ms_cpp = np.arange(nback_post_nontarget_all.shape[0]) * 1000 / fs
            cpp_post = p3_peak_amplitude(
                nback_post_nontarget_all, time_ms_cpp, tmin=0.2, tmax=0.50
            )

            # ONLINE
            time_ms = np.arange(nback_online_target_all.shape[0]) * 1000 / fs
            ga_on = p3_peak_amplitude(
                nback_online_target_all, time_ms, tmin=0.25, tmax=0.60
            )
            time_ms_cpp = np.arange(nback_online_nontarget_all.shape[0]) * 1000 / fs
            cpp_on = p3_peak_amplitude(
                nback_online_nontarget_all, time_ms_cpp, tmin=0.2, tmax=0.50
            )

            plot_topo(
                ga_pre,
                montage,
                labels,
                1,
                "Pre",
                plot,
                type="P300",
            )
            plot_topo(
                ga_post,
                montage,
                labels,
                1,
                "Pos",
                plot,
                type="P300",
            )
            plot_topo(
                ga_on,
                montage,
                labels,
                2,
                "Pos",
                plot,
                type="P300",
            )
            plot_topo(
                cpp_pre,
                montage,
                labels,
                1,
                "Pre",
                plot,
                type="CPP",
            )
            plot_topo(
                cpp_post,
                montage,
                labels,
                1,
                "Pos",
                plot,
                type="CPP",
            )
            plot_topo(
                cpp_on,
                montage,
                labels,
                2,
                "Pos",
                plot,
                type="CPP",
            )

    comparison = int(
        input("Do you want to compare offline to online ERPs?\n [1]Yes\n [2]No\n ")
    )

    if comparison == 1:
        analysis = int(input("How many ERP comparisons do you want to run ?"))
        for i in range(0, analysis):
            run_analysis(
                ID=SUBJECT_IDS,
                session=session,
                labels=labels,
                p300_pre=nback_pre_target_all,
                p300_post=nback_post_target_all,
                nop300_pre=nback_pre_nontarget_all,
                nop300_post=nback_post_nontarget_all,
                p300_online=nback_online_target_all,
                nop300_online=nback_online_nontarget_all,
                comparison=comparison,
                all=all_subjects_data,
            )
    elif comparison == 2:
        analysis = int(input("How many ERP analyses do you want to run ?"))
        for i in range(0, analysis):
            run_analysis(
                ID=SUBJECT_IDS,
                session=session,
                labels=labels,
                p300_pre=nback_pre_target_all,
                p300_post=nback_post_target_all,
                nop300_pre=nback_pre_nontarget_all,
                nop300_post=nback_post_nontarget_all,
                all=all_subjects_data,
                p300_online=None,
                nop300_online=None,
                comparison=None,
            )

    training = int(input("Do you want to train a decoder?\n [1]Yes\n [2]No\n"))
    if training == 1:
        # ----------------------------
        # 0) Choose which run lists to use
        #    (swap these to your PRE/POST sets as needed)
        runs_target = model_post_target_all  # list-like: one element per run
        runs_nontarget = model_post_nontarget_all  # list-like: one element per run
        # ----------------------------

        tpr_tnr_scorer = make_scorer(tpr_tnr_product, greater_is_better=True)

        # Ask if want to apply CCA
        choice = int(
            input(
                "Feature extractor:\n [1] CCA\n [2] Handcrafted (No CCA)\n [3] Xdawn\n> "
            )
        )

        if choice == 1:  # CCA

            E_target = np.concatenate(runs_target, axis=2)  # -> (S, C, sum_T_pos)
            E_nontarget = np.concatenate(runs_nontarget, axis=2)  # -> (S, C, sum_T_neg)

            # Now combine targets + nontargets
            E = np.concatenate([E_target, E_nontarget], axis=2)
            y = np.hstack(
                [
                    np.ones(E_target.shape[2], dtype=int),
                    np.zeros(E_nontarget.shape[2], dtype=int),
                ]
            )

            S, C, T = E.shape
            X_flat = E.transpose(2, 0, 1).reshape(T, S * C)  # carrier matrix (T, S*C)

            # Groups matching your E order
            t_per_run_target = [rt.shape[2] for rt in runs_target]
            t_per_run_nontarget = [rn.shape[2] for rn in runs_nontarget]
            groups = np.hstack(
                [
                    np.hstack(
                        [np.full(Ti, i + 1) for i, Ti in enumerate(t_per_run_target)]
                    ),
                    np.hstack(
                        [np.full(Ti, i + 1) for i, Ti in enumerate(t_per_run_nontarget)]
                    ),
                ]
            ).astype(int)

            print("Projection done")

            pipe = Pipeline(
                [
                    (
                        "cca_wave",
                        CCAWaveformProjector(
                            samples=S,
                            channels=C,
                            n_components=3,
                            max_iter=5000,
                            flatten=False,
                            feature_fn=feature_adapter,
                        ),
                    ),
                    (
                        "clf",
                        XGBClassifier(
                            eval_metric="logloss",
                            random_state=42,
                            n_jobs=1,
                            tree_method="hist",
                        ),
                    ),
                ]
            )

            # === 2) Class weight ===
            pos = int((y == 1).sum())
            neg = int((y == 0).sum())
            pos_weight = neg / max(pos, 1)

            param_grid = {
                "cca_wave__n_components": [2, 3],
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.03, 0.1],
                "clf__subsample": [0.7, 1.0],
                "clf__colsample_bytree": [0.7, 1.0],
                # "clf__gamma": [0, 1],
                # grid scale_pos_weight
                "clf__scale_pos_weight": [pos_weight],
            }

            logo = LeaveOneGroupOut()
            grid = GridSearchCV(
                pipe,
                param_grid,
                scoring=tpr_tnr_scorer,
                cv=logo,
                n_jobs=1,
                refit=False,
                verbose=1,
            )

            # Fit with groups; do NOT prefit CCA on all data beforehand (avoid leakage)
            grid.fit(X_flat, y, groups=groups)

            # Deterministic tie-breaker over cv_results_
            cv = grid.cv_results_
            idx_best = int(np.argmax(cv["mean_test_score"]))
            # If exact ties occur, np.argmax picks the first occurrence (grid order = deterministic).
            best_params = cv["params"][idx_best]

            # Now build the final, deterministic best model and refit once on ALL data
            best_model = Pipeline(
                [
                    (
                        "cca_wave",
                        CCAWaveformProjector(
                            samples=S,
                            channels=C,
                            n_components=best_params["cca_wave__n_components"],
                            max_iter=5000,
                            flatten=False,
                            feature_fn=feature_adapter,
                        ),
                    ),
                    (
                        "clf",
                        XGBClassifier(
                            eval_metric="logloss",
                            random_state=42,
                            n_jobs=1,
                            tree_method="hist",
                            n_estimators=best_params["clf__n_estimators"],
                            max_depth=best_params["clf__max_depth"],
                            learning_rate=best_params["clf__learning_rate"],
                            subsample=best_params["clf__subsample"],
                            colsample_bytree=best_params["clf__colsample_bytree"],
                            scale_pos_weight=best_params["clf__scale_pos_weight"],
                            # If available in your xgboost, uncomment:
                            # deterministic_histogram=True,
                        ),
                    ),
                ]
            )
            best_model.fit(X_flat, y)
            print("✅ Best params (subject-specific, deterministic):", best_params)
            from sklearn.base import clone

            det_best = clone(best_model)
            # lock down the booster
            det_best.named_steps["clf"].set_params(
                n_jobs=1,  # <- single thread
                random_state=42,
                tree_method="hist",
                subsample=1.0,  # keep 1.0 for strict determinism
                colsample_bytree=1.0,
            )
            try:
                det_best.named_steps["clf"].set_params(deterministic_histogram=True)
            except Exception:
                pass

            online_channel_names = [
                "FZ",
                "CZ",
                "PZ",
                "P3",
                "P4",
                "POZ",
            ]

            det_best.fit(X_flat, y)  # refit once, deterministically

            Wc_refit = det_best.named_steps["cca_wave"].Wc_  # (C, K)
            top = rank_channels_component(
                Wc_refit, ch_names=online_channel_names, component=0
            )
            print("Top channels (comp 0):", top[:10], flush=True)

            # === 5) Run-wise CV score summary ===
            scores = cross_val_score(
                det_best,
                X_flat,
                y,
                groups=groups,
                cv=logo,
                scoring=tpr_tnr_scorer,
                n_jobs=1,  # <- was -1
            )
            print(
                f"Run-wise CV TPR×TNR: mean={scores.mean():.3f}  std={scores.std():.3f}"
            )

            # === 6) Per-run breakdown using CV predictions (each run tested by a model trained on the other runs) ===
            y_pred_cv = cross_val_predict(
                best_model, X_flat, y, groups=groups, cv=logo, n_jobs=1
            )
            for r in np.unique(groups):
                m = groups == r
                print(f"\n--- Run {int(r)} (held out) ---")
                print_metrics(y[m], y_pred_cv[m], label="XGBoost:")

            best_params = grid.best_params_

            # Deterministic OOF probabilities
            oof_true = y.copy()
            oof_proba = np.empty_like(y, dtype=float)

            for tr_idx, te_idx in logo.split(X_flat, y, groups):
                est = clone(det_best)
                est.fit(X_flat[tr_idx], y[tr_idx])
                oof_proba[te_idx] = est.predict_proba(X_flat[te_idx])[:, 1]

            # Choose global θ to maximize TPR×TNR (micro)
            ths = np.unique(np.concatenate(([0.0, 1.0], oof_proba)))
            best_thr, best_prod = 0.5, -1.0
            for thr in ths:
                y_hat = (oof_proba >= thr).astype(int)
                _, _, prod = tpr_tnr_from_labels(oof_true, y_hat)
                if prod > best_prod:
                    best_prod, best_thr = prod, float(thr)

            print(
                f"\nChosen GLOBAL threshold (OOF) = {best_thr:.2f}  (TPR×TNR on OOF={best_prod:.3f})"
            )

            # Per-run metrics at that global θ
            for r in np.unique(groups):
                m = groups == r
                y_hat_r = (oof_proba[m] >= best_thr).astype(int)
                tpr, tnr, prod = tpr_tnr_from_labels(oof_true[m], y_hat_r)
                print(
                    f"  Run {int(r)} @ thr={best_thr:.2f}: TPR={tpr:.3f}  TNR={tnr:.3f}  TPR×TNR={prod:.3f}"
                )

            roc_auc = roc_auc_score(oof_true, oof_proba)
            pr_auc = average_precision_score(oof_true, oof_proba)
            prevalence = float(oof_true.mean())
            print(f"CCA overall ROC-AUC = {roc_auc:.3f}")
            print(f"CCA overall PR-AUC  = {pr_auc:.3f} (baseline={prevalence:.2f})")

            fpr, tpr, _ = roc_curve(oof_true, oof_proba)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC (OOF) with CCA (deterministic)")
            plt.legend()
            plt.tight_layout()
            plt.show()

            prec, rec, _ = precision_recall_curve(oof_true, oof_proba)
            plt.figure(figsize=(5, 4))
            plt.plot(rec, prec, label=f"PR (AP={pr_auc:.3f})")
            plt.hlines(
                prevalence,
                0,
                1,
                colors="gray",
                linestyles="--",
                label=f"Baseline={prevalence:.2f}",
            )
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision–Recall (OOF) with CCA (deterministic)")
            plt.legend()
            plt.tight_layout()
            plt.show()

            model_dir = "/home/alexandra-admin/Documents/saved_models_cca"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(
                model_dir, f"p300_model_cca_sub-{SUBJECT_IDS}.pkl"
            )

            saved_threshold = float(best_thr) if "best_thr" in globals() else 0.50

            # Pull the fitted CCA step and its weights
            cca_step = None
            Wc = None
            try:
                if (
                    hasattr(best_model, "named_steps")
                    and "cca_wave" in best_model.named_steps
                ):
                    cca_step = best_model.named_steps["cca_wave"]
                    Wc = getattr(cca_step, "Wc_", None)  # (C, K) or None if not set
            except Exception:
                pass

            # Optional: save a channel ranking snapshot (component 0)
            cca_channel_ranking = None
            try:
                if Wc is not None:
                    cca_channel_ranking = rank_channels_component(
                        Wc, ch_names=online_channel_names, component=0
                    )
            except Exception:
                pass

            # Core feature metadata (keep channel order used for training!)
            clf = best_model.named_steps.get("clf", None)
            n_features = getattr(clf, "n_features_in_", None)

            feature_meta = {
                "channels": list(kept_labels),
                "fs": fs,
                "n_features": n_features,  # <- from the classifier
                "features_per_channel": locals().get("features_per_channel", None),
                "cca": {
                    "n_components": getattr(cca_step, "n_components", None),
                    "samples": getattr(cca_step, "samples", None),
                    "channels": getattr(cca_step, "channels", None),
                    "flatten": getattr(cca_step, "flatten", None),
                    "has_feature_fn": getattr(cca_step, "feature_fn", None) is not None,
                    "Wc": Wc,
                    "channel_ranking_comp0": cca_channel_ranking,
                },
            }
            to_save = {
                "model": best_model,
                "scaler": None,
                "threshold": saved_threshold,
                "feature_meta": feature_meta,
                "subject_id": SUBJECT_IDS,
            }

            with open(model_path, "wb") as f:
                pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"💾 Model saved to {model_path}")
            print(
                f"   threshold={saved_threshold:.2f}, n_features={feature_meta['n_features']}"
            )
            if Wc is not None:
                print(f"   CCA Wc shape: {Wc.shape} (channels x components)")

        elif choice == 2:  # Handcrafted
            # ===== Deterministic subject-specific tuning (LOGO, TPR×TNR) =====

            # 0) Build dataset (no shuffling)
            X, y, groups = build_X_y_groups(runs_target, runs_nontarget, max_runs=None)
            print(X.shape)

            # 1) Class weight (per subject)
            pos = int((y == 1).sum())
            neg = int((y == 0).sum())
            pos_weight = neg / max(pos, 1)

            # 2) Base estimator – single-thread & fixed seed; no stochastic subsampling
            base = XGBClassifier(
                eval_metric="logloss",
                scale_pos_weight=pos_weight,
                random_state=42,
                n_jobs=1,  # <- single thread for determinism
                tree_method="hist",
            )

            # Deterministic, ordered grid
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.03, 0.1],
                "subsample": [1.0],  # <- keep 1.0 for strict determinism
                "colsample_bytree": [1.0],  # <- same
                "gamma": [0, 1],
            }

            logo = LeaveOneGroupOut()
            tpr_tnr_scorer = make_scorer(tpr_tnr_product, greater_is_better=True)

            # 3) GridSearchCV – deterministic: no parallelism, no refit (we refit once explicitly)
            grid = GridSearchCV(
                estimator=base,
                param_grid=param_grid,
                scoring=tpr_tnr_scorer,
                cv=logo,
                n_jobs=1,  # <- deterministic
                refit=False,  # <- we’ll choose + refit once explicitly
                verbose=1,
            )

            # IMPORTANT: pass groups (run-wise splits)
            grid.fit(X, y, groups=groups)

            # Deterministic tie-break (first max by grid order)
            cv = grid.cv_results_
            idx_best = int(np.argmax(cv["mean_test_score"]))
            best_params = cv["params"][idx_best]
            print("✅ Best params (by TPR×TNR):", best_params)

            # 4) Build the final best model and refit once on ALL data (deterministic)
            best_model = XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
                tree_method="hist",
                **best_params,
            )
            best_model.fit(X, y)

            # Optional sanity check: per-run balance
            per_run_balance(y, groups)

            # 5) Leakage-safe OOF probabilities via a single deterministic LOGO loop
            oof_true = y.copy()
            oof_proba = np.empty_like(y, dtype=float)

            for tr_idx, te_idx in logo.split(X, y, groups):
                est = XGBClassifier(
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=1,
                    tree_method="hist",
                    **best_params,
                )
                est.fit(X[tr_idx], y[tr_idx])
                oof_proba[te_idx] = est.predict_proba(X[te_idx])[:, 1]

            # 6) Choose a single global threshold θ to maximize TPR×TNR on OOF
            ths = np.unique(np.concatenate(([0.0, 1.0], oof_proba)))
            best_thr, best_prod = 0.5, -1.0
            for thr in ths:
                y_hat = (oof_proba >= thr).astype(int)
                _, _, prod = tpr_tnr_from_labels(oof_true, y_hat)
                if prod > best_prod:
                    best_prod, best_thr = prod, float(thr)

            print(
                f"\nChosen GLOBAL threshold (OOF) = {best_thr:.2f}  (TPR×TNR on OOF={best_prod:.3f})"
            )

            # 7) Per-run metrics at that GLOBAL threshold (still leakage-safe: OOF)
            per_run_stats = []
            for r in np.unique(groups):
                m = groups == r
                y_true_r = oof_true[m]
                y_hat_r = (oof_proba[m] >= best_thr).astype(int)

                acc = accuracy_score(y_true_r, y_hat_r)
                tpr, tnr, prod = tpr_tnr_from_labels(y_true_r, y_hat_r)

                print(
                    f"  Run {int(r)} @ thr={best_thr:.2f}: "
                    f"Acc={acc:.3f}  TPR={tpr:.3f}  TNR={tnr:.3f}  TPR×TNR={prod:.3f}"
                )
                per_run_stats.append(
                    {
                        "run": int(r),
                        "acc": float(acc),
                        "tpr": float(tpr),
                        "tnr": float(tnr),
                        "prod": float(prod),
                    }
                )
            # 8) ROC / PR from the same OOF vector (deterministic)
            roc_auc = roc_auc_score(oof_true, oof_proba)
            pr_auc = average_precision_score(oof_true, oof_proba)
            prevalence = float(oof_true.mean())
            print(f"Overall ROC-AUC = {roc_auc:.3f}")
            print(f"Overall PR-AUC  = {pr_auc:.3f} (baseline={prevalence:.2f})")

            fpr, tpr_curve, _ = roc_curve(oof_true, oof_proba)
            plt.figure()
            plt.plot(fpr, tpr_curve, label=f"AUC={roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC (OOF, deterministic)")
            plt.legend()
            plt.tight_layout()
            plt.show()

            prec, rec, _ = precision_recall_curve(oof_true, oof_proba)
            plt.figure(figsize=(5, 4))
            plt.plot(rec, prec, label=f"PR (AP={pr_auc:.3f})")
            plt.hlines(
                prevalence,
                0,
                1,
                colors="gray",
                linestyles="--",
                label=f"Baseline={prevalence:.2f}",
            )
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision–Recall (OOF, deterministic)")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # 9) Save exactly what you picked
            model_dir = "/home/alexandra-admin/Documents/saved_models"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"p300_model_sub-{SUBJECT_IDS}.pkl")

            feature_meta = {
                "channels": locals().get("labels", None),
                "fs": locals().get("fs", None),
                "n_features": getattr(best_model, "n_features_in_", None),
            }

            to_save = {
                "model": best_model,
                "scaler": None,
                "threshold": float(best_thr),
                "feature_meta": feature_meta,
                "best_params": best_params,
            }
            with open(model_path, "wb") as f:
                pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"💾 Model saved to {model_path}")
            print(
                f"   threshold={best_thr:.2f}, n_features={feature_meta['n_features']}"
            )

        elif choice == 3:  # XDAWN
            # ===== Deterministic XDAWN (LOGO OOF + single final fit) =====

            # 0) Build epoch cube + labels + groups (no shuffling)
            E, y, groups, S, C = build_epoch_cube_y_groups(
                runs_target, runs_nontarget, max_runs=None, start_gid=1
            )
            # XdawnFeaturizer expects (n_trials, n_channels, n_times)
            X_xdawn = np.transpose(E, (2, 1, 0))  # -> (T, C, S)
            print("\n=== Xdawn (deterministic) ===")

            # 1) Class weight per subject
            pos = int((y == 1).sum())
            neg = int((y == 0).sum())
            pos_weight = neg / max(pos, 1)

            # 2) Deterministic XGB params: no subsampling, single-thread, fixed seed
            xgb_params = dict(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                subsample=1.0,  # <- deterministic
                colsample_bytree=1.0,  # <- deterministic
                reg_lambda=1.0,
                objective="binary:logistic",
                tree_method="hist",
                random_state=42,  # <- deterministic
                n_jobs=1,  # <- deterministic
                scale_pos_weight=pos_weight,
            )

            # 3) Leakage-safe OOF via your helper (sequential LOGO inside)
            res = run_leakage_safe_cv_with_xdawn(
                X=X_xdawn,
                y=y,
                groups=groups,
                sfreq=fs,
                tmin=-0.20,
                p300_window_s=(0.10, 0.80),
                n_components=8,
                xgb_params=xgb_params,  # <- our deterministic params
                feature_fn=extract_features,  # <- your feature extractor
            )

            # 4) Report OOF summary + per-run (deterministic)
            print(f"\nChosen GLOBAL threshold = {res['threshold']:.2f}")
            oof = res["oof_metrics"]
            print(
                f"OOF @thr: ACC={oof['ACC']:.3f}  TPR={oof['TPR']:.3f}  TNR={oof['TNR']:.3f}  "
                f"TPR×TNR={(oof['TPR']*oof['TNR']):.3f}"
            )
            print("\nRun-wise metrics @ global thr:")
            for r, acc, tpr, tnr, prod in res["per_run"]:
                print(
                    f"  Run {r}: Acc={acc:.3f}  TPR={tpr:.3f}  TNR={tnr:.3f}  TPR×TNR={prod:.3f}"
                )
            print(
                f"Run-wise CV TPR×TNR: mean={res['mean_prod']:.3f}  std={res['std_prod']:.3f}"
            )

            # 5) AUCs / curves from the same OOF vector (deterministic)
            oof_true = res["oof_true"]
            oof_proba = res["oof_prob"]

            roc_auc = roc_auc_score(oof_true, oof_proba)
            pr_auc = average_precision_score(oof_true, oof_proba)
            prevalence = float(oof_true.mean())
            print(f"\nXdawn overall ROC-AUC = {roc_auc:.3f}")
            print(f"Xdawn overall PR-AUC  = {pr_auc:.3f} (baseline~{prevalence:.2f})")

            fpr, tpr, _ = roc_curve(oof_true, oof_proba)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], "--", label="Chance")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC (OOF) with XDAWN — deterministic")
            plt.legend()
            plt.tight_layout()
            plt.show()

            prec, rec, _ = precision_recall_curve(oof_true, oof_proba)
            plt.figure()
            plt.plot(rec, prec, label=f"AP={pr_auc:.3f}")
            plt.hlines(
                prevalence,
                0,
                1,
                colors="gray",
                linestyles="--",
                label=f"Baseline={prevalence:.2f}",
            )
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision–Recall (OOF) with XDAWN — deterministic")
            plt.legend()
            plt.tight_layout()
            plt.show()

            # 6) Fit one final end-to-end pipeline on ALL data with the SAME deterministic settings
            xdawn_final = XdawnFeaturizer(
                n_components=8,
                sfreq=fs,
                tmin=-0.20,
                p300_window_s=(0.10, 0.80),
                concat_classes=True,
                feature_fn=extract_features,
            )
            clf_final = XGBClassifier(**xgb_params)
            pipe = Pipeline(
                [
                    ("xdawn", xdawn_final),
                    ("scaler", StandardScaler()),
                    ("clf", clf_final),
                ]
            )
            train_channels = [norm_1020(ch) for ch in config.P300_CHANNEL_NAMES]

            pipe.fit(X_xdawn, y)

            # 7) Package everything for online classification (same as your classify_epoch_once_xdawn bundle)
            bundle = {
                "pipe": pipe,  # sklearn pipeline
                "threshold": float(res["threshold"]),  # θ on P(target)
                "train_channels": list(
                    train_channels
                ),  # preserve exact order if available
                "sfreq": float(fs),
                "featurizer": {
                    "tmin": -0.20,
                    "p300_window_s": (0.10, 0.80),
                    "n_components": 8,
                    "feature_fn_tag": "extract_features@v1",
                },
                "oof_summary": {
                    "oof_metrics": res["oof_metrics"],
                    "per_run": res["per_run"],
                    "mean_prod": res["mean_prod"],
                    "std_prod": res["std_prod"],
                },
                "class_labels": [0, 1],
                "version": "xdawn_pipeline_v1_deterministic",
            }

            model_path = f"/home/alexandra-admin/Documents/saved_models_xdawn/p300_model_xdawn_sub-{SUBJECT_IDS}.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"💾 Saved Xdawn pipeline to {model_path}")
            print(
                f"   θ={bundle['threshold']:.2f}, channels={bundle['train_channels']}"
            )


if __name__ == "__main__":
    main()
