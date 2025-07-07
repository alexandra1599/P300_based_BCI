from load_project_data import load_project_data
from extract_data import extract_data
from remove_aux import remove_aux
from filtering import filtering
from target_epochs import target_epochs
from run_analysis import run_analysis
import numpy as np
import matplotlib.pyplot as plt
import random


def main():
    ID = int(input("Enter Subject ID (e.g., 102): "))
    n_sessions = int(input("How many sessions do you want to load for this subject? "))
    nback_pre_target = []
    nback_pre_nontarget = []
    nback_post_target = []
    nback_post_nontarget = []

    for s in range(n_sessions):
        print(f"\n=== Loading Session {s+1} ===")
        session_input = input("Session (1 = Relaxation, 2 = tACS): ").strip()
        session_map = {"1": "Relaxation", "2": "tACS"}
        session = session_map.get(session_input, session_input)

        print("\nType options:")
        type_options = [
            "Eyes Closed pre",
            "Eyes Closed post",
            "Eyes Closed pre tACS",
            "Eyes Closed post tACS",
            "Eyes Closed find tACS",
            "Eyes Closed pre nothing",
            "Eyes Closed post nothing",
            "Relax",
            "Nothing",
            "Nback",
            "Nback + relax",
            "Nback + tACS",
            "EOG",
        ]
        for i, opt in enumerate(type_options):
            print(f"{i + 1}: {opt}")
        type_index = int(input("Type of the session (number): "))
        data_type = type_options[type_index - 1]

        runs = int(input("Number of runs in the session: "))
        N = int(input("N-back level (input 9 if not N-back): "))
        N_param = None if N == 9 else N

        eeg_runs, marker_runs, labels = load_project_data(
            ID, session, runs, data_type, N_param
        )

        car = input("Apply CAR? [y/n]: ").strip().lower()
        if car == "y":
            do_car = True
        else:
            do_car = False

        session_target = []
        session_nontarget = []

        for run_idx, (eeg_dict, marker_dict) in enumerate(zip(eeg_runs, marker_runs)):
            eeg = eeg_dict["data"]
            timestamps = eeg_dict["timestamps"]
            m_data = marker_dict["values"]
            m_time = marker_dict["timestamps"]

            fs = 512
            filtered_eeg, _ = filtering(
                eeg, session=session, labels=labels, fs=fs, do_car=do_car
            )

            segments_target, segments_nontarget = target_epochs(
                filtered_eeg, m_data, m_time, fs, timestamps
            )

            session_target.append(segments_target)
            session_nontarget.append(segments_nontarget)

        if data_type == "Nback" and session == "Relaxation":
            nback_pre_target = session_target
            nback_pre_nontarget = session_nontarget
        elif data_type == "Nback + relax" and session == "Relaxation":
            nback_post_target = session_target
            nback_post_nontarget = session_nontarget

    # === Concatenate across runs ===
    nback_pre_target_all = (
        np.concatenate(nback_pre_target, axis=2) if nback_pre_target else None
    )
    nback_pre_nontarget_all = (
        np.concatenate(nback_pre_nontarget, axis=2) if nback_pre_nontarget else None
    )
    nback_post_target_all = (
        np.concatenate(nback_post_target, axis=2) if nback_post_target else None
    )
    nback_post_nontarget_all = (
        np.concatenate(nback_post_nontarget, axis=2) if nback_post_nontarget else None
    )

    # === Summary ===
    print("✅ Final concatenated shapes:")
    if nback_pre_target_all is not None:
        print("nback_pre_target_all:", nback_pre_target_all.shape)
    if nback_pre_nontarget_all is not None:
        print("nback_pre_nontarget_all:", nback_pre_nontarget_all.shape)
    if nback_post_target_all is not None:
        print("nback_post_target_all:", nback_post_target_all.shape)
    if nback_post_nontarget_all is not None:
        print("nback_post_nontarget_all:", nback_post_nontarget_all.shape)

    from scipy.signal import savgol_filter

    erp_target = np.mean(nback_pre_target_all, axis=2)  # (samples, channels)
    erp_nontarget = np.mean(nback_pre_nontarget_all, axis=2)

    run_analysis(
        ID=ID,
        session="Relaxation",  # or "tACS"
        labels=labels,
        p300_pre=nback_pre_target_all,
        p300_post=nback_post_target_all,
        nop300_pre=nback_pre_nontarget_all,
        nop300_post=nback_post_nontarget_all,
    )


if __name__ == "__main__":
    main()
