"""
© 2026 Alexandra Mikhael. All Rights Reserved.
"""

import numpy as np
import os
from scipy.stats import spearmanr
import matplotlib as plt
from CPP_Analysis import (
    compare_cpp_conditions,
    cpp_buildup_rate_analysis,
    cpp_latency_distribution_analysis,
    cpp_p300_relationship,
    cpp_onset_analysis,
    compare_cpp_slopes,
    comprehensive_cpp_report,
)
from connectivity import run_complete_connectivity_analysis_unfiltered
from sourcelocal import run_source_localization_analysis
from metrics import (
    p3_metrics,
    n1_p3_ptp,
    single_trial_latency_jitter,
    woody_align_window,
    plot_p300_amplitude_comparison,
)
from filter_trials import filter_trials
from cluster import (
    temporal_cluster_analysis,
    compare_all_three_conditions,
    cluster_latency_and_plot,
)
from Visualize_ERP import (
    time_frequency_analysis,
    difference_wave_analysis,
    latency_distribution_analysis,
    analyze_n100_p300_relationship,
    extract_erp_features_multi_subject,
    hemispheric_lateralization_analysis,
    plot_n100_p300_relationship,
    identify_early_components,
    plot_hemisphere_waveforms_comparison,
    plot_hemisphere_effects_barplot,
    plot_subject_hemisphere_comparison,
)


def trials_time_from_tc_tr(data_tc_tr, ch_idx):
    """data_tc_tr: (time, chan, trials) → returns X: (trials, time) at ch_idx"""
    return data_tc_tr[:, ch_idx, :].T


def run_analysis(
    ID,
    session,
    labels,
    p300_pre,
    p300_post,
    nop300_pre,
    nop300_post,
    p300_online,
    nop300_online,
    fs=512,
    all=None,
    comparison=None,
):
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    output_dir = (
        f"/home/alexandra-admin/Documents/Offline/offline_logs/sub-{ID}/erp_figures"
    )
    os.makedirs(output_dir, exist_ok=True)

    # === Channel indices (adjust based on session) ===
    if session == "tACS":
        pz, cz, c1, c2 = 11, 15, 19, 21
    elif session == "Relaxation":
        nt, fz, pz, cz, cpz, c1, c2 = 9, 6, 26, 15, 20, 20, 21

    prompt = "CPP[1] or P300[2] or Target[3] or Advanced[4] or Cluster[5/6] or Source[7] or Connectivity[8] analysis: "
    c = int(input(prompt))

    time_ms = np.arange(p300_pre.shape[0]) * 1000 / fs

    if c == 1:  # === CPP Analysis ===
        print("\n" + "=" * 60)
        print("CPP (Centro-Parietal Positivity) Analysis")
        print("=" * 60)

        # CPP channels (CP1, CP2 or C1, C2)
        cpp_channels = [c1, c2]
        cpp_channel_names = (
            [labels[c1], labels[c2]] if c1 < len(labels) else ["CP1", "CP2"]
        )
        print(f"Using CPP channels: {cpp_channel_names} (indices: {cpp_channels})")

        if comparison == None:
            # === 1. Basic CPP waveform comparison ===
            print("\n=== Basic CPP Waveform (Pre vs Post) ===")
            cpp_pre = np.mean(nop300_pre[:, cpp_channels, :], axis=1).mean(axis=1)
            cpp_post = np.mean(nop300_post[:, cpp_channels, :], axis=1).mean(axis=1)

            stdpre = np.std(nop300_pre[:, cpp_channels, :], axis=1).mean(
                axis=1
            ) / np.sqrt(nop300_pre.shape[2])
            stdpost = np.std(nop300_post[:, cpp_channels, :], axis=1).mean(
                axis=1
            ) / np.sqrt(nop300_post.shape[2])

            fig = plt.figure(figsize=(10, 6))
            plt.plot(
                time_ms,
                gaussian_filter1d(cpp_pre, sigma=2),
                label="CPP Pre",
                color="blue",
                linewidth=2,
            )
            plt.plot(
                time_ms,
                gaussian_filter1d(cpp_post, sigma=2),
                label="CPP Post",
                color="red",
                linewidth=2,
            )
            plt.fill_between(
                time_ms, cpp_pre - stdpre, cpp_pre + stdpre, color="blue", alpha=0.2
            )
            plt.fill_between(
                time_ms, cpp_post - stdpost, cpp_post + stdpost, color="red", alpha=0.2
            )
            plt.axvspan(200, 500, alpha=0.1, color="yellow", label="CPP window")
            plt.axhline(0, color="k", linestyle="-", linewidth=0.5)
            plt.axvline(0, color="k", linestyle="--", linewidth=1)
            plt.title("CPP Analysis (CP1 + CP2)", fontsize=14, fontweight="bold")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (µV)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            save_path = os.path.join(output_dir, "cpp_pre_vs_post.png")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved: cpp_pre_vs_post.png")
            plt.close(fig)

            # === 2. CPP Latency Distribution (PRE) ===
            print("\n=== CPP Latency Distribution (Pre) ===")
            # Run all CPP analyses with MEM

            # 1. Compare CPP slopes across conditions
            cpp_slope_results = compare_cpp_slopes(
                all,
                times=time_ms,
                channel_indices=[c1, c2],  # CP1, CP2 indices
                buildup_window=(0, 500),
            )

            # 2. CPP-P300 relationship for each condition
            for cond in ["pre", "post"]:
                print(f"\n{'='*70}")
                print(f"CPP-P300 RELATIONSHIP: {cond.upper()}")
                print(f"{'='*70}")

                results = cpp_p300_relationship(
                    all,
                    condition_key_cpp=f"nback_{cond}_nontarget_all",
                    condition_key_p300=f"nback_{cond}_target_all",
                    times=time_ms,
                    cpp_channels=[c1, c2],
                    p300_channel=26,  # Pz
                )

        elif comparison == 1:
            # === WITH ONLINE DATA ===
            print("\n=== CPP Comparison: Pre vs Post vs Online ===")
            # Run all CPP analyses with MEM

            # 1. Compare CPP slopes across conditions
            cpp_slope_results = compare_cpp_slopes(
                all,
                times=time_ms,
                channel_indices=[c1, c2],  # CP1, CP2 indices
                buildup_window=(0, 500),
            )

            # 2. CPP-P300 relationship for each condition
            for cond in ["pre", "post", "online"]:
                print(f"\n{'='*70}")
                print(f"CPP-P300 RELATIONSHIP: {cond.upper()}")
                print(f"{'='*70}")

                results = cpp_p300_relationship(
                    all,
                    condition_key_cpp=f"nback_{cond}_nontarget_all",
                    condition_key_p300=f"nback_{cond}_target_all",
                    times=time_ms,
                    cpp_channels=[c1, c2],
                    p300_channel=26,  # Pz
                )
        print("\n" + "=" * 70)
        print("✅ ENHANCED CPP ANALYSIS COMPLETE!")
        print("=" * 70)

    elif c == 2:  # === P300 Target Analysis ===
        if comparison == None:
            pz_pre_target = np.mean(p300_pre[:, pz, :], axis=1)
            pz_post_target = np.mean(p300_post[:, pz, :], axis=1)

            stdpre = np.std(p300_pre[:, pz, :], axis=1) / np.sqrt(p300_pre.shape[2])
            stdpost = np.std(p300_post[:, pz, :], axis=1) / np.sqrt(p300_post.shape[2])

            plt.figure()
            plt.plot(
                time_ms,
                pz_pre_target,
                # gaussian_filter1d(pz_pre_target, sigma=2),
                label="P300 Pre",
                color="blue",
            )
            plt.plot(
                time_ms,
                pz_post_target,
                # gaussian_filter1d(pz_post_target, sigma=2),
                label="P300 Post",
                color="red",
            )
            plt.fill_between(
                time_ms,
                pz_pre_target - stdpre,
                pz_pre_target + stdpre,
                color="blue",
                alpha=0.2,
            )
            plt.fill_between(
                time_ms,
                pz_post_target - stdpost,
                pz_post_target + stdpost,
                color="red",
                alpha=0.2,
            )
            plt.title("P300 Analysis at Pz")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (µV)")
            plt.legend()
            plt.grid(True)
            plt.show()

            m_pre = p3_metrics(time_ms, pz_pre_target)  # blue
            m_post = p3_metrics(time_ms, pz_post_target)  # red
            print("P300 pre metrics :", m_pre)
            print("P300 post metrics :", m_post)

            # Time-frequency
            print("\n=== Time-Frequency Analysis ===")
            tf_fig, _, _ = time_frequency_analysis(p300_pre, fs=512, channel_idx=pz)
            plt.show()

            # Time-frequency
            print("\n=== Time-Frequency Analysis ===")
            tf_fig, _, _ = time_frequency_analysis(p300_post, fs=512, channel_idx=pz)
            plt.show()

            # Difference wave
            print("\n=== Difference Wave Analysis POST ===")
            fig3 = difference_wave_analysis(
                target_epochs=None,  # Not used for pre_vs_post
                nontarget_epochs=None,
                times=time_ms,
                channels_to_plot=[6, 17, 26],
                labels=labels,
                all=all,
                comparison_type="pre_vs_post",
                condition_prefix="nback_target",  # Will compare nback_pre_target vs nback_post_target
            )

            plt.show()  # <-- Change this
            plt.pause(0.1)

            """
            # Latency distribution
            print("\n=== Latency Distribution Analysis Pre ===")
            lat_fig, lat_stats = latency_distribution_analysis(
                p300_pre, time_ms, 25, window=(250, 600)
            )
            plt.show()
            plt.pause(0.1)

            # Latency distribution
            print("\n=== Latency Distribution Analysis Post ===")
            lat_fig, lat_stats = latency_distribution_analysis(
                p300_post, time_ms, 25, window=(250, 600)
            )
            plt.show()
            plt.pause(0.1)
            """

        elif comparison == 1:
            pz_pre_target = np.mean(p300_pre[:, pz, :], axis=1)
            pz_post_target = np.mean(p300_post[:, pz, :], axis=1)
            pz_online_target = np.mean(p300_online[:, pz, :], axis=1)

            Xpre = p300_pre[:, pz, :]
            clean_pre, reject_idx_pre = filter_trials(Xpre)
            Xpost = p300_post[:, pz, :]
            clean_post, reject_idx_pre = filter_trials(Xpost)
            Xon = p300_online[:, pz, :]
            clean_on, reject_idx_pre = filter_trials(Xon)

            stdpre = np.std(clean_pre, axis=1) / np.sqrt(clean_pre.shape[1])
            stdpost = np.std(clean_post, axis=1) / np.sqrt(clean_post.shape[1])
            stdonline = np.std(clean_on, axis=1) / np.sqrt(clean_on.shape[1])
            print("Size std", stdpre.shape)

            ntimes, ntrials = clean_pre.shape
            print("Time, trials", ntimes, ntrials)
            plt.figure()
            # plot all trials (one line per trial)
            for i in range(ntrials):
                trial = clean_pre[:, i]
                plt.plot(time_ms, trial, color="gray", alpha=0.3)

            # plot average on top
            plt.plot(
                time_ms,
                np.mean(clean_pre, axis=1),
                color="blue",
                linewidth=2,
                label="Average P300",
            )
            plt.title("P300 Pre Offline Single Trial Analysis at Pz")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (µV)")
            plt.legend()
            plt.grid(True)
            plt.show()

            ntimes, ntrials = clean_post.shape
            print("Time, trials", ntimes, ntrials)
            plt.figure()
            # plot all trials (one line per trial)
            for i in range(ntrials):
                trial = clean_post[:, i]
                plt.plot(time_ms, trial, color="gray", alpha=0.3)

            # plot average on top
            plt.plot(
                time_ms,
                np.mean(clean_post, axis=1),
                color="blue",
                linewidth=2,
                label="Average P300",
            )
            plt.title("P300 Post Offline Single Trial Analysis at Pz")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (µV)")
            plt.legend()
            plt.grid(True)
            plt.show()

            ntimes, ntrials = clean_on.shape
            print("Time, trials", ntimes, ntrials)
            plt.figure()
            # plot all trials (one line per trial)
            for i in range(ntrials):
                trial = clean_on[:, i]
                plt.plot(time_ms, trial, color="gray", alpha=0.3)

            # plot average on top
            plt.plot(
                time_ms,
                np.mean(clean_on, axis=1),
                color="blue",
                linewidth=2,
                label="Average P300",
            )
            plt.title("P300 Post Online Single Trial Analysis at Pz")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (µV)")
            plt.legend()
            plt.grid(True)
            plt.show()

            plt.figure()
            plt.plot(
                time_ms,
                np.mean(clean_pre, axis=1),
                # gaussian_filter1d(pz_pre_target, sigma=2),
                label="P300 Pre Offline",
                color="blue",
            )
            plt.plot(
                time_ms,
                np.mean(clean_post, axis=1),
                # gaussian_filter1d(pz_post_target, sigma=2),
                label="P300 Post Offline",
                color="red",
            )
            plt.plot(
                time_ms,
                np.mean(clean_on, axis=1),
                # gaussian_filter1d(pz_online_target, sigma=2),
                label="P300 Online",
                color="green",
            )
            plt.fill_between(
                time_ms,
                np.mean(clean_pre, axis=1) - stdpre,
                np.mean(clean_pre, axis=1) + stdpre,
                color="blue",
                alpha=0.2,
            )
            plt.fill_between(
                time_ms,
                np.mean(clean_post, axis=1) - stdpost,
                np.mean(clean_post, axis=1) + stdpost,
                color="red",
                alpha=0.2,
            )
            plt.fill_between(
                time_ms,
                np.mean(clean_on, axis=1) - stdonline,
                np.mean(clean_on, axis=1) + stdonline,
                color="green",
                alpha=0.2,
            )
            plt.title("P300 Analysis Comparison at Pz")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (µV)")
            plt.legend()
            plt.grid(True)
            plt.show()

            print("Shape raw", pz_pre_target.shape)
            print("Shape clean", clean_pre.shape)

            pre_p300 = np.mean(clean_pre, axis=1)
            post_p300 = np.mean(clean_post, axis=1)
            on_p300 = np.mean(clean_on, axis=1)

            # Amplitude Distribution Per Trial
            fig, data = plot_p300_amplitude_comparison(
                all_subjects_data=all,
                times=time_ms,
                channel="Pz",
                p300_window=(250, 600),
                output_dir="erp_figures",
            )

            m_pre = p3_metrics(time_ms, pre_p300)  # blue
            m_post = p3_metrics(time_ms, post_p300)  # red
            m_on = p3_metrics(time_ms, on_p300)  # green
            print("\n=== P300 Metrics ===")
            print(f"Pre metrics : {m_pre}")
            print(f"Post metrics : {m_post}")
            print(f"Online metrics : {m_on}")

            ptp_pre = n1_p3_ptp(time_ms, pre_p300)
            ptp_post = n1_p3_ptp(time_ms, post_p300)
            ptp_on = n1_p3_ptp(time_ms, on_p300)

            t_ms = np.arange(p300_online.shape[0]) * 1000 / fs

            print(f"X shape (trials×time): {clean_pre.shape}")

            # compute jitter
            jitter_sd_pre, per_trial_lat_pre = single_trial_latency_jitter(
                t_ms, clean_pre.T
            )
            jitter_sd_post, per_trial_lat_post = single_trial_latency_jitter(
                t_ms, clean_post.T
            )
            jitter_sd_on, per_trial_lat_on = single_trial_latency_jitter(
                t_ms, clean_on.T
            )

            print(
                f"Pre jitter SD: {jitter_sd_pre:.2f} ms; n={len(per_trial_lat_pre)} trials"
            )
            print(
                f"Online jitter SD: {jitter_sd_post:.2f} ms; n={len(per_trial_lat_post)} trials"
            )
            print(
                f"Online jitter SD: {jitter_sd_on:.2f} ms; n={len(per_trial_lat_on)} trials"
            )

            # ---- Peak-to-peak amplitude (N2–P3) ----
            print("\n=== N1–P3 Peak-to-Peak Amplitude (µV) ===")
            print(f"Offline PRE:   {ptp_pre:.3f} µV")
            print(f"Offline POST:  {ptp_post:.3f} µV")
            print(f"Online (BCI):  {ptp_on:.3f} µV")

            # ---- Latency jitter ----
            print("\n=== Single-Trial Latency Jitter ===")
            print(f"Pre jitter SD: {jitter_sd_pre:.2f} ms")
            print(f"Number of trials: {len(per_trial_lat_pre)}")
            print(f"Post jitter SD: {jitter_sd_post:.2f} ms")
            print(f"Number of trials: {len(per_trial_lat_post)}")
            print(f"Online jitter SD: {jitter_sd_on:.2f} ms")
            print(f"Number of trials: {len(per_trial_lat_on)}")

            # Optional: Show summary statistics of per-trial latencies
            print(f"Pre Mean latency:  {np.mean(per_trial_lat_pre):.2f} ms")
            print(f"Pre Min latency:   {np.min(per_trial_lat_pre):.2f} ms")
            print(f"Pre Max latency:   {np.max(per_trial_lat_pre):.2f} ms")
            print(f"Post Mean latency:  {np.mean(per_trial_lat_post):.2f} ms")
            print(f"Post Min latency:   {np.min(per_trial_lat_post):.2f} ms")
            print(f"Post Max latency:   {np.max(per_trial_lat_post):.2f} ms")
            print(f"Online Mean latency:  {np.mean(per_trial_lat_on):.2f} ms")
            print(f"Online Min latency:   {np.min(per_trial_lat_on):.2f} ms")
            print(f"Online Max latency:   {np.max(per_trial_lat_on):.2f} ms")

            print("\n=== Woody Aligned Results ===")
            X = clean_pre.T
            Xw_pre, shifts_ms_pre = woody_align_window(
                t_ms, X, window=(300, 600), max_shift_ms=80
            )
            erp_pre_aligned = Xw_pre.mean(axis=0)
            X = clean_post.T
            Xw_post, shifts_ms_post = woody_align_window(
                t_ms, X, window=(300, 600), max_shift_ms=80
            )
            erp_post_aligned = Xw_post.mean(axis=0)
            X = clean_on.T
            Xw, shifts_ms_on = woody_align_window(
                t_ms, X, window=(300, 600), max_shift_ms=80
            )
            erp_online_aligned = Xw.mean(axis=0)

            # Recompute metrics on the aligned ERP
            print(
                f"Alignment SD Pre: {np.std(shifts_ms_pre):.1f} ms (should be close to jitter SD)"
            )
            print(
                f"Alignment SD Post: {np.std(shifts_ms_post):.1f} ms (should be close to jitter SD)"
            )
            print(
                f"Alignment SD: {np.std(shifts_ms_on):.1f} ms (should be close to jitter SD)"
            )

            # per_trial_lat already computed (ms)
            trial_idx = np.arange(len(per_trial_lat_pre)) + 1
            rho, p = spearmanr(trial_idx, per_trial_lat_pre)
            print(f"Latency drift PRE : Spearman r={rho:.2f}, p={p:.3g}")

            trial_idx = np.arange(len(per_trial_lat_post)) + 1
            rho, p = spearmanr(trial_idx, per_trial_lat_post)
            print(f"Latency drift POST: Spearman r={rho:.2f}, p={p:.3g}")

            trial_idx = np.arange(len(per_trial_lat_on)) + 1
            rho, p = spearmanr(trial_idx, per_trial_lat_on)
            print(f"Latency drift ON: Spearman r={rho:.2f}, p={p:.3g}")

    elif c == 3:  # === Target vs Non-Target (Pre and Post) ===
        if comparison == None:
            for label, targ, nontarg in zip(
                ["Pre", "Post"], [p300_pre, p300_post], [nop300_pre, nop300_post]
            ):
                target = np.mean(targ[:, pz, :], axis=1)
                nontarget = np.mean(nontarg[:, nt, :], axis=1)

                std_target = np.std(targ[:, pz, :], axis=1) / np.sqrt(targ.shape[2])
                std_nontarget = np.std(nontarg[:, nt, :], axis=1) / np.sqrt(
                    nontarg.shape[2]
                )

                plt.figure()
                plt.plot(
                    time_ms,
                    gaussian_filter1d(target, sigma=2),
                    label=f"{label} Target",
                    color="blue",
                )
                plt.plot(
                    time_ms,
                    gaussian_filter1d(nontarget, sigma=2),
                    label=f"{label} Non-Target",
                    color="red",
                )
                plt.fill_between(
                    time_ms,
                    target - std_target,
                    target + std_target,
                    color="blue",
                    alpha=0.2,
                )
                plt.fill_between(
                    time_ms,
                    nontarget - std_nontarget,
                    nontarget + std_nontarget,
                    color="red",
                    alpha=0.2,
                )
                plt.title(f"{label} P300: Target vs Non-Target")
                plt.xlabel("Time (ms)")
                plt.ylabel("Amplitude (µV)")
                plt.legend()
                plt.grid(True)

                plt.show()

        elif comparison == 1:
            for label, targ, nontarg in zip(
                ["Pre", "Post", "Online"],
                [p300_pre, p300_post, p300_online],
                [nop300_pre, nop300_post, nop300_online],
            ):
                target = np.mean(targ[:, pz, :], axis=1)
                nontarget = np.mean(nontarg[:, nt, :], axis=1)

                std_target = np.std(targ[:, pz, :], axis=1) / np.sqrt(targ.shape[2])
                std_nontarget = np.std(nontarg[:, nt, :], axis=1) / np.sqrt(
                    nontarg.shape[2]
                )

                plt.figure()
                plt.plot(
                    time_ms,
                    gaussian_filter1d(target, sigma=2),
                    label=f"{label} Target",
                    color="blue",
                )
                plt.plot(
                    time_ms,
                    gaussian_filter1d(nontarget, sigma=2),
                    label=f"{label} Non-Target",
                    color="red",
                )
                plt.fill_between(
                    time_ms,
                    target - std_target,
                    target + std_target,
                    color="blue",
                    alpha=0.2,
                )
                plt.fill_between(
                    time_ms,
                    nontarget - std_nontarget,
                    nontarget + std_nontarget,
                    color="red",
                    alpha=0.2,
                )
                plt.title(f"{label} P300: Target vs Non-Target")
                plt.xlabel("Time (ms)")
                plt.ylabel("Amplitude (µV)")
                plt.legend()
                plt.grid(True)

                plt.show()

    elif c == 4:
        print("\n" + "=" * 70)
        print("TARGETED ADVANCED ANALYSES")
        print("=" * 70)

        # Create output directory for targeted analyses
        targeted_dir = os.path.join(output_dir, "targeted_analysis")
        os.makedirs(targeted_dir, exist_ok=True)

        # Get channel indices
        if session == "tACS":
            pz = 11
        elif session == "Relaxation":
            pz = 26

        time_ms = np.arange(p300_pre.shape[0]) * 1000 / fs

        # ===== 1. N100-P300 RELATIONSHIP =====
        print("\n=== 1. N100-P300 Temporal Relationship ===")
        try:
            # Analyze for Pre condition
            print("\n=== PRE ===")
            df_target = extract_erp_features_multi_subject(
                all,
                "nback_pre_target_all",
                channel_idx=26,  # Pz
                n100_window=(100, 250),
                p300_window=(300, 600),
                times=time_ms,
            )
            print(f"Total trials: {len(df_target)}")
            print(f"Subjects: {df_target['subject'].nunique()}")
            print(f"Trials per subject: {df_target.groupby('subject').size()}")

            save_path = os.path.join(targeted_dir, "n100_p300_relationship_pre.png")
            # Run mixed effects models
            results = analyze_n100_p300_relationship(df_target)

            # Access specific results
            print("\n=== N100 → P300 Amplitude Effect ===")
            print(f"Coefficient: {results['n100_to_p300_amp'].params['n100_amp']:.4f}")
            print(f"P-value: {results['n100_to_p300_amp'].pvalues['n100_amp']:.4f}")
            print(
                f"95% CI: {results['n100_to_p300_amp'].conf_int().loc['n100_amp'].values}"
            )

            # Create plot
            fig, df_filtered = plot_n100_p300_relationship(df_target, results)

            # Now you can save the figure:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.show()

            # Analyze for Post condition
            print("\n=== POST ===")
            df_target = extract_erp_features_multi_subject(
                all,
                "nback_post_target_all",
                channel_idx=26,  # Pz
                n100_window=(100, 250),
                p300_window=(300, 600),
                times=time_ms,
            )
            print(f"Total trials: {len(df_target)}")
            print(f"Subjects: {df_target['subject'].nunique()}")
            print(f"Trials per subject: {df_target.groupby('subject').size()}")

            save_path = os.path.join(targeted_dir, "n100_p300_relationship_post.png")
            # Run mixed effects models
            results = analyze_n100_p300_relationship(df_target)

            # Access specific results
            print("\n=== N100 → P300 Amplitude Effect ===")
            print(f"Coefficient: {results['n100_to_p300_amp'].params['n100_amp']:.4f}")
            print(f"P-value: {results['n100_to_p300_amp'].pvalues['n100_amp']:.4f}")
            print(
                f"95% CI: {results['n100_to_p300_amp'].conf_int().loc['n100_amp'].values}"
            )

            # Create plot
            # Create plot
            fig, df_filtered = plot_n100_p300_relationship(df_target, results)

            # Now you can save the figure:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.show()

            # If online data available
            if comparison == 1:
                print("\n=== ONLINE ===")
                df_target = extract_erp_features_multi_subject(
                    all,
                    "nback_online_target_all",
                    channel_idx=26,  # Pz
                    n100_window=(100, 250),
                    p300_window=(300, 600),
                    times=time_ms,
                )
                print(f"Total trials: {len(df_target)}")
                print(f"Subjects: {df_target['subject'].nunique()}")
                print(f"Trials per subject: {df_target.groupby('subject').size()}")

                save_path = os.path.join(
                    targeted_dir, "n100_p300_relationship_online.png"
                )
                # Run mixed effects models
                results = analyze_n100_p300_relationship(df_target)

                # Access specific results
                print("\n=== N100 → P300 Amplitude Effect ===")
                print(
                    f"Coefficient: {results['n100_to_p300_amp'].params['n100_amp']:.4f}"
                )
                print(f"P-value: {results['n100_to_p300_amp'].pvalues['n100_amp']:.4f}")
                print(
                    f"95% CI: {results['n100_to_p300_amp'].conf_int().loc['n100_amp'].values}"
                )

                # Create plot
                # Create plot
                fig, df_filtered = plot_n100_p300_relationship(df_target, results)

                # Now you can save the figure:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.show()

        except Exception as e:
            print(f"❌ Error in N100-P300 analysis: {e}")
            import traceback

            traceback.print_exc()

        # ===== 2. HEMISPHERIC LATERALIZATION =====
        print("\n=== 2. Hemispheric Lateralization Analysis ===")
        try:
            # Define left/right channels based on session
            if session == "tACS":
                # Adjust these based on your actual channel layout
                left_channels = ["P3", "CP1", "C3"]
                right_channels = ["P4", "CP2", "C4"]
            elif session == "Relaxation":
                left_channels = ["P3"]  # , "CP1", "P7"]
                right_channels = ["P4"]  # , "CP2", "P8"]

            # Pre condition
            # Run analyses for all conditions
            results_pre = hemispheric_lateralization_analysis(
                all,
                "nback_pre_target_all",
                left_channels,
                right_channels,
                labels,
                time_ms,
            )

            results_post = hemispheric_lateralization_analysis(
                all,
                "nback_post_target_all",
                left_channels,
                right_channels,
                labels,
                time_ms,
            )

            # Online condition (if available)
            if comparison == 1:
                results_online = hemispheric_lateralization_analysis(
                    all,
                    "nback_online_target_all",
                    left_channels,
                    right_channels,
                    labels,
                    time_ms,
                )

                results_dict = {
                    "pre": results_pre,
                    "post": results_post,
                    "online": results_online,
                }

                # Create figures
                fig1 = plot_hemisphere_waveforms_comparison(results_dict, time_ms)
                fig2 = plot_hemisphere_effects_barplot(results_dict)
                fig3 = plot_subject_hemisphere_comparison(results_dict)
                plt.show()

        except Exception as e:
            print(f"❌ Error in lateralization analysis: {e}")
            import traceback

            traceback.print_exc()

        print(f"\n📁 All targeted analyses saved to: {targeted_dir}")
        print("=" * 70)

    elif c == 5:  # === TEMPORAL CLUSTERING ANALYSIS ===
        print("\n" + "=" * 70)
        print("TEMPORAL CLUSTERING ANALYSIS - LEARNING PROGRESSION")
        print("=" * 70)

        # Check if we have online data
        if comparison != 1 or p300_online is None:
            print("❌ This analysis requires online data")
            return

        n_online_trials = p300_online.shape[2]
        print(f"\nTotal online P300 trials: {n_online_trials}")

        trials_per_run = 10
        expected_total = trials_per_run * 8

        if n_online_trials != expected_total:
            print(f"⚠️ Expected {expected_total} trials but found {n_online_trials}")
            trials_per_run = int(
                input(f"Trials per run? [default: {n_online_trials//8}]: ")
                or n_online_trials // 8
            )

        online_run_labels = np.repeat(np.arange(1, 9), trials_per_run)

        if len(online_run_labels) < n_online_trials:
            remainder = n_online_trials - len(online_run_labels)
            online_run_labels = np.concatenate(
                [online_run_labels, np.full(remainder, 8)]
            )

        online_run_labels = online_run_labels[:n_online_trials]

        print(f"\n✓ Assigned run labels:")
        for run_num in range(1, 9):
            count = np.sum(online_run_labels == run_num)
            print(f"   Run {run_num}: {count} trials")

        # ============================================================
        # PREPARE WAVEFORM DATA FOR CLUSTERING
        # ============================================================

        print("\n=== Preparing Waveform Data for Clustering ===")

        # Get Pz index
        if session == "tACS":
            pz_idx = 11
        elif session == "Relaxation":
            pz_idx = 26
        else:
            pz_idx = 11

        # Option 1: Use single channel (Pz)
        print(f"Using channel: {labels[pz_idx] if pz_idx < len(labels) else 'Pz'}")

        # Extract waveforms at Pz: (trials, time)
        waveforms_pz = p300_online[:, pz_idx, :].T  # Shape: (trials, time)

        print(f"✓ Waveform data shape: {waveforms_pz.shape}")
        print(
            f"   ({waveforms_pz.shape[0]} trials × {waveforms_pz.shape[1]} timepoints)"
        )

        # Optional: Focus on P300 time window only (300-600ms)
        p300_window_start = int(0.3 * fs)  # 300ms
        p300_window_end = int(0.6 * fs)  # 600ms

        waveforms_window = waveforms_pz[:, p300_window_start:p300_window_end]

        print(f"✓ Using P300 window (300-600ms): {waveforms_window.shape[1]} samples")

        # Use windowed waveforms as features
        online_features = waveforms_window  # Shape: (trials, time_samples)
        trial_indices = np.arange(n_online_trials)

        # ============================================================
        # ALTERNATIVE: Use Multiple Channels
        # ============================================================

        use_multi_channel = (
            input("\nUse multiple channels? [y/n, default: n]: ").lower() == "y"
        )

        if use_multi_channel:
            # Use central-parietal channels: Cz, CPz, Pz
            if session == "Relaxation":
                ch_indices = [15, 20, 26]  # Cz, CPz, Pz
                ch_names = ["Cz", "CPz", "Pz"]
            else:
                ch_indices = [pz_idx]  # Just Pz if others not available
                ch_names = [labels[pz_idx] if pz_idx < len(labels) else "Pz"]

            print(f"Using channels: {ch_names}")

            # Extract waveforms from multiple channels
            multi_ch_waveforms = []
            for ch_idx in ch_indices:
                ch_waveforms = p300_online[:, ch_idx, :].T  # (trials, time)
                ch_window = ch_waveforms[:, p300_window_start:p300_window_end]
                multi_ch_waveforms.append(ch_window)

            # Concatenate channels: (trials, time*channels)
            online_features = np.hstack(multi_ch_waveforms)
            print(f"✓ Multi-channel features: {online_features.shape}")

        # ============================================================
        # TEMPORAL CLUSTERING ANALYSIS
        # ============================================================

        print("\n=== Performing Temporal Clustering on Waveforms ===")

        cluster_labels, cluster_info = temporal_cluster_analysis(
            features=online_features,  # Now using raw waveforms!
            run_labels=online_run_labels,
            trial_indices=trial_indices,
            n_clusters=None,  # Auto-detect
        )

        # ============================================================
        # VISUALIZE CLUSTER WAVEFORMS
        # ============================================================

        print("\n=== Analyzing Cluster Waveforms ===")

        n_clusters = cluster_info["n_clusters"]
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

        fig_erp, ax_erp = plt.subplots(figsize=(10, 6))

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id

            # Get full waveforms for this cluster
            cluster_trials = p300_online[:, pz_idx, cluster_mask]

            # Compute ERP
            cluster_erp = cluster_trials.mean(axis=1)
            cluster_sem = cluster_trials.std(axis=1) / np.sqrt(cluster_trials.shape[1])

            # Plot
            ax_erp.plot(
                time_ms,
                cluster_erp,
                label=f"Cluster {cluster_id} (n={np.sum(cluster_mask)})",
                color=colors[cluster_id],
                linewidth=2,
            )
            ax_erp.fill_between(
                time_ms,
                cluster_erp - cluster_sem,
                cluster_erp + cluster_sem,
                color=colors[cluster_id],
                alpha=0.2,
            )

        ax_erp.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
        ax_erp.axvspan(300, 600, alpha=0.1, color="green", label="P300 window")
        ax_erp.set_xlabel("Time (ms)", fontsize=12)
        ax_erp.set_ylabel("Amplitude (µV)", fontsize=12)
        ax_erp.set_title(
            f'ERP by Cluster at {labels[pz_idx] if pz_idx < len(labels) else "Pz"}',
            fontsize=14,
            fontweight="bold",
        )
        ax_erp.legend()
        ax_erp.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # ============================================================
        # COMPUTE P300 METRICS PER CLUSTER
        # ============================================================

        print(f"\n{'='*60}")
        print("P300 METRICS BY CLUSTER")
        print(f"{'='*60}")

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_trials = p300_online[:, pz_idx, cluster_mask]
            cluster_erp = cluster_trials.mean(axis=1)

            # Compute P300 metrics
            metrics = p3_metrics(time_ms, cluster_erp)

            print(f"\nCluster {cluster_id} (n={np.sum(cluster_mask)}):")
            print(f"  Peak amplitude: {metrics['peak_uv']:.2f} µV")
            print(f"  Peak latency:   {metrics['latency_ms']:.1f} ms")
            print(f"  FWHM:           {metrics['fwhm_ms']:.1f} ms")
            print(f"  Area:           {metrics['area_uv_ms']:.1f} µV·ms")
            print(f"  SNR:            {metrics['snr']:.2f}")

        print(f"{'='*60}")

        # ============================================================
        # SAVE RESULTS
        # ============================================================

        cluster_dir = os.path.join(output_dir, "temporal_clustering")
        os.makedirs(cluster_dir, exist_ok=True)

        np.savez(
            os.path.join(cluster_dir, f"clustering_waveforms_{ID}_{session}.npz"),
            cluster_labels=cluster_labels,
            run_labels=online_run_labels,
            trial_indices=trial_indices,
            waveforms=waveforms_pz,  # Save full waveforms
            waveforms_windowed=waveforms_window,  # P300 window only
            n_clusters=cluster_info["n_clusters"],
            cluster_by_run=cluster_info["cluster_by_run"],
            subject_id=ID,
            session=session,
            pz_idx=pz_idx,
        )

        print(f"\n✅ Results saved to: {cluster_dir}")

        # ============================================================
        # LEARNING PROGRESSION ANALYSIS
        # ============================================================

        print(f"\n{'='*70}")
        print("LEARNING PROGRESSION SUMMARY")
        print(f"{'='*70}")

        cluster_avg_runs = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            avg_run = np.mean(online_run_labels[cluster_mask])
            cluster_avg_runs.append(avg_run)

        sorted_clusters = np.argsort(cluster_avg_runs)

        print("\nClusters ordered by temporal progression:")
        for idx, cluster_id in enumerate(sorted_clusters):
            avg_run = cluster_avg_runs[cluster_id]
            cluster_mask = cluster_labels == cluster_id

            cluster_trials = p300_online[:, pz_idx, cluster_mask]
            cluster_erp = cluster_trials.mean(axis=1)
            metrics = p3_metrics(time_ms, cluster_erp)

            if avg_run < 3:
                phase = "EARLY (Learning)"
            elif avg_run > 6:
                phase = "LATE (Expert)"
            else:
                phase = "MIDDLE (Transition)"

            print(f"\n{idx+1}. Cluster {cluster_id} - {phase}")
            print(f"   Average run: {avg_run:.2f}")
            print(f"   P300 amplitude: {metrics['peak_uv']:.2f} µV")
            print(f"   P300 latency: {metrics['latency_ms']:.1f} ms")

        # Learning effect
        early_cluster = sorted_clusters[0]
        late_cluster = sorted_clusters[-1]

        early_trials = p300_online[:, pz_idx, cluster_labels == early_cluster]
        late_trials = p300_online[:, pz_idx, cluster_labels == late_cluster]

        early_metrics = p3_metrics(time_ms, early_trials.mean(axis=1))
        late_metrics = p3_metrics(time_ms, late_trials.mean(axis=1))

        print(f"\n{'='*70}")
        print("LEARNING EFFECT (Early vs Late):")
        print(f"{'='*70}")
        print(
            f"Amplitude: {early_metrics['peak_uv']:.2f} → {late_metrics['peak_uv']:.2f} µV"
        )
        print(f"  Δ = {late_metrics['peak_uv'] - early_metrics['peak_uv']:+.2f} µV")
        print(
            f"Latency: {early_metrics['latency_ms']:.1f} → {late_metrics['latency_ms']:.1f} ms"
        )
        print(
            f"  Δ = {late_metrics['latency_ms'] - early_metrics['latency_ms']:+.1f} ms"
        )

        if late_metrics["peak_uv"] > early_metrics["peak_uv"] * 1.1:
            print("\n✓ P300 amplitude INCREASED with learning!")
        elif late_metrics["peak_uv"] < early_metrics["peak_uv"] * 0.9:
            print("\n⚠️ P300 amplitude DECREASED")
        else:
            print("\n→ P300 amplitude stable")

        print(f"{'='*70}")
        print("✅ TEMPORAL CLUSTERING COMPLETE")
        print(f"{'='*70}")

    elif c == 6:  # === LATENCY-BASED CLUSTERING ===

        print("\n" + "=" * 70)
        print("LATENCY-BASED CLUSTERING ANALYSIS")
        print("=" * 70)

        # ============================================================
        # SELECT CONDITION
        # ============================================================

        print("\nSelect condition to analyze:")
        print("  [1] Pre (offline)")
        print("  [2] Post (offline)")
        print("  [3] Online (across runs)")
        print("  [4] Compare all three")

        condition_choice = int(input("Choice: "))

        # Map choices to data
        condition_map = {
            1: ("Pre", p300_pre, None),
            2: ("Post", p300_post, None),
            3: ("Online", p300_online, "runs"),
            4: ("All", None, "compare"),
        }

        if condition_choice not in condition_map:
            print("❌ Invalid choice")
            return

        condition_name, p300_data, special = condition_map[condition_choice]

        # Create output directory
        latency_dir = os.path.join(output_dir, "latency_clustering")
        os.makedirs(latency_dir, exist_ok=True)

        # ============================================================
        # GET PZ INDEX
        # ============================================================

        if session == "tACS":
            pz_idx = 11
        elif session == "Relaxation":
            pz_idx = 26
        else:
            pz_idx = 11

        pz_name = labels[pz_idx] if pz_idx < len(labels) else "Pz"

        # ============================================================
        # DEFINE CLUSTERING FUNCTION
        # ============================================================

        def cluster_by_latency(
            p300_data, time_ms, pz_idx, condition_name, run_labels=None, save_dir=None
        ):
            """
            Cluster trials by P300 latency

            Returns: dict with results
            """
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            from scipy.stats import linregress

            # Get trials × time at Pz
            X_pz = p300_data[:, pz_idx, :].T  # (trials, time)
            n_trials = X_pz.shape[0]

            print(f"\n{'='*60}")
            print(f"ANALYZING: {condition_name}")
            print(f"{'='*60}")
            print(f"Total trials: {n_trials}")

            # Define windows
            p300_window = (time_ms >= 300) & (time_ms <= 600)
            n200_window = (time_ms >= 150) & (time_ms <= 250)

            # Extract latencies and amplitudes
            per_trial_latencies = []
            per_trial_amplitudes = []
            per_trial_is_positive = []

            for trial_idx in range(n_trials):
                trial_waveform = X_pz[trial_idx, :]
                trial_p300_window = trial_waveform[p300_window]

                # Check polarity (mean in window)
                mean_in_window = trial_p300_window.mean()

                if mean_in_window > 0:
                    # Positive P300: find max
                    peak_idx = np.argmax(trial_p300_window)
                    peak_amplitude = trial_p300_window[peak_idx]
                    is_positive = True
                else:
                    # Negative/absent: find min
                    peak_idx = np.argmin(trial_p300_window)
                    peak_amplitude = trial_p300_window[peak_idx]
                    is_positive = False

                peak_latency = time_ms[p300_window][peak_idx]

                per_trial_latencies.append(peak_latency)
                per_trial_amplitudes.append(peak_amplitude)
                per_trial_is_positive.append(is_positive)

            per_trial_latencies = np.array(per_trial_latencies)
            per_trial_amplitudes = np.array(per_trial_amplitudes)
            per_trial_is_positive = np.array(per_trial_is_positive)

            # Report polarity
            n_positive = np.sum(per_trial_is_positive)
            n_negative = n_trials - n_positive

            print(f"\n✓ Extracted {n_trials} trials:")
            print(f"   Positive P300: {n_positive} ({100*n_positive/n_trials:.0f}%)")
            print(
                f"   Negative/artifact: {n_negative} ({100*n_negative/n_trials:.0f}%)"
            )

            if n_positive > 0:
                pos_lats = per_trial_latencies[per_trial_is_positive]
                print(
                    f"   Mean latency (positive only): {pos_lats.mean():.1f} ± {pos_lats.std():.1f} ms"
                )

            # Filter to positive trials only for clustering
            if n_positive < 10:
                print(f"\n⚠️ Only {n_positive} positive P300s - too few for clustering")
                return None

            print(f"\n=== Clustering {n_positive} Positive P300 Trials ===")

            positive_mask = per_trial_is_positive
            latencies_positive = per_trial_latencies[positive_mask]
            amplitudes_positive = per_trial_amplitudes[positive_mask]

            if run_labels is not None:
                run_labels_positive = run_labels[positive_mask]
            else:
                run_labels_positive = None

            # Cluster by latency
            latencies_for_clustering = latencies_positive.reshape(-1, 1)

            # Find optimal k (2-4 clusters)
            silhouette_scores = []
            K_range = range(2, min(5, n_positive // 5))

            for k in K_range:
                kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=20)
                labels_temp = kmeans_temp.fit_predict(latencies_for_clustering)
                sil = silhouette_score(latencies_for_clustering, labels_temp)
                silhouette_scores.append(sil)

            optimal_k = list(K_range)[np.argmax(silhouette_scores)]

            print(f"Optimal k: {optimal_k} (silhouette: {max(silhouette_scores):.3f})")

            # Cluster with optimal k
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
            cluster_labels_positive = kmeans.fit_predict(latencies_for_clustering)

            # Get cluster centers and sort
            cluster_centers = kmeans.cluster_centers_.flatten()
            sorted_cluster_ids = np.argsort(cluster_centers)

            # Name clusters
            cluster_names = {}
            if optimal_k == 2:
                cluster_names = {
                    sorted_cluster_ids[0]: "FAST",
                    sorted_cluster_ids[1]: "SLOW",
                }
            elif optimal_k == 3:
                cluster_names = {
                    sorted_cluster_ids[0]: "FAST",
                    sorted_cluster_ids[1]: "MEDIUM",
                    sorted_cluster_ids[2]: "SLOW",
                }
            else:
                cluster_names = {i: f"C{i}" for i in range(optimal_k)}

            # Print cluster info
            print(f"\n{'='*60}")
            print("LATENCY CLUSTERS (Positive P300s only)")
            print(f"{'='*60}")

            for cluster_id in sorted_cluster_ids:
                cluster_mask_pos = cluster_labels_positive == cluster_id
                cluster_lats = latencies_positive[cluster_mask_pos]
                cluster_amps = amplitudes_positive[cluster_mask_pos]

                print(f"\n{cluster_names[cluster_id]} Cluster (ID={cluster_id}):")
                print(f"  N trials: {np.sum(cluster_mask_pos)}")
                print(
                    f"  Latency: {cluster_lats.mean():.1f} ± {cluster_lats.std():.1f} ms"
                )
                print(
                    f"  Range: {cluster_lats.min():.1f} - {cluster_lats.max():.1f} ms"
                )
                print(
                    f"  Amplitude: {cluster_amps.mean():.2f} ± {cluster_amps.std():.2f} µV"
                )

            print(f"{'='*60}")

            # ============================================================
            # PLOT 1: ERP BY CLUSTER
            # ============================================================

            fig_erp, ax_erp = plt.subplots(figsize=(12, 6))

            colors = plt.cm.RdYlBu_r(np.linspace(0, 1, optimal_k))

            # Map back to full trial space
            positive_indices = np.where(positive_mask)[0]

            for cluster_id in sorted_cluster_ids:
                cluster_mask_pos = cluster_labels_positive == cluster_id

                # Get full trial indices for this cluster
                full_trial_indices = positive_indices[cluster_mask_pos]

                # Get trials from original data
                cluster_trials = X_pz[full_trial_indices, :]

                cluster_erp = cluster_trials.mean(axis=0)
                cluster_sem = cluster_trials.std(axis=0) / np.sqrt(
                    len(full_trial_indices)
                )

                # Get latency window for shading
                cluster_lats = latencies_positive[cluster_mask_pos]
                lat_mean = cluster_lats.mean()
                lat_std = cluster_lats.std()
                window = (lat_mean - lat_std, lat_mean + lat_std)

                ax_erp.plot(
                    time_ms,
                    cluster_erp,
                    linewidth=2.5,
                    label=f"{cluster_names[cluster_id]} (n={np.sum(cluster_mask_pos)})",
                    color=colors[cluster_id],
                )
                ax_erp.fill_between(
                    time_ms,
                    cluster_erp - cluster_sem,
                    cluster_erp + cluster_sem,
                    color=colors[cluster_id],
                    alpha=0.2,
                )

                # Shade latency window
                ax_erp.axvspan(
                    window[0], window[1], alpha=0.08, color=colors[cluster_id]
                )

            ax_erp.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
            ax_erp.axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)
            ax_erp.set_xlabel("Time (ms)", fontsize=13)
            ax_erp.set_ylabel("Amplitude (µV)", fontsize=13)
            ax_erp.set_title(
                f"{condition_name} ERP at {pz_name}: Latency Clusters",
                fontsize=15,
                fontweight="bold",
            )
            ax_erp.legend(fontsize=11)
            ax_erp.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_dir:
                fig_path = os.path.join(
                    save_dir, f"erp_latency_clusters_{condition_name.lower()}.png"
                )
                fig_erp.savefig(fig_path, dpi=300, bbox_inches="tight")
                print(f"✅ Saved: {fig_path}")

            plt.show()
            plt.close(fig_erp)

            # ============================================================
            # COMPUTE N200 METRICS PER CLUSTER
            # ============================================================

            print(f"\n{'='*60}")
            print("N200 AND P300 METRICS BY CLUSTER")
            print(f"{'='*60}")

            cluster_metrics = {}

            for cluster_id in sorted_cluster_ids:
                cluster_mask_pos = cluster_labels_positive == cluster_id
                full_trial_indices = positive_indices[cluster_mask_pos]
                cluster_trials = X_pz[full_trial_indices, :]
                cluster_erp = cluster_trials.mean(axis=0)

                # N200
                n200_amp = cluster_erp[n200_window].min()
                n200_lat = time_ms[n200_window][np.argmin(cluster_erp[n200_window])]

                # P300
                p300_metrics = p3_metrics(time_ms, cluster_erp)

                # Peak-to-peak
                n2p3_ptp = p300_metrics["peak_uv"] - n200_amp

                cluster_metrics[cluster_id] = {
                    "n200_amp": n200_amp,
                    "n200_lat": n200_lat,
                    "p300_amp": p300_metrics["peak_uv"],
                    "p300_lat": p300_metrics["latency_ms"],
                    "n2p3_ptp": n2p3_ptp,
                }

                print(f"\n{cluster_names[cluster_id]} (n={np.sum(cluster_mask_pos)}):")
                print(f"  N200: {n200_amp:.2f} µV at {n200_lat:.0f} ms")
                print(
                    f"  P300: {p300_metrics['peak_uv']:.2f} µV at {p300_metrics['latency_ms']:.0f} ms"
                )
                print(f"  N2-P3 p2p: {n2p3_ptp:.2f} µV")

            print(f"{'='*60}")

            # ============================================================
            # TEMPORAL EVOLUTION (if run labels provided)
            # ============================================================

            if run_labels_positive is not None:
                print("\n=== Temporal Evolution Across Runs ===")

                # Create evolution plots
                fig_evo, axes = plt.subplots(2, 2, figsize=(14, 10))

                unique_runs = np.unique(run_labels_positive)
                n_runs = len(unique_runs)

                # 1. Stacked bar
                ax1 = axes[0, 0]

                cluster_by_run = np.zeros((n_runs, optimal_k))
                for run_idx, run_num in enumerate(unique_runs):
                    run_mask = run_labels_positive == run_num
                    for cluster_id in range(optimal_k):
                        cluster_by_run[run_idx, cluster_id] = np.sum(
                            (cluster_labels_positive == cluster_id) & run_mask
                        )

                bottom = np.zeros(n_runs)
                for cluster_id in sorted_cluster_ids:
                    ax1.bar(
                        unique_runs,
                        cluster_by_run[:, cluster_id],
                        bottom=bottom,
                        label=cluster_names[cluster_id],
                        color=colors[cluster_id],
                        alpha=0.8,
                    )
                    bottom += cluster_by_run[:, cluster_id]

                ax1.set_xlabel("Run Number", fontsize=12)
                ax1.set_ylabel("Number of Trials (Positive P300s)", fontsize=12)
                ax1.set_title(
                    "Latency Cluster Composition by Run", fontsize=14, fontweight="bold"
                )
                ax1.legend()
                ax1.grid(True, alpha=0.3, axis="y")

                # 2. Mean latency per run
                ax2 = axes[0, 1]

                mean_lat_per_run = []
                std_lat_per_run = []

                for run_num in unique_runs:
                    run_mask = run_labels_positive == run_num
                    run_lats = latencies_positive[run_mask]
                    mean_lat_per_run.append(run_lats.mean())
                    std_lat_per_run.append(run_lats.std())

                mean_lat_per_run = np.array(mean_lat_per_run)
                std_lat_per_run = np.array(std_lat_per_run)

                ax2.plot(unique_runs, mean_lat_per_run, "o-", linewidth=2, markersize=8)
                ax2.fill_between(
                    unique_runs,
                    mean_lat_per_run - std_lat_per_run,
                    mean_lat_per_run + std_lat_per_run,
                    alpha=0.3,
                )

                # Trend line
                from scipy.stats import linregress

                slope, intercept, r_value, p_value, std_err = linregress(
                    unique_runs, mean_lat_per_run
                )
                trend_line = slope * unique_runs + intercept
                ax2.plot(
                    unique_runs,
                    trend_line,
                    "--",
                    color="red",
                    alpha=0.6,
                    label=f"Trend: {slope:.1f} ms/run\n(r={r_value:.2f}, p={p_value:.3f})",
                )

                ax2.set_xlabel("Run Number", fontsize=12)
                ax2.set_ylabel("Mean P300 Latency (ms)", fontsize=12)
                ax2.set_title("P300 Latency Evolution", fontsize=14, fontweight="bold")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # 3. Heatmap
                ax3 = axes[1, 0]

                im = ax3.imshow(
                    cluster_by_run.T,
                    aspect="auto",
                    cmap="YlOrRd",
                    interpolation="nearest",
                )
                ax3.set_xlabel("Run Number", fontsize=12)
                ax3.set_ylabel("Cluster", fontsize=12)
                ax3.set_title(
                    "Cluster Distribution Heatmap", fontsize=14, fontweight="bold"
                )
                ax3.set_xticks(range(n_runs))
                ax3.set_xticklabels(unique_runs.astype(int))
                ax3.set_yticks(range(optimal_k))
                ax3.set_yticklabels([cluster_names[i] for i in range(optimal_k)])
                plt.colorbar(im, ax=ax3, label="Trial Count")

                # 4. Proportion evolution
                ax4 = axes[1, 1]

                for cluster_id in sorted_cluster_ids:
                    proportions = cluster_by_run[:, cluster_id] / cluster_by_run.sum(
                        axis=1
                    )
                    ax4.plot(
                        unique_runs,
                        proportions,
                        "o-",
                        linewidth=2,
                        markersize=8,
                        label=cluster_names[cluster_id],
                        color=colors[cluster_id],
                    )

                ax4.set_xlabel("Run Number", fontsize=12)
                ax4.set_ylabel("Proportion of Trials", fontsize=12)
                ax4.set_title(
                    "Cluster Proportion Evolution", fontsize=14, fontweight="bold"
                )
                ax4.set_ylim([0, 1])
                ax4.legend()
                ax4.grid(True, alpha=0.3)

                plt.tight_layout()

                if save_dir:
                    fig_path = os.path.join(
                        save_dir, f"temporal_evolution_{condition_name.lower()}.png"
                    )
                    fig_evo.savefig(fig_path, dpi=300, bbox_inches="tight")
                    print(f"✅ Saved: {fig_path}")

                plt.show()
                plt.close(fig_evo)

                # Print summary
                print(f"\n{'='*60}")
                print("TEMPORAL EVOLUTION SUMMARY")
                print(f"{'='*60}")

                if p_value < 0.05:
                    if slope < 0:
                        print(
                            f"✓ P300 latency DECREASED: {slope:.1f} ms/run (p={p_value:.3f})"
                        )
                    else:
                        print(
                            f"⚠️ P300 latency INCREASED: {slope:.1f} ms/run (p={p_value:.3f})"
                        )
                else:
                    print(f"→ No significant latency trend (p={p_value:.3f})")

                # Early vs late
                early_runs = unique_runs[:3]
                late_runs = unique_runs[-3:]

                early_mask = np.isin(run_labels_positive, early_runs)
                late_mask = np.isin(run_labels_positive, late_runs)

                print(f"\nEarly runs ({early_runs[0]}-{early_runs[-1]}):")
                for cluster_id in sorted_cluster_ids:
                    count = np.sum((cluster_labels_positive == cluster_id) & early_mask)
                    pct = 100 * count / np.sum(early_mask)
                    print(f"  {cluster_names[cluster_id]}: {count} ({pct:.0f}%)")

                print(f"\nLate runs ({late_runs[0]}-{late_runs[-1]}):")
                for cluster_id in sorted_cluster_ids:
                    count = np.sum((cluster_labels_positive == cluster_id) & late_mask)
                    pct = 100 * count / np.sum(late_mask)
                    print(f"  {cluster_names[cluster_id]}: {count} ({pct:.0f}%)")

                print(f"{'='*60}")

            # Return results
            return {
                "condition": condition_name,
                "n_trials": n_trials,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "latencies_all": per_trial_latencies,
                "amplitudes_all": per_trial_amplitudes,
                "is_positive": per_trial_is_positive,
                "latencies_positive": latencies_positive,
                "amplitudes_positive": amplitudes_positive,
                "cluster_labels": cluster_labels_positive,
                "cluster_names": cluster_names,
                "cluster_metrics": cluster_metrics,
                "optimal_k": optimal_k,
                "run_labels": (
                    run_labels_positive if run_labels_positive is not None else None
                ),
            }

        # ============================================================
        # RUN ANALYSIS BASED ON CHOICE
        # ============================================================

        if condition_choice == 4:
            # Compare all three
            print("\n" + "=" * 70)
            print("COMPARING PRE, POST, AND ONLINE")
            print("=" * 70)

            results = {}

            # Pre
            if p300_pre is not None:
                results["pre"] = cluster_by_latency(
                    p300_pre,
                    time_ms,
                    pz_idx,
                    "Pre",
                    run_labels=None,
                    save_dir=latency_dir,
                )

            # Post
            if p300_post is not None:
                results["post"] = cluster_by_latency(
                    p300_post,
                    time_ms,
                    pz_idx,
                    "Post",
                    run_labels=None,
                    save_dir=latency_dir,
                )

            # Online
            if comparison == 1 and p300_online is not None:
                # Create run labels for online
                n_online_trials = p300_online.shape[2]
                trials_per_run = 10
                online_run_labels = np.repeat(np.arange(1, 9), trials_per_run)
                if len(online_run_labels) < n_online_trials:
                    remainder = n_online_trials - len(online_run_labels)
                    online_run_labels = np.concatenate(
                        [online_run_labels, np.full(remainder, 8)]
                    )
                online_run_labels = online_run_labels[:n_online_trials]

                results["online"] = cluster_by_latency(
                    p300_online,
                    time_ms,
                    pz_idx,
                    "Online",
                    run_labels=online_run_labels,
                    save_dir=latency_dir,
                )

            # Comparison summary
            print("\n" + "=" * 70)
            print("COMPARISON SUMMARY")
            print("=" * 70)

            for key in ["pre", "post", "online"]:
                if key in results and results[key] is not None:
                    res = results[key]
                    print(f"\n{res['condition'].upper()}:")
                    print(f"  Total trials: {res['n_trials']}")
                    print(
                        f"  Positive P300s: {res['n_positive']} ({100*res['n_positive']/res['n_trials']:.0f}%)"
                    )
                    if res["n_positive"] > 0:
                        print(
                            f"  Mean latency: {res['latencies_positive'].mean():.1f} ms"
                        )

                        print(f"  Clusters: {res['optimal_k']}")

            print("=" * 70)

            # ============================================================
            # PRE vs POST DIRECT COMPARISON
            # ============================================================

            if "pre" in results and "post" in results and "online" in results:
                # Three-way comparison (main analysis)
                if (
                    results["pre"] is not None
                    and results["post"] is not None
                    and results["online"] is not None
                ):
                    compare_all_three_conditions(
                        results_pre=results["pre"],
                        results_post=results["post"],
                        results_online=results["online"],
                        save_dir=latency_dir,
                    )

        else:
            # Single condition
            if special == "runs":
                # Online with run labels
                n_online_trials = p300_data.shape[2]
                trials_per_run = 10
                online_run_labels = np.repeat(np.arange(1, 9), trials_per_run)
                if len(online_run_labels) < n_online_trials:
                    remainder = n_online_trials - len(online_run_labels)
                    online_run_labels = np.concatenate(
                        [online_run_labels, np.full(remainder, 8)]
                    )
                online_run_labels = online_run_labels[:n_online_trials]

                result = cluster_by_latency(
                    p300_data,
                    time_ms,
                    pz_idx,
                    condition_name,
                    run_labels=online_run_labels,
                    save_dir=latency_dir,
                )
            else:
                # Pre or Post (no run labels)
                result = cluster_by_latency(
                    p300_data,
                    time_ms,
                    pz_idx,
                    condition_name,
                    run_labels=None,
                    save_dir=latency_dir,
                )

        print("\n" + "=" * 70)
        print("✅ LATENCY CLUSTERING ANALYSIS COMPLETE")
        print(f"📁 Figures saved to: {latency_dir}")
        print("=" * 70)

    elif c == 7:  # === SOURCE LOCALIZATION ===
        print("\n" + "=" * 70)
        print("SOURCE LOCALIZATION ANALYSIS")
        print("=" * 70)

        output_dir = "/home/alexandra-admin/Documents/Offline/offline_logs"

        # Check available data
        p300_data_dict = {
            "Pre": p300_pre,
            "POST": p300_post,  # all["nback_post_target_all"],
            "ONLINE": p300_online,  # all["nback_online_target_all"],
        }

        # Create output directory
        source_dir = os.path.join(output_dir, "source_localization")

        # Path to your channel locations file
        ch_pos_file = os.path.join(
            "/home/alexandra-admin/Documents/PhD/Task Code/ch32Locations.mat"
        )

        # Check if file exists
        if not os.path.exists(ch_pos_file):
            print(f"⚠️ Channel positions file not found: {ch_pos_file}")
            print("Please provide the correct path to ch32Locations.mat")
            ch_pos_file = input(
                "Enter path to ch32Locations.mat (or press Enter to use standard montage): "
            ).strip()
            if not ch_pos_file:
                ch_pos_file = None

        # Run source localization
        all_components = run_source_localization_analysis(
            p300_data_dict=p300_data_dict,
            labels=labels,
            time_ms=time_ms,
            output_dir=source_dir,
            sfreq=512,
            ch_pos_file=ch_pos_file,
        )

    elif c == 8:
        # Create output directory
        source_dir = os.path.join(output_dir, "connectivity_results")
        results = run_complete_connectivity_analysis_unfiltered(
            all_subjects_data=all,
            times=time_ms,
            labels=labels,
            save_results=True,
            output_dir=source_dir,
        )
