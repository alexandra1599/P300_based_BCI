import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import os


def cpp_latency_distribution_analysis(
    epochs, times, channel_indices, window=(200, 500)
):
    """
    Analyze CPP buildup characteristics (like P300 latency analysis but for CPP)

    Parameters:
        epochs: (samples, channels, trials) - NON-TARGET trials
        times: time vector in ms
        channel_indices: list of channels to average (e.g., [CP1, CP2])
        window: time window to search for CPP peak (ms)

    Returns:
        fig, stats_dict
    """
    from scipy import stats

    # Average across CPP channels (CP1, CP2, etc.)
    if len(channel_indices) > 1:
        data = epochs[:, channel_indices, :].mean(axis=1)  # (samples, trials)
    else:
        data = epochs[:, channel_indices[0], :]

    # Find window indices
    mask = (times >= window[0]) & (times <= window[1])
    window_times = times[mask]

    # Find peak latency for each trial
    latencies = []
    amplitudes = []

    for trial in range(data.shape[1]):
        trial_data = data[mask, trial]
        peak_idx = np.argmax(trial_data)
        latencies.append(window_times[peak_idx])
        amplitudes.append(trial_data[peak_idx])

    latencies = np.array(latencies)
    amplitudes = np.array(amplitudes)

    # Statistics
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    median_lat = np.median(latencies)

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Histogram of latencies
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(latencies, bins=20, edgecolor="black", alpha=0.7, color="skyblue")
    ax1.axvline(
        mean_lat,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_lat:.1f} ms",
    )
    ax1.axvline(
        median_lat,
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_lat:.1f} ms",
    )
    ax1.set_xlabel("CPP Peak Latency (ms)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"CPP Latency Distribution\n(SD = {std_lat:.1f} ms)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Histogram of amplitudes
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(amplitudes, bins=20, edgecolor="black", alpha=0.7, color="lightcoral")
    ax2.axvline(
        np.mean(amplitudes),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(amplitudes):.2f} µV",
    )
    ax2.set_xlabel("CPP Peak Amplitude (µV)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"CPP Amplitude Distribution\n(SD = {np.std(amplitudes):.2f} µV)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Scatter: Latency vs Amplitude
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(latencies, amplitudes, alpha=0.5, s=30)

    # Correlation
    corr, p_val = stats.pearsonr(latencies, amplitudes)
    ax3.text(
        0.05,
        0.95,
        f"r = {corr:.3f}\np = {p_val:.3g}",
        transform=ax3.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Fit line
    z = np.polyfit(latencies, amplitudes, 1)
    p = np.poly1d(z)
    ax3.plot(latencies, p(latencies), "r--", linewidth=2, alpha=0.8)

    ax3.set_xlabel("CPP Peak Latency (ms)")
    ax3.set_ylabel("CPP Peak Amplitude (µV)")
    ax3.set_title("Latency vs Amplitude Relationship")
    ax3.grid(True, alpha=0.3)

    # 4. Trial-by-trial evolution
    ax4 = fig.add_subplot(gs[1, 1])
    trial_nums = np.arange(1, len(latencies) + 1)
    ax4.plot(trial_nums, latencies, "o-", alpha=0.6, markersize=4)
    ax4.set_xlabel("Trial Number")
    ax4.set_ylabel("CPP Peak Latency (ms)")
    ax4.set_title("CPP Latency Across Trials (Drift Check)")
    ax4.grid(True, alpha=0.3)

    # Trend line
    z = np.polyfit(trial_nums, latencies, 1)
    p = np.poly1d(z)
    ax4.plot(
        trial_nums,
        p(trial_nums),
        "r--",
        linewidth=2,
        alpha=0.8,
        label=f"Slope: {z[0]:.3f} ms/trial",
    )
    ax4.legend()

    # 5. Single-trial waveforms sorted by latency
    ax5 = fig.add_subplot(gs[2, :])

    # Sort trials by latency
    sort_idx = np.argsort(latencies)
    sorted_data = data[:, sort_idx]

    # Create image
    im = ax5.imshow(
        sorted_data.T,
        aspect="auto",
        cmap="RdBu_r",
        extent=[times[0], times[-1], 0, len(latencies)],
        vmin=-np.percentile(np.abs(sorted_data), 95),
        vmax=np.percentile(np.abs(sorted_data), 95),
    )

    # Overlay peak latencies
    ax5.plot(
        latencies[sort_idx], np.arange(len(latencies)), "k.", markersize=2, alpha=0.5
    )

    ax5.axvline(0, color="white", linestyle="--", linewidth=1)
    ax5.axvspan(window[0], window[1], alpha=0.2, color="yellow")
    ax5.set_xlabel("Time (ms)")
    ax5.set_ylabel("Trial (sorted by CPP latency)")
    ax5.set_title("Single-Trial CPP Waveforms (sorted by peak latency)")

    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label("Amplitude (µV)", rotation=270, labelpad=15)

    plt.suptitle(
        f"CPP Buildup Analysis (n={len(latencies)} trials)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    return fig, {
        "latencies": latencies,
        "amplitudes": amplitudes,
        "mean_lat": mean_lat,
        "std_lat": std_lat,
    }


def cpp_buildup_rate_analysis(
    all_subjects_data,
    condition_key,
    times,
    channel_indices,
    baseline_window=(-200, 0),
    buildup_window=(0, 500),
):
    """
    Analyze CPP buildup rate using mixed effects models.
    """
    import pandas as pd
    from statsmodels.regression.mixed_linear_model import MixedLM

    data_list = []

    for subj_id, subj_data in all_subjects_data.items():
        epochs = subj_data[condition_key]

        # Average across CPP channels
        if len(channel_indices) > 1:
            data = epochs[:, channel_indices, :].mean(axis=1)
        else:
            data = epochs[:, channel_indices[0], :]

        # Baseline correction
        baseline_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
        baseline = data[baseline_mask, :].mean(axis=0)
        data_corrected = data - baseline

        # Get buildup window
        buildup_mask = (times >= buildup_window[0]) & (times <= buildup_window[1])
        buildup_times = times[buildup_mask]
        buildup_data = data_corrected[buildup_mask, :]

        # Compute slope for each trial
        for trial in range(buildup_data.shape[1]):
            coeffs = np.polyfit(buildup_times, buildup_data[:, trial], 1)
            slope = coeffs[0] * 1000  # Convert to µV/s

            data_list.append({"subject": subj_id, "trial": trial, "slope": slope})

    df = pd.DataFrame(data_list)

    # Subject-level means
    subject_slopes = df.groupby("subject")["slope"].mean()

    print(f"\n=== CPP Buildup Rate Analysis ===")
    print(f"Mean slope: {subject_slopes.mean():.2f} ± {subject_slopes.std():.2f} µV/s")

    return df, subject_slopes


def compare_cpp_conditions(cpp_pre, cpp_post, cpp_online, times, channel_indices):
    """
    Compare CPP characteristics across Pre, Post, and Online sessions

    Parameters:
        cpp_pre, cpp_post, cpp_online: (samples, channels, trials)
        times: time vector in ms
        channel_indices: CPP channels to average
    """
    from scipy.ndimage import gaussian_filter1d

    # Average across CPP channels
    def get_avg(epochs):
        if len(channel_indices) > 1:
            return epochs[:, channel_indices, :].mean(axis=1)  # (samples, trials)
        else:
            return epochs[:, channel_indices[0], :]

    data_pre = get_avg(cpp_pre)
    data_post = get_avg(cpp_post)
    data_online = get_avg(cpp_online)

    # Grand averages
    ga_pre = data_pre.mean(axis=1)
    ga_post = data_post.mean(axis=1)
    ga_online = data_online.mean(axis=1)

    # SEMs
    sem_pre = data_pre.std(axis=1) / np.sqrt(data_pre.shape[1])
    sem_post = data_post.std(axis=1) / np.sqrt(data_post.shape[1])
    sem_online = data_online.std(axis=1) / np.sqrt(data_online.shape[1])

    # Smooth
    ga_pre_smooth = gaussian_filter1d(ga_pre, sigma=2)
    ga_post_smooth = gaussian_filter1d(ga_post, sigma=2)
    ga_online_smooth = gaussian_filter1d(ga_online, sigma=2)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(times, ga_pre_smooth, "b-", linewidth=2.5, label="Pre (offline)", alpha=0.8)
    ax.fill_between(times, ga_pre - sem_pre, ga_pre + sem_pre, color="b", alpha=0.2)

    ax.plot(
        times, ga_post_smooth, "r-", linewidth=2.5, label="Post (offline)", alpha=0.8
    )
    ax.fill_between(times, ga_post - sem_post, ga_post + sem_post, color="r", alpha=0.2)

    ax.plot(
        times, ga_online_smooth, "g-", linewidth=2.5, label="Online (BCI)", alpha=0.8
    )
    ax.fill_between(
        times, ga_online - sem_online, ga_online + sem_online, color="g", alpha=0.2
    )

    # Highlight CPP window
    ax.axvspan(200, 500, alpha=0.1, color="yellow", label="CPP window")
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax.axvline(0, color="k", linestyle="--", linewidth=1, label="Stimulus")

    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Amplitude (µV)", fontsize=12)
    ax.set_title(
        "CPP Comparison: Pre vs Post vs Online", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from scipy.signal import savgol_filter
import os


def detect_cpp_onset(
    signal,
    times,
    baseline_window=(-200, 0),
    search_window=(100, 400),
    threshold_method="sustained_positive",
    smooth_window=51,
    min_sustained_duration=50,
):
    """
    Detect CPP onset - when sustained positive evidence accumulation begins

    Parameters:
        signal: 1D array (samples,) - single trial or average
        times: time vector in ms
        baseline_window: for establishing baseline
        search_window: where to look for onset (start later to avoid early noise)
        threshold_method: 'sustained_positive', 'derivative', or 'amplitude'
        smooth_window: samples for smoothing (must be odd)
        min_sustained_duration: minimum duration (ms) of sustained increase

    Returns:
        onset_time: time in ms when CPP starts
        onset_idx: sample index of onset
    """
    # Ensure smooth_window is odd
    if smooth_window % 2 == 0:
        smooth_window += 1

    # Baseline correction
    baseline_mask = (times >= baseline_window[0]) & (times <= baseline_window[1])
    baseline_mean = np.mean(signal[baseline_mask])
    signal_corrected = signal - baseline_mean

    # Smooth the signal
    signal_smooth = savgol_filter(signal_corrected, smooth_window, 3)

    # Get search window - START LATER to avoid N200
    search_mask = (times >= search_window[0]) & (times <= search_window[1])
    search_times = times[search_mask]
    search_signal = signal_smooth[search_mask]

    if len(search_signal) == 0:
        raise ValueError(f"Search window {search_window} has no samples")

    dt = times[1] - times[0]  # sampling interval

    if threshold_method == "sustained_positive":
        # NEW METHOD: Find where signal becomes positive AND stays positive

        # 1. Signal must be positive (evidence accumulation has begun)
        positive_mask = search_signal > 0

        if not np.any(positive_mask):
            # Fallback: use minimum point (N200 trough) + 50ms
            min_idx = np.argmin(search_signal)
            onset_idx_local = min(min_idx + int(50 / dt), len(search_signal) - 1)
        else:
            # 2. Find first point where signal goes positive and STAYS positive
            min_sustained_samples = int(min_sustained_duration / dt)
            onset_idx_local = None

            for i in range(len(search_signal) - min_sustained_samples):
                if np.all(search_signal[i : i + min_sustained_samples] > 0):
                    onset_idx_local = i
                    break

            if onset_idx_local is None:
                # Fallback: first positive crossing
                positive_crossings = np.where(positive_mask)[0]
                onset_idx_local = (
                    positive_crossings[0] if len(positive_crossings) > 0 else 0
                )

    elif threshold_method == "derivative":
        # Find when derivative becomes consistently positive
        derivative = np.gradient(search_signal, dt)
        derivative_smooth = savgol_filter(
            derivative, min(31, len(derivative) // 2 * 2 + 1), 2
        )

        # Must be positive for at least min_sustained_duration
        window_samples = int(min_sustained_duration / dt)
        onset_idx_local = None

        for i in range(len(derivative_smooth) - window_samples):
            if (
                np.all(derivative_smooth[i : i + window_samples] > 0)
                and search_signal[i] > 0
            ):
                onset_idx_local = i
                break

        if onset_idx_local is None:
            # Fallback: maximum derivative point (steepest ascent)
            onset_idx_local = np.argmax(derivative_smooth)

    elif threshold_method == "amplitude":
        # Find when signal crosses a threshold (e.g., 30% of peak amplitude)
        peak_amp = np.max(search_signal)
        if peak_amp <= 0:
            # No positive peak - fallback to minimum + offset
            onset_idx_local = np.argmin(search_signal)
        else:
            threshold = 0.3 * peak_amp  # 30% of peak
            crossings = np.where(search_signal > threshold)[0]
            onset_idx_local = crossings[0] if len(crossings) > 0 else 0

    else:
        raise ValueError(
            "threshold_method must be 'sustained_positive', 'derivative', or 'amplitude'"
        )

    # Convert back to original time indexing
    onset_time = search_times[onset_idx_local]
    onset_idx = np.where(times == onset_time)[0][0]

    return onset_time, onset_idx


def cpp_onset_analysis(
    all_subjects_data,
    condition_key,
    times,
    channel_indices,
    baseline_window=(-200, 0),
    search_window=(0, 400),
):
    """
    Analyze CPP onset across subjects using mixed effects models.

    Parameters:
    -----------
    all_subjects_data : dict
        Dictionary with subject IDs as keys
    condition_key : str
        e.g., 'nback_pre_nontarget_all' (CPP from non-targets)
    times : array
        Time vector in ms
    channel_indices : list
        CPP channels to average
    """
    import pandas as pd
    from statsmodels.regression.mixed_linear_model import MixedLM

    # Collect onset times for all subjects
    data_list = []

    for subj_id, subj_data in all_subjects_data.items():
        # Extract CPP data: (time, channels, trials)
        epochs = subj_data[condition_key]

        # Average across CPP channels
        if len(channel_indices) > 1:
            data = epochs[:, channel_indices, :].mean(axis=1)  # (time, trials)
        else:
            data = epochs[:, channel_indices[0], :]

        # Detect onset for each trial
        for trial in range(data.shape[1]):
            try:
                onset_t, onset_idx = detect_cpp_onset(
                    data[:, trial],
                    times,
                    baseline_window,
                    search_window,
                    threshold_method="sustained_positive",
                    min_sustained_duration=50,
                )

                data_list.append(
                    {"subject": subj_id, "trial": trial, "onset_time": onset_t}
                )
            except:
                continue

    df = pd.DataFrame(data_list)

    # Subject-level means
    subject_onsets = df.groupby("subject")["onset_time"].mean()

    print(f"\n=== CPP Onset Analysis ===")
    print(
        f"Mean onset across subjects: {subject_onsets.mean():.2f} ± {subject_onsets.std():.2f} ms"
    )
    print(f"Subjects: {len(subject_onsets)}")

    # One-sample t-test: Is onset different from a theoretical value (e.g., 200ms)?
    from scipy import stats

    t_stat, p_val = stats.ttest_1samp(subject_onsets, 200)  # Test against 200ms
    print(f"Test vs 200ms: t = {t_stat:.3f}, p = {p_val:.4f}")

    return df, subject_onsets


def compare_cpp_slopes(
    all_subjects_data, times, channel_indices, buildup_window=(0, 500)
):
    """
    Compare CPP buildup slopes across PRE, POST, ONLINE using MEM.

    This is the KEY analysis for your CPP results!
    """
    import pandas as pd
    from statsmodels.regression.mixed_linear_model import MixedLM
    import matplotlib.pyplot as plt

    # Collect slopes for all conditions
    data_list = []

    for condition in [
        "nback_pre_nontarget_all",
        "nback_post_nontarget_all",
        "online_nontarget_all",
    ]:
        # Determine condition label
        if "pre" in condition:
            cond_label = "pre"
        elif "post" in condition:
            cond_label = "post"
        else:
            cond_label = "online"

        for subj_id, subj_data in all_subjects_data.items():
            if condition not in subj_data:
                continue

            epochs = subj_data[condition]

            # Average across CPP channels
            if len(channel_indices) > 1:
                data = epochs[:, channel_indices, :].mean(axis=1)
            else:
                data = epochs[:, channel_indices[0], :]

            # Get buildup window
            buildup_mask = (times >= buildup_window[0]) & (times <= buildup_window[1])
            buildup_times = times[buildup_mask]
            buildup_data = data[buildup_mask, :]

            # Compute slope for each trial
            for trial in range(buildup_data.shape[1]):
                coeffs = np.polyfit(buildup_times, buildup_data[:, trial], 1)
                slope = coeffs[0] * 1000  # µV/s

                data_list.append(
                    {
                        "subject": subj_id,
                        "trial": trial,
                        "slope": slope,
                        "condition": cond_label,
                    }
                )

    df = pd.DataFrame(data_list)

    # Mixed Effects Model: condition as fixed effect, subject as random effect
    print("\n" + "=" * 70)
    print("MIXED EFFECTS MODEL: CPP Buildup Rate")
    print("=" * 70)

    model = MixedLM.from_formula(
        "slope ~ condition",  # Fixed effect: condition
        data=df,
        groups=df["subject"],  # Random effect: subject
    )
    result = model.fit(reml=True)
    print(result.summary())

    # Subject-level means for plotting
    subject_means = df.groupby(["subject", "condition"])["slope"].mean().unstack()

    # Overall condition means
    condition_means = df.groupby("condition")["slope"].mean()
    condition_sems = df.groupby("condition")["slope"].sem()

    print("\n" + "=" * 70)
    print("CONDITION MEANS")
    print("=" * 70)
    for cond in ["pre", "post", "online"]:
        if cond in condition_means.index:
            print(
                f"{cond.upper():8s}: {condition_means[cond]:6.2f} ± {condition_sems[cond]:5.2f} µV/s"
            )

    # Create figures
    fig = create_cpp_slope_figures(df, subject_means, result)

    return {
        "df": df,
        "mem_result": result,
        "subject_means": subject_means,
        "condition_means": condition_means,
        "fig": fig,
    }


def create_cpp_slope_figures(df, subject_means, mem_result):
    """Create figures for CPP slope comparison."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Box plot with individual subjects
    ax = axes[0, 0]

    conditions_order = ["pre", "post", "online"]
    data_for_box = [df[df["condition"] == c]["slope"].values for c in conditions_order]

    bp = ax.boxplot(
        data_for_box, labels=["PRE", "POST", "ONLINE"], patch_artist=True, widths=0.6
    )

    colors = ["lightblue", "lightcoral", "lightgreen"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    # Overlay individual subject means
    for i, cond in enumerate(conditions_order):
        if cond in subject_means.columns:
            y_vals = subject_means[cond].values
            x_vals = np.random.normal(i + 1, 0.04, size=len(y_vals))
            ax.scatter(x_vals, y_vals, alpha=0.6, s=60, edgecolor="black", linewidth=1)

    ax.set_ylabel("CPP Buildup Rate (µV/s)", fontsize=12)
    ax.set_title("CPP Slope Across Conditions\n(Subject Averages)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add significance from MEM
    # Extract p-values from model
    if "condition[T.post]" in mem_result.pvalues:
        p_pre_post = mem_result.pvalues["condition[T.post]"]
        # Add significance line
        y_max = max([d.max() for d in data_for_box])
        if p_pre_post < 0.05:
            sig_str = (
                "***" if p_pre_post < 0.001 else "**" if p_pre_post < 0.01 else "*"
            )
            ax.plot([1, 2], [y_max * 1.1, y_max * 1.1], "k-", linewidth=1.5)
            ax.text(1.5, y_max * 1.15, sig_str, ha="center", fontsize=16)

    # 2. Bar plot with MEM coefficients
    ax = axes[0, 1]

    # Extract coefficients
    coeffs = []
    labels = []
    errors = []

    # Intercept = PRE mean
    coeffs.append(mem_result.params["Intercept"])
    labels.append("PRE")
    errors.append(mem_result.bse["Intercept"])

    # POST = Intercept + condition[T.post]
    if "condition[T.post]" in mem_result.params:
        coeffs.append(
            mem_result.params["Intercept"] + mem_result.params["condition[T.post]"]
        )
        labels.append("POST")
        errors.append(
            np.sqrt(
                mem_result.bse["Intercept"] ** 2
                + mem_result.bse["condition[T.post]"] ** 2
            )
        )

    # ONLINE = Intercept + condition[T.online]
    if "condition[T.pre]" in mem_result.params:
        coeffs.append(
            mem_result.params["Intercept"] + mem_result.params["condition[T.pre]"]
        )
        labels.append("ONLINE")
        errors.append(
            np.sqrt(
                mem_result.bse["Intercept"] ** 2
                + mem_result.bse["condition[T.pre]"] ** 2
            )
        )

    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        coeffs,
        yerr=errors,
        capsize=10,
        color=colors[: len(labels)],
        edgecolor="black",
        linewidth=1.5,
        alpha=0.7,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel("CPP Buildup Rate (µV/s)", fontsize=12)
    ax.set_title("MEM Estimated Means", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Individual subject trajectories
    ax = axes[1, 0]

    for subj in subject_means.index:
        vals = []
        for cond in conditions_order:
            if cond in subject_means.columns:
                vals.append(subject_means.loc[subj, cond])
        ax.plot(range(len(vals)), vals, "o-", alpha=0.6, linewidth=2, markersize=8)

    ax.set_xticks(range(len(conditions_order)))
    ax.set_xticklabels(["PRE", "POST", "ONLINE"], fontweight="bold")
    ax.set_ylabel("CPP Buildup Rate (µV/s)", fontsize=12)
    ax.set_title("Individual Subject Trajectories", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 4. Distribution comparison
    ax = axes[1, 1]

    for cond, color in zip(conditions_order, colors):
        data = df[df["condition"] == cond]["slope"]
        ax.hist(
            data, bins=20, alpha=0.5, label=cond.upper(), color=color, edgecolor="black"
        )

    ax.set_xlabel("CPP Buildup Rate (µV/s)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Slope Distributions", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "CPP Buildup Rate Analysis (Mixed Effects Model)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig


def cpp_p300_relationship(
    all_subjects_data,
    condition_key_cpp,
    condition_key_p300,
    times,
    cpp_channels,
    p300_channel,
    cpp_window=(200, 500),
    p300_window=(300, 600),
):
    """
    Analyze CPP-P300 relationship using MEM.

    Parameters:
    -----------
    condition_key_cpp : str
        e.g., 'nback_pre_nontarget_all' (CPP from non-targets)
    condition_key_p300 : str
        e.g., 'nback_pre_target_all' (P300 from targets)
    """
    import pandas as pd
    from statsmodels.regression.mixed_linear_model import MixedLM

    data_list = []

    for subj_id, subj_data in all_subjects_data.items():
        # Get CPP data (non-targets)
        cpp_epochs = subj_data[condition_key_cpp]
        if len(cpp_channels) > 1:
            cpp_data = cpp_epochs[:, cpp_channels, :].mean(axis=1)
        else:
            cpp_data = cpp_epochs[:, cpp_channels[0], :]

        # Get P300 data (targets)
        p300_epochs = subj_data[condition_key_p300]
        p300_data = p300_epochs[:, p300_channel, :]

        # CPP metrics
        cpp_mask = (times >= cpp_window[0]) & (times <= cpp_window[1])
        cpp_window_times = times[cpp_mask]

        n_cpp_trials = cpp_data.shape[1]
        n_p300_trials = p300_data.shape[1]
        n_trials = min(n_cpp_trials, n_p300_trials)

        for trial in range(n_trials):
            # CPP slope
            coeffs = np.polyfit(cpp_window_times, cpp_data[cpp_mask, trial], 1)
            cpp_slope = coeffs[0] * 1000  # µV/s

            # CPP amplitude
            cpp_amp = np.max(cpp_data[cpp_mask, trial])

            # P300 amplitude
            p300_mask = (times >= p300_window[0]) & (times <= p300_window[1])
            p300_amp = np.max(p300_data[p300_mask, trial])

            # P300 latency
            p300_window_times = times[p300_mask]
            peak_idx = np.argmax(p300_data[p300_mask, trial])
            p300_lat = p300_window_times[peak_idx]

            data_list.append(
                {
                    "subject": subj_id,
                    "trial": trial,
                    "cpp_slope": cpp_slope,
                    "cpp_amp": cpp_amp,
                    "p300_amp": p300_amp,
                    "p300_lat": p300_lat,
                }
            )

    df = pd.DataFrame(data_list)

    print("\n" + "=" * 70)
    print("CPP-P300 RELATIONSHIP (Mixed Effects Models)")
    print("=" * 70)

    # Model 1: CPP Slope → P300 Amplitude
    print("\n=== Model 1: CPP Slope → P300 Amplitude ===")
    model1 = MixedLM.from_formula("p300_amp ~ cpp_slope", data=df, groups=df["subject"])
    result1 = model1.fit(reml=True)
    print(result1.summary())

    # Model 2: CPP Amplitude → P300 Amplitude
    print("\n=== Model 2: CPP Amplitude → P300 Amplitude ===")
    model2 = MixedLM.from_formula("p300_amp ~ cpp_amp", data=df, groups=df["subject"])
    result2 = model2.fit(reml=True)
    print(result2.summary())

    # Model 3: CPP Slope → P300 Latency
    print("\n=== Model 3: CPP Slope → P300 Latency ===")
    model3 = MixedLM.from_formula("p300_lat ~ cpp_slope", data=df, groups=df["subject"])
    result3 = model3.fit(reml=True)
    print(result3.summary())

    return {
        "df": df,
        "model_slope_to_amp": result1,
        "model_amp_to_amp": result2,
        "model_slope_to_lat": result3,
    }


def comprehensive_cpp_report(
    cpp_pre,
    cpp_post,
    cpp_online,
    p300_pre,
    p300_post,
    p300_online,
    times,
    cpp_channels,
    p300_channel,
    labels,
):
    """
    Generate a comprehensive CPP analysis report combining all analyses

    Parameters:
        cpp_pre, cpp_post, cpp_online: NON-TARGET epochs (samples, channels, trials)
        p300_pre, p300_post, p300_online: TARGET epochs (samples, channels, trials)
        times: time vector
        cpp_channels: indices for CPP channels
        p300_channel: index for P300 channel
        labels: channel names

    Returns:
        Dictionary with all results and summary statistics
    """

    print("\n" + "=" * 70)
    print("COMPREHENSIVE CPP ANALYSIS REPORT")
    print("=" * 70)

    cpp_channel_names = (
        [labels[i] for i in cpp_channels]
        if cpp_channels[0] < len(labels)
        else ["CP1", "CP2"]
    )
    p300_channel_name = labels[p300_channel] if p300_channel < len(labels) else "Pz"

    print(f"\nCPP Channels: {cpp_channel_names}")
    print(f"P300 Channel: {p300_channel_name}")

    results = {}

    # 1. Onset Analysis
    print("\n--- CPP Onset Detection ---")
    for name, data in [("Pre", cpp_pre), ("Post", cpp_post), ("Online", cpp_online)]:
        try:
            _, onset_stats = cpp_onset_analysis(data, times, cpp_channels)
            results[f"onset_{name.lower()}"] = onset_stats
            print(
                f"{name:8s}: Mean onset = {onset_stats['mean_onset']:6.1f} ± {onset_stats['std_onset']:5.1f} ms"
            )
        except Exception as e:
            print(f"{name:8s}: Failed - {e}")

    # 2. Slope Comparison
    print("\n--- CPP Buildup Rate Comparison ---")
    try:
        _, slope_stats = compare_cpp_slopes(
            cpp_pre, cpp_post, cpp_online, times, cpp_channels
        )
        results["slopes"] = slope_stats
        print(
            f"Pre:    {slope_stats['mean_pre']:6.2f} ± {np.std(slope_stats['slopes_pre']):5.2f} µV/s"
        )
        print(
            f"Post:   {slope_stats['mean_post']:6.2f} ± {np.std(slope_stats['slopes_post']):5.2f} µV/s"
        )
        print(
            f"Online: {slope_stats['mean_online']:6.2f} ± {np.std(slope_stats['slopes_online']):5.2f} µV/s"
        )
        print(f"\nStatistical Tests:")
        print(
            f"  Pre vs Post:   p = {slope_stats['p_pre_post']:.4f}, d = {slope_stats['d_pre_post']:+.2f}"
        )
        print(
            f"  Pre vs Online: p = {slope_stats['p_pre_online']:.4f}, d = {slope_stats['d_pre_online']:+.2f}"
        )
        print(
            f"  Post vs Online: p = {slope_stats['p_post_online']:.4f}, d = {slope_stats['d_post_online']:+.2f}"
        )
    except Exception as e:
        print(f"Slope analysis failed: {e}")

    # 3. CPP-P300 Relationships
    print("\n--- CPP-P300 Relationships ---")
    for name, cpp_data, p300_data in [
        ("Pre", cpp_pre, p300_pre),
        ("Post", cpp_post, p300_post),
        ("Online", cpp_online, p300_online),
    ]:
        try:
            _, rel_stats = cpp_p300_relationship(
                cpp_data, p300_data, times, cpp_channels, p300_channel
            )
            results[f"cpp_p300_{name.lower()}"] = rel_stats
            print(f"\n{name}:")
            print(
                f"  CPP Amp → P300 Amp:  r = {rel_stats['corr_amp_amp']:+.3f}, p = {rel_stats['p_amp_amp']:.4f}"
            )
            print(
                f"  CPP Slope → P300 Amp: r = {rel_stats['corr_slope_amp']:+.3f}, p = {rel_stats['p_slope_amp']:.4f}"
            )
            print(
                f"  CPP Amp → P300 Lat:   r = {rel_stats['corr_amp_lat']:+.3f}, p = {rel_stats['p_amp_lat']:.4f}"
            )
            print(
                f"  CPP Slope → P300 Lat: r = {rel_stats['corr_slope_lat']:+.3f}, p = {rel_stats['p_slope_lat']:.4f}"
            )
        except Exception as e:
            print(f"{name}: Failed - {e}")

    # 4. Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Onset changes
    if "onset_pre" in results and "onset_post" in results:
        onset_change = (
            results["onset_post"]["mean_onset"] - results["onset_pre"]["mean_onset"]
        )
        print(f"\n1. CPP Onset: {onset_change:+.1f} ms change from Pre to Post")
        if abs(onset_change) > 20:
            print(
                f"   → {'Earlier' if onset_change < 0 else 'Later'} evidence accumulation onset"
            )
        else:
            print(f"   → Stable onset timing")

    # Slope changes
    if "slopes" in results:
        if results["slopes"]["p_pre_post"] < 0.05:
            direction = (
                "Faster"
                if results["slopes"]["mean_post"] > results["slopes"]["mean_pre"]
                else "Slower"
            )
            print(
                f"\n2. CPP Buildup Rate: {direction} in Post vs Pre (p = {results['slopes']['p_pre_post']:.3f})"
            )
            print(
                f"   → Evidence accumulation speed {'increased' if direction == 'Faster' else 'decreased'}"
            )
        else:
            print(
                f"\n2. CPP Buildup Rate: No significant change (p = {results['slopes']['p_pre_post']:.3f})"
            )

    # CPP-P300 coupling
    if "cpp_p300_pre" in results:
        strong_corrs = []
        for name, data in [
            ("Pre", "cpp_p300_pre"),
            ("Post", "cpp_p300_post"),
            ("Online", "cpp_p300_online"),
        ]:
            if data in results:
                if (
                    abs(results[data]["corr_slope_amp"]) > 0.3
                    and results[data]["p_slope_amp"] < 0.05
                ):
                    strong_corrs.append(
                        f"{name} (r={results[data]['corr_slope_amp']:.2f})"
                    )

        if strong_corrs:
            print(f"\n3. CPP-P300 Coupling: Strong in {', '.join(strong_corrs)}")
            print(f"   → Faster CPP buildup predicts larger P300")
        else:
            print(f"\n3. CPP-P300 Coupling: Weak or non-significant across conditions")

    print("\n" + "=" * 70)

    return results
