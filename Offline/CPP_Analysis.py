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
    epochs, times, channel_indices, baseline_window=(-200, 0), buildup_window=(0, 500)
):
    """
    Analyze CPP buildup rate (slope of accumulation)

    Parameters:
        epochs: (samples, channels, trials)
        times: time vector in ms
        channel_indices: CPP channels to average
        baseline_window: time for baseline
        buildup_window: time window to compute slope
    """
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
    slopes = []
    for trial in range(buildup_data.shape[1]):
        # Linear fit
        coeffs = np.polyfit(buildup_times, buildup_data[:, trial], 1)
        slopes.append(coeffs[0])  # slope in µV/ms

    slopes = np.array(slopes)

    # Average CPP waveform
    avg_cpp = data_corrected.mean(axis=1)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Average CPP with buildup window highlighted
    ax = axes[0, 0]
    ax.plot(times, avg_cpp, "b-", linewidth=2, label="Average CPP")
    ax.axvspan(
        buildup_window[0],
        buildup_window[1],
        alpha=0.2,
        color="green",
        label="Buildup window",
    )
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("Average CPP Waveform")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Buildup rate distribution
    ax = axes[0, 1]
    ax.hist(slopes * 1000, bins=20, edgecolor="black", alpha=0.7)  # convert to µV/s
    ax.axvline(
        np.mean(slopes) * 1000,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(slopes)*1000:.2f} µV/s",
    )
    ax.set_xlabel("CPP Buildup Rate (µV/s)")
    ax.set_ylabel("Count")
    ax.set_title(f"Buildup Rate Distribution\n(SD = {np.std(slopes)*1000:.2f} µV/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Buildup rate across trials
    ax = axes[1, 0]
    trial_nums = np.arange(1, len(slopes) + 1)
    ax.plot(trial_nums, slopes * 1000, "o-", alpha=0.6, markersize=4)
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Buildup Rate (µV/s)")
    ax.set_title("CPP Buildup Rate Across Trials")
    ax.grid(True, alpha=0.3)

    # Trend
    z = np.polyfit(trial_nums, slopes * 1000, 1)
    p = np.poly1d(z)
    ax.plot(
        trial_nums,
        p(trial_nums),
        "r--",
        linewidth=2,
        alpha=0.8,
        label=f"Slope: {z[0]:.3f} (µV/s)/trial",
    )
    ax.legend()

    # 4. Example single trials (fast vs slow buildup)
    ax = axes[1, 1]

    # Find fastest and slowest buildup trials
    fast_trials = np.argsort(slopes)[-5:]  # 5 fastest
    slow_trials = np.argsort(slopes)[:5]  # 5 slowest

    for trial in fast_trials:
        ax.plot(times, data_corrected[:, trial], "b-", alpha=0.3, linewidth=1)
    for trial in slow_trials:
        ax.plot(times, data_corrected[:, trial], "r-", alpha=0.3, linewidth=1)

    # Plot averages
    ax.plot(
        times,
        data_corrected[:, fast_trials].mean(axis=1),
        "b-",
        linewidth=2.5,
        label=f"Fast buildup (top 5)",
    )
    ax.plot(
        times,
        data_corrected[:, slow_trials].mean(axis=1),
        "r-",
        linewidth=2.5,
        label=f"Slow buildup (bottom 5)",
    )

    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.axvspan(buildup_window[0], buildup_window[1], alpha=0.1, color="green")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("Fast vs Slow CPP Buildup")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, {
        "slopes": slopes,
        "mean_slope": np.mean(slopes) * 1000,  # µV/s
        "std_slope": np.std(slopes) * 1000,
    }


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
    search_window=(0, 400),
    threshold_method="derivative",
    smooth_window=51,
):
    """
    Detect CPP onset - when evidence accumulation begins

    Parameters:
        signal: 1D array (samples,) - single trial or average
        times: time vector in ms
        baseline_window: for establishing baseline
        search_window: where to look for onset
        threshold_method: 'derivative' or 'amplitude'
        smooth_window: samples for smoothing (must be odd)

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

    # Get search window
    search_mask = (times >= search_window[0]) & (times <= search_window[1])
    search_times = times[search_mask]
    search_signal = signal_smooth[search_mask]

    if threshold_method == "derivative":
        # Find when derivative becomes consistently positive
        # (evidence starts accumulating)
        dt = times[1] - times[0]  # sampling interval
        derivative = np.gradient(search_signal, dt)
        derivative_smooth = savgol_filter(
            derivative, min(31, len(derivative) // 2 * 2 + 1), 2
        )

        # Find first point where derivative stays positive for at least 50ms
        window_samples = int(50 / dt)  # 50ms window
        onset_idx_local = None

        for i in range(len(derivative_smooth) - window_samples):
            if np.all(derivative_smooth[i : i + window_samples] > 0):
                onset_idx_local = i
                break

        if onset_idx_local is None:
            # Fallback: use maximum derivative point
            onset_idx_local = np.argmax(derivative_smooth)

    elif threshold_method == "amplitude":
        # Find when signal crosses 50% of peak amplitude
        peak_amp = np.max(search_signal)
        threshold = 0.5 * peak_amp

        # Find first crossing
        crossings = np.where(search_signal > threshold)[0]
        onset_idx_local = crossings[0] if len(crossings) > 0 else 0

    else:
        raise ValueError("threshold_method must be 'derivative' or 'amplitude'")

    # Convert back to original time indexing
    onset_time = search_times[onset_idx_local]
    onset_idx = np.where(times == onset_time)[0][0]

    return onset_time, onset_idx


def cpp_onset_analysis(
    epochs, times, channel_indices, baseline_window=(-200, 0), search_window=(0, 400)
):
    """
    Analyze CPP onset across trials

    Parameters:
        epochs: (samples, channels, trials)
        times: time vector in ms
        channel_indices: CPP channels to average

    Returns:
        fig, stats_dict with onset times per trial
    """
    # Average across CPP channels
    if len(channel_indices) > 1:
        data = epochs[:, channel_indices, :].mean(axis=1)  # (samples, trials)
    else:
        data = epochs[:, channel_indices[0], :]

    # Detect onset for each trial
    onset_times = []
    onset_indices = []

    for trial in range(data.shape[1]):
        try:
            onset_t, onset_idx = detect_cpp_onset(
                data[:, trial],
                times,
                baseline_window=baseline_window,
                search_window=search_window,
                threshold_method="derivative",
            )
            onset_times.append(onset_t)
            onset_indices.append(onset_idx)
        except Exception as e:
            print(f"Warning: Could not detect onset for trial {trial}: {e}")
            onset_times.append(np.nan)
            onset_indices.append(np.nan)

    onset_times = np.array(onset_times)
    onset_times = onset_times[~np.isnan(onset_times)]  # Remove failed detections

    # Statistics
    mean_onset = np.mean(onset_times)
    std_onset = np.std(onset_times)
    median_onset = np.median(onset_times)

    # Average CPP for visualization
    avg_cpp = data.mean(axis=1)
    avg_onset_t, avg_onset_idx = detect_cpp_onset(
        avg_cpp, times, baseline_window, search_window, "derivative"
    )

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Average CPP with detected onset
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, avg_cpp, "b-", linewidth=2.5, label="Average CPP")
    ax1.axvline(
        avg_onset_t,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Onset: {avg_onset_t:.1f} ms",
    )
    ax1.axvspan(
        search_window[0],
        search_window[1],
        alpha=0.1,
        color="yellow",
        label="Search window",
    )
    ax1.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax1.axvline(0, color="k", linestyle="--", linewidth=1)
    ax1.scatter(
        [avg_onset_t], [avg_cpp[avg_onset_idx]], color="r", s=100, zorder=5, marker="o"
    )
    ax1.set_xlabel("Time (ms)", fontsize=11)
    ax1.set_ylabel("Amplitude (µV)", fontsize=11)
    ax1.set_title("CPP with Detected Onset", fontsize=13, fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # 2. Onset time distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(onset_times, bins=20, edgecolor="black", alpha=0.7, color="skyblue")
    ax2.axvline(
        mean_onset,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_onset:.1f} ms",
    )
    ax2.axvline(
        median_onset,
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_onset:.1f} ms",
    )
    ax2.set_xlabel("CPP Onset Time (ms)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Onset Distribution\n(SD = {std_onset:.1f} ms)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Onset across trials (drift check)
    ax3 = fig.add_subplot(gs[1, 1])
    trial_nums = np.arange(1, len(onset_times) + 1)
    ax3.plot(trial_nums, onset_times, "o-", alpha=0.6, markersize=4)
    ax3.set_xlabel("Trial Number")
    ax3.set_ylabel("CPP Onset Time (ms)")
    ax3.set_title("Onset Timing Across Trials")
    ax3.grid(True, alpha=0.3)

    # Trend line
    z = np.polyfit(trial_nums, onset_times, 1)
    p = np.poly1d(z)
    ax3.plot(
        trial_nums,
        p(trial_nums),
        "r--",
        linewidth=2,
        alpha=0.8,
        label=f"Slope: {z[0]:.3f} ms/trial",
    )
    ax3.legend()

    # 4. Single-trial CPPs with onset markers
    ax4 = fig.add_subplot(gs[2, :])

    # Sort by onset time
    valid_trials = ~np.isnan(onset_times)
    sort_idx = np.argsort(onset_times)
    sorted_data = data[:, sort_idx]
    sorted_onsets = onset_times[sort_idx]

    # Plot heatmap
    im = ax4.imshow(
        sorted_data.T,
        aspect="auto",
        cmap="RdBu_r",
        extent=[times[0], times[-1], 0, len(sorted_onsets)],
        vmin=-np.percentile(np.abs(sorted_data), 95),
        vmax=np.percentile(np.abs(sorted_data), 95),
    )

    # Overlay onset markers
    ax4.plot(
        sorted_onsets,
        np.arange(len(sorted_onsets)),
        "k.",
        markersize=3,
        alpha=0.7,
        label="Detected onset",
    )

    ax4.axvline(0, color="white", linestyle="--", linewidth=1)
    ax4.axvspan(search_window[0], search_window[1], alpha=0.15, color="yellow")
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("Trial (sorted by onset time)")
    ax4.set_title("Single-Trial CPPs (sorted by onset)")
    ax4.legend(loc="upper right")

    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label("Amplitude (µV)", rotation=270, labelpad=15)

    plt.suptitle(
        f"CPP Onset Analysis (n={len(onset_times)} trials)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    return fig, {
        "onset_times": onset_times,
        "mean_onset": mean_onset,
        "std_onset": std_onset,
        "median_onset": median_onset,
    }


def compare_cpp_slopes(
    cpp_pre, cpp_post, cpp_online, times, channel_indices, buildup_window=(0, 500)
):
    """
    Compare CPP buildup slopes across three conditions

    Parameters:
        cpp_pre, cpp_post, cpp_online: (samples, channels, trials)
        times: time vector in ms
        channel_indices: CPP channels to average
        buildup_window: time window for slope calculation

    Returns:
        fig, comparison statistics
    """

    def compute_slopes(data):
        """Compute slopes for all trials in dataset"""
        if len(channel_indices) > 1:
            data_avg = data[:, channel_indices, :].mean(axis=1)
        else:
            data_avg = data[:, channel_indices[0], :]

        buildup_mask = (times >= buildup_window[0]) & (times <= buildup_window[1])
        buildup_times = times[buildup_mask]
        buildup_data = data_avg[buildup_mask, :]

        slopes = []
        for trial in range(buildup_data.shape[1]):
            coeffs = np.polyfit(buildup_times, buildup_data[:, trial], 1)
            slopes.append(coeffs[0] * 1000)  # Convert to µV/s

        return np.array(slopes), data_avg.mean(axis=1)

    # Compute slopes for each condition
    slopes_pre, avg_pre = compute_slopes(cpp_pre)
    slopes_post, avg_post = compute_slopes(cpp_post)
    slopes_online, avg_online = compute_slopes(cpp_online)

    # Statistics
    mean_pre, std_pre = np.mean(slopes_pre), np.std(slopes_pre)
    mean_post, std_post = np.mean(slopes_post), np.std(slopes_post)
    mean_online, std_online = np.mean(slopes_online), np.std(slopes_online)

    # Statistical tests
    t_pre_post, p_pre_post = stats.ttest_ind(slopes_pre, slopes_post)
    t_pre_online, p_pre_online = stats.ttest_ind(slopes_pre, slopes_online)
    t_post_online, p_post_online = stats.ttest_ind(slopes_post, slopes_online)

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Average CPP waveforms with linear fits
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, avg_pre, "b-", linewidth=2.5, label="Pre", alpha=0.7)
    ax1.plot(times, avg_post, "r-", linewidth=2.5, label="Post", alpha=0.7)
    ax1.plot(times, avg_online, "g-", linewidth=2.5, label="Online", alpha=0.7)

    # Add linear fit lines in buildup window
    buildup_mask = (times >= buildup_window[0]) & (times <= buildup_window[1])
    buildup_times = times[buildup_mask]

    for avg, color, label in [
        (avg_pre, "b", "Pre fit"),
        (avg_post, "r", "Post fit"),
        (avg_online, "g", "Online fit"),
    ]:
        coeffs = np.polyfit(buildup_times, avg[buildup_mask], 1)
        fit_line = np.poly1d(coeffs)
        ax1.plot(
            buildup_times,
            fit_line(buildup_times),
            color=color,
            linestyle="--",
            linewidth=2,
            alpha=0.5,
        )

    ax1.axvspan(
        buildup_window[0],
        buildup_window[1],
        alpha=0.1,
        color="yellow",
        label="Buildup window",
    )
    ax1.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax1.axvline(0, color="k", linestyle="--", linewidth=1)
    ax1.set_xlabel("Time (ms)", fontsize=11)
    ax1.set_ylabel("Amplitude (µV)", fontsize=11)
    ax1.set_title("CPP Waveforms with Linear Fits", fontsize=13, fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # 2. Slope distributions (overlapping histograms)
    ax2 = fig.add_subplot(gs[1, 0])
    bins = np.linspace(
        min(slopes_pre.min(), slopes_post.min(), slopes_online.min()),
        max(slopes_pre.max(), slopes_post.max(), slopes_online.max()),
        25,
    )
    ax2.hist(
        slopes_pre, bins=bins, alpha=0.5, label="Pre", color="blue", edgecolor="black"
    )
    ax2.hist(
        slopes_post, bins=bins, alpha=0.5, label="Post", color="red", edgecolor="black"
    )
    ax2.hist(
        slopes_online,
        bins=bins,
        alpha=0.5,
        label="Online",
        color="green",
        edgecolor="black",
    )

    ax2.axvline(mean_pre, color="b", linestyle="--", linewidth=2)
    ax2.axvline(mean_post, color="r", linestyle="--", linewidth=2)
    ax2.axvline(mean_online, color="g", linestyle="--", linewidth=2)

    ax2.set_xlabel("CPP Buildup Rate (µV/s)")
    ax2.set_ylabel("Count")
    ax2.set_title("Slope Distribution Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Box plot comparison
    ax3 = fig.add_subplot(gs[1, 1])
    bp = ax3.boxplot(
        [slopes_pre, slopes_post, slopes_online],
        labels=["Pre", "Post", "Online"],
        patch_artist=True,
        widths=0.6,
    )

    # Color the boxes
    colors = ["lightblue", "lightcoral", "lightgreen"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    # Add means
    means = [mean_pre, mean_post, mean_online]
    ax3.plot([1, 2, 3], means, "ro-", linewidth=2, markersize=8, label="Mean")

    ax3.set_ylabel("CPP Buildup Rate (µV/s)", fontsize=11)
    ax3.set_title("Slope Comparison (Box Plot)", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.legend()

    # 4. Statistical comparison table
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis("off")

    table_data = [
        ["Condition", "Mean (µV/s)", "SD (µV/s)", "N trials"],
        ["Pre", f"{mean_pre:.2f}", f"{std_pre:.2f}", f"{len(slopes_pre)}"],
        ["Post", f"{mean_post:.2f}", f"{std_post:.2f}", f"{len(slopes_post)}"],
        ["Online", f"{mean_online:.2f}", f"{std_online:.2f}", f"{len(slopes_online)}"],
        ["", "", "", ""],
        ["Comparison", "t-statistic", "p-value", "Significant?"],
        [
            "Pre vs Post",
            f"{t_pre_post:.3f}",
            f"{p_pre_post:.4f}",
            "✓" if p_pre_post < 0.05 else "✗",
        ],
        [
            "Pre vs Online",
            f"{t_pre_online:.3f}",
            f"{p_pre_online:.4f}",
            "✓" if p_pre_online < 0.05 else "✗",
        ],
        [
            "Post vs Online",
            f"{t_post_online:.3f}",
            f"{p_post_online:.4f}",
            "✓" if p_post_online < 0.05 else "✗",
        ],
    ]

    table = ax4.table(
        cellText=table_data, cellLoc="center", loc="center", bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header rows
    for i in [0, 5]:
        for j in range(4):
            table[(i, j)].set_facecolor("#4CAF50")
            table[(i, j)].set_text_props(weight="bold", color="white")

    ax4.set_title("Statistical Summary", fontsize=12, fontweight="bold", pad=20)

    # 5. Effect sizes (Cohen's d)
    ax5 = fig.add_subplot(gs[2, 1])

    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        return (np.mean(x) - np.mean(y)) / np.sqrt(
            ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2)
            / dof
        )

    d_pre_post = cohens_d(slopes_pre, slopes_post)
    d_pre_online = cohens_d(slopes_pre, slopes_online)
    d_post_online = cohens_d(slopes_post, slopes_online)

    comparisons = ["Pre vs\nPost", "Pre vs\nOnline", "Post vs\nOnline"]
    effect_sizes = [d_pre_post, d_pre_online, d_post_online]
    colors_es = [
        "red" if abs(d) > 0.5 else "orange" if abs(d) > 0.2 else "green"
        for d in effect_sizes
    ]

    bars = ax5.bar(
        comparisons, effect_sizes, color=colors_es, alpha=0.7, edgecolor="black"
    )
    ax5.axhline(0, color="k", linestyle="-", linewidth=1)
    ax5.axhline(0.2, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax5.axhline(-0.2, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax5.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax5.axhline(-0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Add value labels
    for bar, val in zip(bars, effect_sizes):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.2f}",
            ha="center",
            va="bottom" if val > 0 else "top",
            fontweight="bold",
        )

    ax5.set_ylabel("Cohen's d", fontsize=11)
    ax5.set_title(
        "Effect Sizes\n(Small: 0.2, Medium: 0.5, Large: 0.8)",
        fontsize=11,
        fontweight="bold",
    )
    ax5.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "CPP Buildup Rate Comparison: Pre vs Post vs Online",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    return fig, {
        "slopes_pre": slopes_pre,
        "slopes_post": slopes_post,
        "slopes_online": slopes_online,
        "mean_pre": mean_pre,
        "mean_post": mean_post,
        "mean_online": mean_online,
        "p_pre_post": p_pre_post,
        "p_pre_online": p_pre_online,
        "p_post_online": p_post_online,
        "d_pre_post": d_pre_post,
        "d_pre_online": d_pre_online,
        "d_post_online": d_post_online,
    }


def cpp_p300_relationship(
    cpp_epochs,
    p300_epochs,
    times,
    cpp_channels,
    p300_channel,
    cpp_window=(200, 500),
    p300_window=(300, 600),
):
    """
    Analyze relationship between CPP and P300

    Parameters:
        cpp_epochs: (samples, channels, trials) - NON-TARGET trials
        p300_epochs: (samples, channels, trials) - TARGET trials
        times: time vector in ms
        cpp_channels: indices for CPP channels
        p300_channel: index for P300 channel (e.g., Pz)
        cpp_window: time window for CPP measurement
        p300_window: time window for P300 measurement

    Returns:
        fig, correlation statistics
    """

    # Extract CPP features (from non-target trials)
    if len(cpp_channels) > 1:
        cpp_data = cpp_epochs[:, cpp_channels, :].mean(axis=1)
    else:
        cpp_data = cpp_epochs[:, cpp_channels[0], :]

    # Extract P300 features (from target trials)
    p300_data = p300_epochs[:, p300_channel, :]

    # Compute CPP metrics per trial
    cpp_mask = (times >= cpp_window[0]) & (times <= cpp_window[1])
    cpp_window_times = times[cpp_mask]

    cpp_amplitudes = []
    cpp_slopes = []

    for trial in range(cpp_data.shape[1]):
        # Peak amplitude
        cpp_amplitudes.append(np.max(cpp_data[cpp_mask, trial]))

        # Slope
        coeffs = np.polyfit(cpp_window_times, cpp_data[cpp_mask, trial], 1)
        cpp_slopes.append(coeffs[0] * 1000)  # µV/s

    cpp_amplitudes = np.array(cpp_amplitudes)
    cpp_slopes = np.array(cpp_slopes)

    # Compute P300 metrics per trial
    p300_mask = (times >= p300_window[0]) & (times <= p300_window[1])
    p300_window_times = times[p300_mask]

    p300_amplitudes = []
    p300_latencies = []

    for trial in range(p300_data.shape[1]):
        # Peak amplitude
        trial_p300 = p300_data[p300_mask, trial]
        p300_amplitudes.append(np.max(trial_p300))

        # Peak latency
        peak_idx = np.argmax(trial_p300)
        p300_latencies.append(p300_window_times[peak_idx])

    p300_amplitudes = np.array(p300_amplitudes)
    p300_latencies = np.array(p300_latencies)

    # Match trial counts (use minimum)
    n_trials = min(len(cpp_amplitudes), len(p300_amplitudes))
    cpp_amplitudes = cpp_amplitudes[:n_trials]
    cpp_slopes = cpp_slopes[:n_trials]
    p300_amplitudes = p300_amplitudes[:n_trials]
    p300_latencies = p300_latencies[:n_trials]

    # Correlations
    corr_amp_amp, p_amp_amp = stats.pearsonr(cpp_amplitudes, p300_amplitudes)
    corr_slope_amp, p_slope_amp = stats.pearsonr(cpp_slopes, p300_amplitudes)
    corr_amp_lat, p_amp_lat = stats.pearsonr(cpp_amplitudes, p300_latencies)
    corr_slope_lat, p_slope_lat = stats.pearsonr(cpp_slopes, p300_latencies)

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. CPP Amplitude vs P300 Amplitude
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(cpp_amplitudes, p300_amplitudes, alpha=0.6, s=40)

    # Fit line
    z = np.polyfit(cpp_amplitudes, p300_amplitudes, 1)
    p_fit = np.poly1d(z)
    x_fit = np.linspace(cpp_amplitudes.min(), cpp_amplitudes.max(), 100)
    ax1.plot(x_fit, p_fit(x_fit), "r--", linewidth=2, alpha=0.8)

    ax1.text(
        0.05,
        0.95,
        f"r = {corr_amp_amp:.3f}\np = {p_amp_amp:.3g}",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax1.set_xlabel("CPP Amplitude (µV)", fontsize=11)
    ax1.set_ylabel("P300 Amplitude (µV)", fontsize=11)
    ax1.set_title("CPP Amplitude vs P300 Amplitude", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # 2. CPP Slope vs P300 Amplitude
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(cpp_slopes, p300_amplitudes, alpha=0.6, s=40, color="green")

    z = np.polyfit(cpp_slopes, p300_amplitudes, 1)
    p_fit = np.poly1d(z)
    x_fit = np.linspace(cpp_slopes.min(), cpp_slopes.max(), 100)
    ax2.plot(x_fit, p_fit(x_fit), "r--", linewidth=2, alpha=0.8)

    ax2.text(
        0.05,
        0.95,
        f"r = {corr_slope_amp:.3f}\np = {p_slope_amp:.3g}",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax2.set_xlabel("CPP Buildup Rate (µV/s)", fontsize=11)
    ax2.set_ylabel("P300 Amplitude (µV)", fontsize=11)
    ax2.set_title("CPP Slope vs P300 Amplitude", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # 3. CPP Amplitude vs P300 Latency
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(cpp_amplitudes, p300_latencies, alpha=0.6, s=40, color="purple")

    z = np.polyfit(cpp_amplitudes, p300_latencies, 1)
    p_fit = np.poly1d(z)
    x_fit = np.linspace(cpp_amplitudes.min(), cpp_amplitudes.max(), 100)
    ax3.plot(x_fit, p_fit(x_fit), "r--", linewidth=2, alpha=0.8)

    ax3.text(
        0.05,
        0.95,
        f"r = {corr_amp_lat:.3f}\np = {p_amp_lat:.3g}",
        transform=ax3.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax3.set_xlabel("CPP Amplitude (µV)", fontsize=11)
    ax3.set_ylabel("P300 Latency (ms)", fontsize=11)
    ax3.set_title("CPP Amplitude vs P300 Latency", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # 4. CPP Slope vs P300 Latency
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(cpp_slopes, p300_latencies, alpha=0.6, s=40, color="orange")

    z = np.polyfit(cpp_slopes, p300_latencies, 1)
    p_fit = np.poly1d(z)
    x_fit = np.linspace(cpp_slopes.min(), cpp_slopes.max(), 100)
    ax4.plot(x_fit, p_fit(x_fit), "r--", linewidth=2, alpha=0.8)

    ax4.text(
        0.05,
        0.95,
        f"r = {corr_slope_lat:.3f}\np = {p_slope_lat:.3g}",
        transform=ax4.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax4.set_xlabel("CPP Buildup Rate (µV/s)", fontsize=11)
    ax4.set_ylabel("P300 Latency (ms)", fontsize=11)
    ax4.set_title("CPP Slope vs P300 Latency", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    # 5. Average waveforms comparison
    ax5 = fig.add_subplot(gs[2, 0])

    cpp_avg = cpp_data.mean(axis=1)
    p300_avg = p300_data.mean(axis=1)

    # Normalize for comparison
    cpp_norm = (cpp_avg - cpp_avg.min()) / (cpp_avg.max() - cpp_avg.min())
    p300_norm = (p300_avg - p300_avg.min()) / (p300_avg.max() - p300_avg.min())

    ax5.plot(times, cpp_norm, "b-", linewidth=2.5, label="CPP (normalized)", alpha=0.7)
    ax5.plot(
        times, p300_norm, "r-", linewidth=2.5, label="P300 (normalized)", alpha=0.7
    )

    ax5.axvspan(
        cpp_window[0], cpp_window[1], alpha=0.1, color="blue", label="CPP window"
    )
    ax5.axvspan(
        p300_window[0], p300_window[1], alpha=0.1, color="red", label="P300 window"
    )
    ax5.axvline(0, color="k", linestyle="--", linewidth=1)
    ax5.set_xlabel("Time (ms)", fontsize=11)
    ax5.set_ylabel("Normalized Amplitude", fontsize=11)
    ax5.set_title("CPP vs P300 Time Courses", fontsize=12, fontweight="bold")
    ax5.legend(loc="best")
    ax5.grid(True, alpha=0.3)

    # 6. Correlation summary table
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")

    table_data = [
        ["Relationship", "r", "p-value", "Interpretation"],
        [
            "CPP Amp → P300 Amp",
            f"{corr_amp_amp:.3f}",
            f"{p_amp_amp:.4f}",
            (
                "Strong +"
                if corr_amp_amp > 0.5 and p_amp_amp < 0.05
                else (
                    "Moderate +"
                    if corr_amp_amp > 0.3 and p_amp_amp < 0.05
                    else "Weak/None"
                )
            ),
        ],
        [
            "CPP Slope → P300 Amp",
            f"{corr_slope_amp:.3f}",
            f"{p_slope_amp:.4f}",
            (
                "Strong +"
                if corr_slope_amp > 0.5 and p_slope_amp < 0.05
                else (
                    "Moderate +"
                    if corr_slope_amp > 0.3 and p_slope_amp < 0.05
                    else "Weak/None"
                )
            ),
        ],
        [
            "CPP Amp → P300 Lat",
            f"{corr_amp_lat:.3f}",
            f"{p_amp_lat:.4f}",
            (
                "Strong -"
                if corr_amp_lat < -0.5 and p_amp_lat < 0.05
                else (
                    "Moderate -"
                    if corr_amp_lat < -0.3 and p_amp_lat < 0.05
                    else "Weak/None"
                )
            ),
        ],
        [
            "CPP Slope → P300 Lat",
            f"{corr_slope_lat:.3f}",
            f"{p_slope_lat:.4f}",
            (
                "Strong -"
                if corr_slope_lat < -0.5 and p_slope_lat < 0.05
                else (
                    "Moderate -"
                    if corr_slope_lat < -0.3 and p_slope_lat < 0.05
                    else "Weak/None"
                )
            ),
        ],
    ]

    table = ax6.table(
        cellText=table_data, cellLoc="center", loc="center", bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor("#2196F3")
        table[(0, j)].set_text_props(weight="bold", color="white")

    # Color-code significant correlations
    for i in range(1, 5):
        p_val = float(table_data[i][2])
        if p_val < 0.05:
            for j in range(4):
                table[(i, j)].set_facecolor("#C8E6C9")  # Light green for significant

    ax6.set_title("Correlation Summary", fontsize=12, fontweight="bold", pad=20)

    plt.suptitle(
        "CPP-P300 Relationship Analysis", fontsize=14, fontweight="bold", y=0.995
    )

    return fig, {
        "cpp_amplitudes": cpp_amplitudes,
        "cpp_slopes": cpp_slopes,
        "p300_amplitudes": p300_amplitudes,
        "p300_latencies": p300_latencies,
        "corr_amp_amp": corr_amp_amp,
        "p_amp_amp": p_amp_amp,
        "corr_slope_amp": corr_slope_amp,
        "p_slope_amp": p_slope_amp,
        "corr_amp_lat": corr_amp_lat,
        "p_amp_lat": p_amp_lat,
        "corr_slope_lat": corr_slope_lat,
        "p_slope_lat": p_slope_lat,
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
