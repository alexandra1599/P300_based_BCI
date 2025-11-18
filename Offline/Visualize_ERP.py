import numpy as np
import matplotlib.pyplot as plt


def time_frequency_analysis(epochs, fs, channel_idx, freqs=np.arange(1, 30, 1)):
    """
    Compute time-frequency representation using Short-Time Fourier Transform

    Parameters:
        epochs: (samples, channels, trials)
        fs: sampling rate
        channel_idx: channel to analyze (e.g., Pz)
        freqs: frequency range to analyze
    """
    from scipy.signal import stft, get_window

    print(f"DEBUG: time_frequency_analysis called with epochs shape: {epochs.shape}")
    print(f"DEBUG: channel_idx = {channel_idx}, fs = {fs}")

    # Average across trials for one channel
    erp = epochs[:, channel_idx, :].mean(axis=1)
    print(f"DEBUG: ERP shape after averaging: {erp.shape}")

    # Compute STFT (Short-Time Fourier Transform)
    nperseg = int(0.2 * fs)  # 200ms window
    noverlap = int(0.19 * fs)  # 95% overlap for smooth visualization

    print(f"DEBUG: Computing STFT with nperseg={nperseg}...")
    f, t, Zxx = stft(
        erp,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        boundary=None,
        padded=False,
    )

    # Compute power
    power = np.abs(Zxx) ** 2
    print(f"DEBUG: Power shape: {power.shape}")

    # Convert STFT time to match ERP time (in ms, relative to stimulus)
    # Assuming epochs start before stimulus
    times_ms = np.arange(len(erp)) * 1000 / fs
    stft_times_ms = t * 1000  # STFT time in ms

    # Frequency mask to plot only desired range
    freq_mask = (f >= freqs[0]) & (f <= freqs[-1])
    f_plot = f[freq_mask]
    power_plot = power[freq_mask, :]

    # Plot
    print(f"DEBUG: Creating figure...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Time-frequency plot
    im = ax1.pcolormesh(stft_times_ms, f_plot, power_plot, shading="auto", cmap="jet")
    ax1.set_ylabel("Frequency (Hz)", fontsize=11)
    ax1.set_title("Time-Frequency Power (STFT)", fontsize=12, fontweight="bold")
    ax1.axvline(0, color="white", linestyle="--", linewidth=2, label="Stimulus")
    ax1.set_xlim([times_ms[0], times_ms[-1]])
    cbar = plt.colorbar(im, ax=ax1, label="Power (µV²)")
    ax1.legend(loc="upper right")

    # ERP overlay
    ax2.plot(times_ms, erp, "k", linewidth=2)
    ax2.set_xlabel("Time (ms)", fontsize=11)
    ax2.set_ylabel("Amplitude (µV)", fontsize=11)
    ax2.set_title("ERP Waveform", fontsize=12, fontweight="bold")
    ax2.axvline(0, color="r", linestyle="--", linewidth=1.5, label="Stimulus")
    ax2.axvspan(250, 600, alpha=0.1, color="green", label="P300 window")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    print(f"DEBUG: Figure created successfully")
    plt.show()

    return fig, power_plot, f_plot


def cluster_based_comparison(epochs1, epochs2, times, channel_idx, n_permutations=1000):
    """
    Cluster-based permutation test for ERP differences

    Parameters:
        epochs1, epochs2: (samples, channels, trials) for two conditions
        times: time vector in ms
        channel_idx: channel to test
    """
    from scipy import stats

    # Extract data for one channel: (samples, trials)
    data1 = epochs1[:, channel_idx, :]
    data2 = epochs2[:, channel_idx, :]

    n_samples = data1.shape[0]
    n_trials1 = data1.shape[1]
    n_trials2 = data2.shape[1]

    # Compute observed t-statistic at each time point
    observed_t = np.zeros(n_samples)
    observed_p = np.zeros(n_samples)

    for t in range(n_samples):
        t_stat, p_val = stats.ttest_ind(data1[t, :], data2[t, :])
        observed_t[t] = t_stat
        observed_p[t] = p_val

    # Identify clusters in observed data (p < 0.05)
    threshold = 2.0  # approximate t-threshold for p=0.05
    clusters_obs = find_clusters(observed_t, threshold)

    # Permutation test
    max_cluster_stats = []
    all_data = np.concatenate([data1, data2], axis=1)

    for perm in range(n_permutations):
        # Shuffle trial labels
        perm_idx = np.random.permutation(n_trials1 + n_trials2)
        perm_data1 = all_data[:, perm_idx[:n_trials1]]
        perm_data2 = all_data[:, perm_idx[n_trials1:]]

        # Compute permuted t-statistics
        perm_t = np.zeros(n_samples)
        for t in range(n_samples):
            perm_t[t], _ = stats.ttest_ind(perm_data1[t, :], perm_data2[t, :])

        # Find largest cluster in permuted data
        clusters_perm = find_clusters(perm_t, threshold)
        if clusters_perm:
            max_cluster_stats.append(max([c["stat"] for c in clusters_perm]))
        else:
            max_cluster_stats.append(0)

    # Compute cluster p-values
    max_cluster_stats = np.array(max_cluster_stats)
    for cluster in clusters_obs:
        cluster["p_value"] = np.mean(max_cluster_stats >= cluster["stat"])

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # ERPs
    ax = axes[0]
    erp1 = data1.mean(axis=1)
    erp2 = data2.mean(axis=1)
    sem1 = data1.std(axis=1) / np.sqrt(n_trials1)
    sem2 = data2.std(axis=1) / np.sqrt(n_trials2)

    ax.plot(times, erp1, "b-", label="Pre", linewidth=2)
    ax.fill_between(times, erp1 - sem1, erp1 + sem1, color="b", alpha=0.2)
    ax.plot(times, erp2, "r-", label="Post", linewidth=2)
    ax.fill_between(times, erp2 - sem2, erp2 + sem2, color="r", alpha=0.2)
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("ERP Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # T-statistics
    ax = axes[1]
    ax.plot(times, observed_t, "k-", linewidth=1.5)
    ax.axhline(threshold, color="r", linestyle="--", label=f"Threshold (t={threshold})")
    ax.axhline(-threshold, color="r", linestyle="--")
    ax.set_ylabel("t-statistic")
    ax.set_title("Point-wise t-test")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Significant clusters
    ax = axes[2]
    ax.plot(times, erp1 - erp2, "k-", linewidth=2, label="Difference wave")

    for cluster in clusters_obs:
        if cluster["p_value"] < 0.05:
            ax.axvspan(
                times[cluster["start"]],
                times[cluster["end"]],
                alpha=0.3,
                color="red",
                label=(
                    f"p={cluster['p_value']:.3f}" if cluster == clusters_obs[0] else ""
                ),
            )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude difference (µV)")
    ax.set_title("Significant Clusters (p < 0.05)")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return fig, clusters_obs


def find_clusters(t_values, threshold):
    """Find clusters of contiguous significant t-values"""
    above_threshold = np.abs(t_values) > threshold
    clusters = []

    in_cluster = False
    for i, val in enumerate(above_threshold):
        if val and not in_cluster:
            # Start of cluster
            cluster_start = i
            in_cluster = True
        elif not val and in_cluster:
            # End of cluster
            cluster_end = i - 1
            cluster_stat = np.sum(np.abs(t_values[cluster_start : cluster_end + 1]))
            clusters.append(
                {"start": cluster_start, "end": cluster_end, "stat": cluster_stat}
            )
            in_cluster = False

    # Handle cluster extending to end
    if in_cluster:
        cluster_stat = np.sum(np.abs(t_values[cluster_start:]))
        clusters.append(
            {"start": cluster_start, "end": len(t_values) - 1, "stat": cluster_stat}
        )

    return clusters


def difference_wave_analysis(
    target_epochs, nontarget_epochs, times, channels_to_plot, labels, fs=512
):
    """
    Compute and visualize difference waves (target - non-target)

    Parameters:
        target_epochs, nontarget_epochs: (samples, channels, trials)
        times: time vector in ms
        channels_to_plot: list of channel indices
        labels: channel names
    """
    from scipy.stats import ttest_rel
    from scipy.ndimage import gaussian_filter1d

    fig, axes = plt.subplots(
        len(channels_to_plot), 1, figsize=(12, 4 * len(channels_to_plot))
    )
    if len(channels_to_plot) == 1:
        axes = [axes]

    for idx, (ax, ch_idx) in enumerate(zip(axes, channels_to_plot)):
        # Extract channel data
        target = target_epochs[:, ch_idx, :]
        nontarget = nontarget_epochs[:, ch_idx, :]

        # Average ERPs
        target_avg = target.mean(axis=1)
        nontarget_avg = nontarget.mean(axis=1)
        difference = target_avg - nontarget_avg

        # SEM
        target_sem = target.std(axis=1) / np.sqrt(target.shape[1])
        nontarget_sem = nontarget.std(axis=1) / np.sqrt(nontarget.shape[1])

        # Point-wise t-test (not corrected - just for visualization)
        t_stats = np.zeros(len(times))
        p_values = np.zeros(len(times))
        for t in range(len(times)):
            # Use equal trial counts for paired test, or independent for unequal
            min_trials = min(target.shape[1], nontarget.shape[1])
            t_stats[t], p_values[t] = ttest_rel(
                target[t, :min_trials], nontarget[t, :min_trials]
            )

        # Smooth for visualization
        target_smooth = gaussian_filter1d(target_avg, sigma=2)
        nontarget_smooth = gaussian_filter1d(nontarget_avg, sigma=2)
        diff_smooth = gaussian_filter1d(difference, sigma=2)

        # Plot ERPs
        ax.plot(times, target_smooth, "b-", label="Target", linewidth=2)
        ax.fill_between(
            times,
            target_avg - target_sem,
            target_avg + target_sem,
            color="b",
            alpha=0.2,
        )
        ax.plot(times, nontarget_smooth, "r-", label="Non-target", linewidth=2)
        ax.fill_between(
            times,
            nontarget_avg - nontarget_sem,
            nontarget_avg + nontarget_sem,
            color="r",
            alpha=0.2,
        )
        ax.plot(
            times, diff_smooth, "k-", label="Difference", linewidth=2.5, linestyle="--"
        )

        # Shade significant regions (uncorrected p < 0.01)
        sig_mask = p_values < 0.01
        if np.any(sig_mask):
            sig_regions = np.where(
                np.diff(np.concatenate(([False], sig_mask, [False])))
            )[0].reshape(-1, 2)
            for start, end in sig_regions:
                ax.axvspan(
                    times[start],
                    times[end - 1],
                    alpha=0.2,
                    color="yellow",
                    label="p<0.01" if start == sig_regions[0][0] else "",
                )

        # Highlight P300 window
        ax.axvspan(250, 600, alpha=0.1, color="green", label="P300 window")

        # Formatting
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Amplitude (µV)", fontsize=11)
        ax.set_title(
            f"Channel {labels[ch_idx]}: Target vs Non-Target",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        if idx == len(channels_to_plot) - 1:
            ax.set_xlabel("Time (ms)", fontsize=11)

    plt.tight_layout()
    plt.show()

    return fig


def butterfly_plot(epochs, times, labels, title="", highlight_channels=None):
    """
    Plot all channels overlaid (butterfly plot)

    Parameters:
        epochs: (samples, channels, trials)
        times: time vector
        labels: channel names
        highlight_channels: list of channel names to highlight
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    n_channels = epochs.shape[1]
    erps = epochs.mean(axis=2)  # Average across trials

    highlight_set = set(highlight_channels) if highlight_channels else set()

    for ch in range(n_channels):
        ch_name = labels[ch]
        if ch_name in highlight_set:
            ax.plot(times, erps[:, ch], linewidth=2.5, label=ch_name, alpha=0.9)
        else:
            ax.plot(times, erps[:, ch], color="gray", linewidth=0.5, alpha=0.4)

    # Highlight P300 window
    ax.axvspan(250, 600, alpha=0.1, color="green", label="P300 window")
    ax.axvline(0, color="k", linestyle="--", linewidth=1, label="Stimulus")
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Amplitude (µV)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig


def latency_distribution_analysis(epochs, times, channel_idx, window=(250, 600)):
    """
    Analyze distribution of single-trial P300 peak latencies

    Parameters:
        epochs: (samples, channels, trials)
        times: time vector in ms
        channel_idx: channel to analyze
        window: time window to search for peaks (ms)
    """
    from scipy import stats

    # Extract data for one channel
    data = epochs[:, channel_idx, :]  # (samples, trials)

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

    # Create figure with multiple subplots
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
    ax1.set_xlabel("Peak Latency (ms)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"P300 Latency Distribution\n(SD = {std_lat:.1f} ms)")
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
    ax2.set_xlabel("Peak Amplitude (µV)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"P300 Amplitude Distribution\n(SD = {np.std(amplitudes):.2f} µV)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Scatter: Latency vs Amplitude
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(latencies, amplitudes, alpha=0.5, s=30)

    # Add correlation
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

    ax3.set_xlabel("Peak Latency (ms)")
    ax3.set_ylabel("Peak Amplitude (µV)")
    ax3.set_title("Latency vs Amplitude Relationship")
    ax3.grid(True, alpha=0.3)

    # 4. Trial-by-trial evolution
    ax4 = fig.add_subplot(gs[1, 1])
    trial_nums = np.arange(1, len(latencies) + 1)
    ax4.plot(trial_nums, latencies, "o-", alpha=0.6, markersize=4)
    ax4.set_xlabel("Trial Number")
    ax4.set_ylabel("Peak Latency (ms)")
    ax4.set_title("Latency Across Trials (Drift Check)")
    ax4.grid(True, alpha=0.3)

    # Add trend line
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
    ax5.set_ylabel("Trial (sorted by latency)")
    ax5.set_title("Single-Trial Waveforms (sorted by P300 latency)")

    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label("Amplitude (µV)", rotation=270, labelpad=15)

    plt.suptitle(
        f"P300 Latency Analysis (n={len(latencies)} trials)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.show()

    return fig, {
        "latencies": latencies,
        "amplitudes": amplitudes,
        "mean_lat": mean_lat,
        "std_lat": std_lat,
    }


def compute_surface_laplacian_simple(epochs, montage, labels):
    """
    Approximate surface Laplacian using nearest neighbors
    Sharpens topography by removing volume conduction
    """
    # Get channel positions
    ch_pos = montage.get_positions()["ch_pos"]

    # Match labels to positions
    positions = []
    for label in labels:
        label_upper = label.upper()
        if label_upper in ch_pos:
            positions.append(ch_pos[label_upper][:2])  # x, y only
        else:
            positions.append([np.nan, np.nan])

    positions = np.array(positions)

    # Compute Laplacian for each channel
    laplacian_epochs = np.zeros_like(epochs)

    for ch in range(len(labels)):
        if np.any(np.isnan(positions[ch])):
            continue

        # Find 4 nearest neighbors
        dists = np.sqrt(np.sum((positions - positions[ch]) ** 2, axis=1))
        dists[ch] = np.inf  # Exclude self
        nearest = np.argsort(dists)[:4]

        # Laplacian = center - average of neighbors
        laplacian_epochs[:, ch, :] = epochs[:, ch, :] - np.mean(
            epochs[:, nearest, :], axis=1
        )

    return laplacian_epochs
