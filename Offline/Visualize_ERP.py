"""
© 2026 Alexandra Mikhael. All Rights Reserved.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM


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


def find_optimal_clusters(features, max_k=6):
    """
    Find optimal number of clusters using elbow method + silhouette score

    Parameters:
        features: (n_trials, n_features) array
        max_k: maximum number of clusters to test

    Returns:
        optimal_k: best number of clusters
        all_scores: dict with inertia and silhouette scores
    """
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features, labels))

    # Plot elbow curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Elbow method
    ax1.plot(K_range, inertias, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
    ax1.set_title("Elbow Method", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Silhouette score
    ax2.plot(K_range, silhouette_scores, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Clusters (k)", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("Silhouette Score (Higher is Better)", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color="g", linestyle="--", label="Good threshold")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Find optimal k (highest silhouette score)
    optimal_k = K_range[np.argmax(silhouette_scores)]

    print(f"\n{'='*60}")
    print(f"OPTIMAL NUMBER OF CLUSTERS")
    print(f"{'='*60}")
    print(f"Recommended k: {optimal_k}")
    print(f"Silhouette score: {max(silhouette_scores):.3f}")
    print(f"\nAll scores:")
    for k, inertia, sil in zip(K_range, inertias, silhouette_scores):
        marker = " ← BEST" if k == optimal_k else ""
        print(f"  k={k}: Inertia={inertia:.1f}, Silhouette={sil:.3f}{marker}")
    print(f"{'='*60}\n")

    return optimal_k, {
        "K_range": list(K_range),
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
    }


def cluster_based_comparison(
    epochs_pre, epochs_post, times, channel_idx, n_permutations=1000
):
    """
    Cluster-based permutation test for ERP comparison (Pre vs Post)

    Returns:
        fig: matplotlib figure
        stats_dict: dictionary with keys:
            - 't_obs': observed t-statistics
            - 'p_obs': observed p-values
            - 'significant_clusters': list of (start_idx, end_idx, mass, p_value)
            - 'all_clusters': all detected clusters
            - 'global_p': global p-value
    """
    from scipy import stats
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract data for specified channel
    data_pre = epochs_pre[:, channel_idx, :]
    data_post = epochs_post[:, channel_idx, :]

    n_samples = data_pre.shape[0]
    n_trials_pre = data_pre.shape[1]
    n_trials_post = data_post.shape[1]

    # Compute observed t-statistic at each time point
    t_obs = np.zeros(n_samples)
    p_obs = np.zeros(n_samples)

    for t in range(n_samples):
        t_stat, p_val = stats.ttest_ind(data_pre[t, :], data_post[t, :])
        t_obs[t] = t_stat
        p_obs[t] = p_val

    # Identify clusters (p < 0.05)
    cluster_threshold = 0.05
    significant_mask = p_obs < cluster_threshold

    # Find contiguous clusters
    def find_clusters(mask):
        clusters = []
        in_cluster = False
        cluster_start = 0

        for i in range(len(mask)):
            if mask[i] and not in_cluster:
                in_cluster = True
                cluster_start = i
            elif not mask[i] and in_cluster:
                clusters.append((cluster_start, i - 1))
                in_cluster = False

        if in_cluster:
            clusters.append((cluster_start, len(mask) - 1))

        return clusters

    observed_clusters = find_clusters(significant_mask)

    # Compute cluster masses
    cluster_masses = []
    for start, end in observed_clusters:
        mass = np.sum(np.abs(t_obs[start : end + 1]))
        cluster_masses.append(mass)

    max_cluster_mass_obs = max(cluster_masses) if cluster_masses else 0

    print(f"Found {len(observed_clusters)} initial clusters")
    print(f"Max observed cluster mass: {max_cluster_mass_obs:.2f}")

    # Permutation test
    print(f"Running {n_permutations} permutations...")

    max_cluster_masses_perm = np.zeros(n_permutations)
    all_data = np.concatenate([data_pre, data_post], axis=1)
    total_trials = n_trials_pre + n_trials_post

    for perm in range(n_permutations):
        if perm % 100 == 0:
            print(f"  Permutation {perm}/{n_permutations}")

        # Shuffle
        perm_indices = np.random.permutation(total_trials)
        perm_group1 = all_data[:, perm_indices[:n_trials_pre]]
        perm_group2 = all_data[:, perm_indices[n_trials_pre:]]

        # Compute t-statistics
        t_perm = np.zeros(n_samples)
        p_perm = np.zeros(n_samples)

        for t in range(n_samples):
            t_stat, p_val = stats.ttest_ind(perm_group1[t, :], perm_group2[t, :])
            t_perm[t] = t_stat
            p_perm[t] = p_val

        # Find clusters
        perm_mask = p_perm < cluster_threshold
        perm_clusters = find_clusters(perm_mask)

        # Get max cluster mass
        if perm_clusters:
            perm_masses = [np.sum(np.abs(t_perm[s : e + 1])) for s, e in perm_clusters]
            max_cluster_masses_perm[perm] = max(perm_masses)

    # Determine significance
    significant_clusters = []
    cluster_p_values = []
    cluster_alpha = 0.05

    for i, (start, end) in enumerate(observed_clusters):
        mass = cluster_masses[i]
        p_cluster = np.sum(max_cluster_masses_perm >= mass) / n_permutations
        cluster_p_values.append(p_cluster)

        if p_cluster < cluster_alpha:
            significant_clusters.append((start, end, mass, p_cluster))
            print(
                f"✓ Significant: {times[start]:.0f}-{times[end]:.0f} ms, p={p_cluster:.4f}"
            )

    # Global p-value
    p_global = np.sum(max_cluster_masses_perm >= max_cluster_mass_obs) / n_permutations

    # Create figure (your existing plotting code)
    fig = plt.figure(figsize=(15, 10))
    # ... [all your plotting code] ...

    # CRITICAL: Return dictionary, not list
    return fig, {
        "t_obs": t_obs,
        "p_obs": p_obs,
        "significant_clusters": significant_clusters,
        "all_clusters": observed_clusters,
        "cluster_p_values": cluster_p_values,
        "permutation_distribution": max_cluster_masses_perm,
        "global_p": p_global,
    }


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


def run_mixed_effects_analysis(
    all_subjects_data, condition_pair, times, channel_idx=26
):
    """
    Run mixed effects model comparing two conditions across subjects.

    Parameters:
    -----------
    all_subjects_data : dict
        Dictionary with subject IDs as keys
    condition_pair : tuple
        e.g., ('nback_pre_target_all', 'nback_post_target_all')
    times : array
        Time points array
    channel_idx : int
        Channel index to analyze (default 26 for Pz)
    """
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import pandas as pd
    from statsmodels.regression.mixed_linear_model import MixedLM

    # Suppress convergence warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    condition1_key, condition2_key = condition_pair

    # Prepare data in long format
    data_list = []

    for subj_id, subj_data in all_subjects_data.items():
        # Extract channel first: (time, channels, trials) -> (time, trials)
        cond1 = subj_data[condition1_key][:, channel_idx, :]
        cond2 = subj_data[condition2_key][:, channel_idx, :]

        n_timepoints, n_cond1_trials = cond1.shape
        _, n_cond2_trials = cond2.shape

        # Add condition 1 trials
        for trial in range(n_cond1_trials):
            for t_idx in range(n_timepoints):
                data_list.append(
                    {
                        "subject": subj_id,
                        "time_idx": t_idx,
                        "time": times[t_idx] if times is not None else t_idx,
                        "amplitude": cond1[t_idx, trial],
                        "condition": "condition1",
                        "trial": trial,
                    }
                )

        # Add condition 2 trials
        for trial in range(n_cond2_trials):
            for t_idx in range(n_timepoints):
                data_list.append(
                    {
                        "subject": subj_id,
                        "time_idx": t_idx,
                        "time": times[t_idx] if times is not None else t_idx,
                        "amplitude": cond2[t_idx, trial],
                        "condition": "condition2",
                        "trial": trial,
                    }
                )

    df = pd.DataFrame(data_list)

    # Run mixed effects model at each time point
    results = []
    for t_idx in range(len(times)):
        df_time = df[df["time_idx"] == t_idx]

        try:
            # Mixed effects: condition as fixed effect, random intercept per subject
            model = MixedLM.from_formula(
                "amplitude ~ condition", data=df_time, groups=df_time["subject"]
            )
            result = model.fit(reml=True, method="nm")  # Use REML

            results.append(
                {
                    "time": times[t_idx] if times is not None else t_idx,
                    "time_idx": t_idx,
                    "t_stat": result.tvalues["condition[T.condition2]"],
                    "p_value": result.pvalues["condition[T.condition2]"],
                    "coef": result.params["condition[T.condition2]"],
                    "converged": result.converged,
                }
            )
        except Exception as e:
            results.append(
                {
                    "time": times[t_idx] if times is not None else t_idx,
                    "time_idx": t_idx,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "coef": np.nan,
                    "converged": False,
                }
            )

    results_df = pd.DataFrame(results)
    return results_df


def difference_wave_analysis(
    target_epochs,
    nontarget_epochs,
    times,
    channels_to_plot,
    labels,
    all,
    comparison_type="target_vs_nontarget",
    condition_prefix="nback_pre",
    fs=512,
):
    """
    Compute and visualize difference waves

    Parameters:
        target_epochs, nontarget_epochs: (samples, channels, trials) - for backward compatibility
        times: time vector in ms
        channels_to_plot: list of channel indices
        labels: channel names
        all_subjects_data: dict with all subjects' data
        comparison_type: 'target_vs_nontarget' or 'pre_vs_post'
        condition_prefix: 'nback_pre', 'nback_post', 'model_pre', etc.
        fs: sampling rate
    """
    from scipy.stats import ttest_rel
    from scipy.ndimage import gaussian_filter1d

    fig, axes = plt.subplots(
        len(channels_to_plot), 1, figsize=(12, 4 * len(channels_to_plot))
    )
    if len(channels_to_plot) == 1:
        axes = [axes]

    for idx, (ax, ch_idx) in enumerate(zip(axes, channels_to_plot)):

        # Determine which comparison to make
        if comparison_type == "target_vs_nontarget":
            condition_pair = (
                f"{condition_prefix}_target_all",
                f"{condition_prefix}_nontarget_all",
            )
            title_suffix = "Target vs Non-Target"
            label1, label2 = "Target", "Non-target"

            # Extract channel data for plotting (backward compatible)
            target = target_epochs[:, ch_idx, :]
            nontarget = nontarget_epochs[:, ch_idx, :]

        elif comparison_type == "pre_vs_post":
            # Extract condition name (nback, model, online)
            condition_name = condition_prefix.split("_")[
                0
            ]  # e.g., 'nback' from 'nback_pre'

            # Determine if comparing target or nontarget
            if "target" in condition_prefix or condition_prefix.endswith("_target"):
                condition_pair = (
                    f"{condition_name}_pre_target_all",
                    f"{condition_name}_post_target_all",
                )
                suffix = "Target"
            else:
                condition_pair = (
                    f"{condition_name}_pre_nontarget_all",
                    f"{condition_name}_post_nontarget_all",
                )
                suffix = "Non-Target"

            title_suffix = f"{suffix}: Pre vs Post"
            label1, label2 = "Pre", "Post"

            # Extract data for plotting from all_subjects_data
            target = extract_averaged_data(all, condition_pair[0], ch_idx)
            nontarget = extract_averaged_data(all, condition_pair[1], ch_idx)

        else:
            raise ValueError(f"Unknown comparison_type: {comparison_type}")

        # Average ERPs
        target_avg = target.mean(axis=1) if target.ndim > 1 else target
        nontarget_avg = nontarget.mean(axis=1) if nontarget.ndim > 1 else nontarget
        difference = target_avg - nontarget_avg

        # SEM
        if target.ndim > 1:
            target_sem = target.std(axis=1) / np.sqrt(target.shape[1])
            nontarget_sem = nontarget.std(axis=1) / np.sqrt(nontarget.shape[1])
        else:
            target_sem = np.zeros_like(target)
            nontarget_sem = np.zeros_like(nontarget)

        # Run mixed effects analysis
        results = run_mixed_effects_analysis(
            all,
            condition_pair,
            times,
            channel_idx=ch_idx,
        )

        # Extract arrays for plotting
        t_stats = results["t_stat"].values
        p_values = results["p_value"].values

        print(
            f"{title_suffix} - Channel {labels[ch_idx-1]} - Significant time points (p < 0.05): {np.sum(p_values < 0.05)}"
        )

        # Smooth for visualization
        target_smooth = gaussian_filter1d(target_avg, sigma=2)
        nontarget_smooth = gaussian_filter1d(nontarget_avg, sigma=2)
        diff_smooth = gaussian_filter1d(difference, sigma=2)

        # Plot ERPs
        ax.plot(times, target_smooth, "b-", label=label1, linewidth=2)
        ax.fill_between(
            times,
            target_avg - target_sem,
            target_avg + target_sem,
            color="b",
            alpha=0.2,
        )
        ax.plot(times, nontarget_smooth, "r-", label=label2, linewidth=2)
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
            f"Channel {labels[ch_idx-1]}: {title_suffix}",
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


def extract_averaged_data(all_subjects_data, condition_key, ch_idx):
    """
    Extract and average data across subjects for a given condition and channel.
    Returns: (timepoints, subjects) array
    """
    subject_averages = []

    for subj_id, subj_data in all_subjects_data.items():
        # Extract channel: (time, channels, trials) -> (time, trials) -> (time,)
        data = subj_data[condition_key][:, ch_idx, :].mean(axis=1)
        subject_averages.append(data)

    # Stack: shape (timepoints, n_subjects)
    return np.array(subject_averages).T


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


def extract_erp_features_multi_subject(
    all_subjects_data, condition_key, channel_idx, n100_window, p300_window, times
):
    """
    Extract N100 and P300 features for all subjects with subject tracking.

    Parameters:
    -----------
    all_subjects_data : dict
        Dictionary with subject IDs as keys
    condition_key : str
        e.g., 'nback_pre_target_all'
    channel_idx : int
        Channel to analyze
    n100_window : tuple
        (start, end) in ms for N100
    p300_window : tuple
        (start, end) in ms for P300
    times : array
        Time vector in ms

    Returns:
    --------
    DataFrame with columns: subject, trial, n100_amp, n100_lat, p300_amp, p300_lat, interval
    """
    n100_mask = (times >= n100_window[0]) & (times <= n100_window[1])
    p300_mask = (times >= p300_window[0]) & (times <= p300_window[1])

    all_features = []

    for subj_id, subj_data in all_subjects_data.items():
        # Extract data: (time, channels, trials) -> (time, trials)
        data = subj_data[condition_key][:, channel_idx, :]
        n_trials = data.shape[1]

        for trial in range(n_trials):
            # N100 (negative peak)
            n100_signal = data[n100_mask, trial]
            n100_min_idx = np.argmin(n100_signal)
            n100_amp = n100_signal[n100_min_idx]
            n100_lat = times[n100_mask][n100_min_idx]

            # P300 (positive peak)
            p300_signal = data[p300_mask, trial]
            p300_max_idx = np.argmax(p300_signal)
            p300_amp = p300_signal[p300_max_idx]
            p300_lat = times[p300_mask][p300_max_idx]

            # Compute interval
            interval = p300_lat - n100_lat

            all_features.append(
                {
                    "subject": subj_id,
                    "trial": trial,
                    "n100_amp": n100_amp,
                    "n100_lat": n100_lat,
                    "p300_amp": p300_amp,
                    "p300_lat": p300_lat,
                    "n100_p300_interval": interval,
                }
            )

    return pd.DataFrame(all_features)

def analyze_n100_p300_relationship(df):
    """
    Run mixed effects models to analyze N100-P300 relationships.

    Parameters:
    -----------
    df : DataFrame
        Output from extract_erp_features_multi_subject

    Returns:
    --------
    Dictionary of model results
    """
    results = {}

    # Model 1: Does N100 amplitude predict P300 amplitude?
    print("\n=== Model 1: N100 Amplitude → P300 Amplitude ===")
    model1 = MixedLM.from_formula(
        "p300_amp ~ n100_amp",  # Fixed effect
        data=df,
        groups=df["subject"],  # Random intercept per subject
    )
    result1 = model1.fit(reml=True)
    print(result1.summary())
    results["n100_to_p300_amp"] = result1

    # Model 2: Does N100 latency predict P300 latency?
    print("\n=== Model 2: N100 Latency → P300 Latency ===")
    model2 = MixedLM.from_formula("p300_lat ~ n100_lat", data=df, groups=df["subject"])
    result2 = model2.fit(reml=True)
    print(result2.summary())
    results["n100_to_p300_lat"] = result2

    # Model 3: Does N100 amplitude predict N100-P300 interval?
    print("\n=== Model 3: N100 Amplitude → Interval ===")
    model3 = MixedLM.from_formula(
        "n100_p300_interval ~ n100_amp", data=df, groups=df["subject"]
    )
    result3 = model3.fit(reml=True)
    print(result3.summary())
    results["n100_amp_to_interval"] = result3

    # Model 4: Does P300 amplitude predict interval?
    print("\n=== Model 4: P300 Amplitude → Interval ===")
    model4 = MixedLM.from_formula(
        "n100_p300_interval ~ p300_amp", data=df, groups=df["subject"]
    )
    result4 = model4.fit(reml=True)
    print(result4.summary())
    results["p300_amp_to_interval"] = result4

    # Model 5: Multiple predictors - N100 and P300 amplitudes predict interval
    print("\n=== Model 5: N100 + P300 Amplitudes → Interval ===")
    model5 = MixedLM.from_formula(
        "n100_p300_interval ~ n100_amp + p300_amp", data=df, groups=df["subject"]
    )
    result5 = model5.fit(reml=True)
    print(result5.summary())
    results["both_amps_to_interval"] = result5

    return results


import seaborn as sns


def plot_n100_p300_relationship(df, results):
    """Plot N100-P300 relationships with mixed effects fit, keeping only valid trials."""

    # 🔧 FILTER: Keep only trials where N100 < 0 and P300 > 0
    df_filtered = df[(df["n100_amp"] < 0) & (df["p300_amp"] > 0)].copy()

    n_original = len(df)
    n_filtered = len(df_filtered)
    n_removed = n_original - n_filtered
    pct_removed = 100 * n_removed / n_original

    print(f"\n📊 Data Filtering Summary:")
    print(f"   Original trials: {n_original}")
    print(f"   Valid trials (N100<0 & P300>0): {n_filtered}")
    print(f"   Removed trials: {n_removed} ({pct_removed:.1f}%)")

    # Count per subject
    print(f"\n   Trials per subject after filtering:")
    for subj in df_filtered["subject"].unique():
        n_subj = len(df_filtered[df_filtered["subject"] == subj])
        print(f"      Subject {subj}: {n_subj} trials")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: N100 amp vs P300 amp
    ax = axes[0, 0]
    for subj in df_filtered["subject"].unique():
        subj_data = df_filtered[df_filtered["subject"] == subj]
        ax.scatter(subj_data["n100_amp"], subj_data["p300_amp"], alpha=0.5, s=20)

    # Add regression line from original model (or refit if you prefer)
    coef = results["n100_to_p300_amp"].params["n100_amp"]
    intercept = results["n100_to_p300_amp"].params["Intercept"]
    x_range = np.linspace(
        df_filtered["n100_amp"].min(), df_filtered["n100_amp"].max(), 100
    )
    ax.plot(
        x_range,
        intercept + coef * x_range,
        "r-",
        linewidth=2,
        label=f'β={coef:.3f}, p={results["n100_to_p300_amp"].pvalues["n100_amp"]:.4f}',
    )
    ax.set_xlabel("N100 Amplitude (µV)")
    ax.set_ylabel("P300 Amplitude (µV)")

    ax.set_title(f"N100 → P300 Amplitude (n={n_filtered} valid trials)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.axvline(0, color="k", linestyle="--", alpha=0.3)

    # Plot 2: N100 lat vs P300 lat
    ax = axes[0, 1]
    for subj in df_filtered["subject"].unique():
        subj_data = df_filtered[df_filtered["subject"] == subj]
        ax.scatter(subj_data["n100_lat"], subj_data["p300_lat"], alpha=0.5, s=20)

    coef = results["n100_to_p300_lat"].params["n100_lat"]
    intercept = results["n100_to_p300_lat"].params["Intercept"]
    x_range = np.linspace(
        df_filtered["n100_lat"].min(), df_filtered["n100_lat"].max(), 100
    )
    ax.plot(
        x_range,
        intercept + coef * x_range,
        "r-",
        linewidth=2,
        label=f'β={coef:.3f}, p={results["n100_to_p300_lat"].pvalues["n100_lat"]:.4f}',
    )
    ax.set_xlabel("N100 Latency (ms)")
    ax.set_ylabel("P300 Latency (ms)")
    ax.set_title(f"N100 → P300 Latency (n={n_filtered})")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: N100 amp vs Interval
    ax = axes[1, 0]
    for subj in df_filtered["subject"].unique():
        subj_data = df_filtered[df_filtered["subject"] == subj]
        ax.scatter(
            subj_data["n100_amp"], subj_data["n100_p300_interval"], alpha=0.5, s=20
        )

    coef = results["n100_amp_to_interval"].params["n100_amp"]
    intercept = results["n100_amp_to_interval"].params["Intercept"]
    x_range = np.linspace(
        df_filtered["n100_amp"].min(), df_filtered["n100_amp"].max(), 100
    )
    ax.plot(
        x_range,
        intercept + coef * x_range,
        "r-",
        linewidth=2,
        label=f'β={coef:.3f}, p={results["n100_amp_to_interval"].pvalues["n100_amp"]:.4f}',
    )
    ax.set_xlabel("N100 Amplitude (µV)")
    ax.set_ylabel("N100-P300 Interval (ms)")
    ax.set_title(f"N100 Amplitude → Interval (n={n_filtered})")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axvline(0, color="k", linestyle="--", alpha=0.3)

    # Plot 4: P300 amp vs Interval
    ax = axes[1, 1]
    for subj in df_filtered["subject"].unique():
        subj_data = df_filtered[df_filtered["subject"] == subj]
        ax.scatter(
            subj_data["p300_amp"],
            subj_data["n100_p300_interval"],
            alpha=0.5,
            s=20,
        )

    coef = results["p300_amp_to_interval"].params["p300_amp"]
    intercept = results["p300_amp_to_interval"].params["Intercept"]
    x_range = np.linspace(
        df_filtered["p300_amp"].min(), df_filtered["p300_amp"].max(), 100
    )
    ax.plot(
        x_range,
        intercept + coef * x_range,
        "r-",
        linewidth=2,
        label=f'β={coef:.3f}, p={results["p300_amp_to_interval"].pvalues["p300_amp"]:.4f}',
    )
    ax.set_xlabel("P300 Amplitude (µV)")
    ax.set_ylabel("N100-P300 Interval (ms)")
    ax.set_title(f"P300 Amplitude → Interval (n={n_filtered})")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axvline(0, color="k", linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig, df_filtered


def identify_early_components(
    erp_data,
    channel_labels,
    time_ms,
    condition_name="Unknown",
    data_format="time_ch_trials",
):
    """
    Identify and characterize early ERP components (N1, P2, etc.)
    """

    # ===== HANDLE DATA FORMAT =====
    if data_format == "time_ch_trials":
        print(f"Input shape: {erp_data.shape} (time, channels, trials)")
        erp_data = np.transpose(erp_data, (2, 1, 0))  # → (trials, channels, time)
        print(f"Transposed to: {erp_data.shape} (trials, channels, time)")

    n_trials, n_channels, n_timepoints = erp_data.shape

    # Find key channels
    channel_labels_upper = [ch.upper() for ch in channel_labels]

    # Find channels (with fallbacks)
    pz_idx = channel_labels_upper.index("PZ") if "PZ" in channel_labels_upper else None

    # For frontal: try FZ, FC1, FC2
    fz_idx = None
    for ch in ["FZ", "FC1", "FC2", "CZ"]:
        if ch in channel_labels_upper:
            fz_idx = channel_labels_upper.index(ch)
            fz_name = ch
            break

    # For occipital: try OZ, POZ, O1
    oz_idx = None
    for ch in ["OZ", "POZ", "O1", "O2"]:
        if ch in channel_labels_upper:
            oz_idx = channel_labels_upper.index(ch)
            oz_name = ch
            break

    if pz_idx is None or fz_idx is None or oz_idx is None:
        print(f"⚠️ Required channels not found!")
        print(f"   PZ: {'✓' if pz_idx else '✗'}")
        print(f"   FZ: {'✓' if fz_idx else '✗'}")
        print(f"   OZ: {'✓' if oz_idx else '✗'}")
        return None

    print(f"\nUsing channels:")
    print(f"  Parietal: PZ (index {pz_idx})")
    print(f"  Frontal: {fz_name} (index {fz_idx})")
    print(f"  Occipital: {oz_name} (index {oz_idx})")

    # Grand average - NOW CORRECT SHAPE
    grand_avg = np.mean(erp_data, axis=0)  # Average across trials → (channels, time)
    print(f"Grand average shape: {grand_avg.shape} (channels, time)")

    # Create comprehensive plot
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # ===== PLOT 1: Full waveform at Pz =====
    ax = axes[0, 0]
    ax.plot(time_ms, grand_avg[pz_idx, :], "b-", linewidth=2.5, label="Pz")

    # Mark component windows
    ax.axvspan(80, 150, alpha=0.2, color="purple", label="N1 window")
    ax.axvspan(150, 250, alpha=0.2, color="orange", label="P2 window")
    ax.axvspan(200, 280, alpha=0.2, color="pink", label="N2 window")
    ax.axvspan(300, 500, alpha=0.2, color="red", label="P300 window")

    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.5)

    ax.set_xlabel("Time (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Amplitude (μV)", fontsize=12, fontweight="bold")
    ax.set_title(f"{condition_name}: Full ERP at Pz", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time_ms.min(), time_ms.max()])

    # ===== PLOT 2: Zoom on early components (0-300ms) =====
    ax = axes[0, 1]
    ax.plot(time_ms, grand_avg[pz_idx, :], "b-", linewidth=3, label="Pz", alpha=0.8)
    ax.plot(time_ms, grand_avg[fz_idx, :], "r-", linewidth=3, label=fz_name, alpha=0.8)
    ax.plot(time_ms, grand_avg[oz_idx, :], "g-", linewidth=3, label=oz_name, alpha=0.8)

    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.5)

    ax.set_xlabel("Time (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Amplitude (μV)", fontsize=12, fontweight="bold")
    ax.set_title("Early Components (0-300ms)", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    early_window = (time_ms >= 0) & (time_ms <= 300)
    ax.set_xlim([0, 300])

    # ===== ANALYZE COMPONENTS =====
    components = {}

    # P1 (50-100ms) - Positive, occipital
    p1_window = (time_ms >= 50) & (time_ms <= 100)
    if p1_window.sum() > 0:
        p1_data = grand_avg[oz_idx, p1_window]
        p1_peak = p1_data.max()
        p1_lat = time_ms[p1_window][np.argmax(p1_data)]
        components["P1"] = {"peak": p1_peak, "latency": p1_lat, "channel": oz_name}

    # N1 (80-150ms) - Negative, central/occipital
    n1_window = (time_ms >= 80) & (time_ms <= 150)
    if n1_window.sum() > 0:
        n1_data_pz = grand_avg[pz_idx, n1_window]
        n1_data_oz = grand_avg[oz_idx, n1_window]

        n1_peak_pz = n1_data_pz.min()
        n1_lat_pz = time_ms[n1_window][np.argmin(n1_data_pz)]

        n1_peak_oz = n1_data_oz.min()
        n1_lat_oz = time_ms[n1_window][np.argmin(n1_data_oz)]

        # Use whichever is more negative
        if n1_peak_oz < n1_peak_pz:
            components["N1"] = {
                "peak": n1_peak_oz,
                "latency": n1_lat_oz,
                "channel": oz_name,
            }
        else:
            components["N1"] = {
                "peak": n1_peak_pz,
                "latency": n1_lat_pz,
                "channel": "PZ",
            }

    # P2 (150-250ms) - Positive, frontal/central
    p2_window = (time_ms >= 150) & (time_ms <= 250)
    if p2_window.sum() > 0:
        p2_data_pz = grand_avg[pz_idx, p2_window]
        p2_data_fz = grand_avg[fz_idx, p2_window]

        p2_peak_pz = p2_data_pz.max()
        p2_lat_pz = time_ms[p2_window][np.argmax(p2_data_pz)]

        p2_peak_fz = p2_data_fz.max()
        p2_lat_fz = time_ms[p2_window][np.argmax(p2_data_fz)]

        # Use whichever is larger
        if p2_peak_fz > p2_peak_pz:
            components["P2"] = {
                "peak": p2_peak_fz,
                "latency": p2_lat_fz,
                "channel": fz_name,
            }
        else:
            components["P2"] = {
                "peak": p2_peak_pz,
                "latency": p2_lat_pz,
                "channel": "PZ",
            }

    # N2 (180-280ms) - Negative, frontal
    n2_window = (time_ms >= 180) & (time_ms <= 280)
    if n2_window.sum() > 0:
        n2_data_fz = grand_avg[fz_idx, n2_window]
        n2_peak_fz = n2_data_fz.min()
        n2_lat_fz = time_ms[n2_window][np.argmin(n2_data_fz)]
        components["N2"] = {
            "peak": n2_peak_fz,
            "latency": n2_lat_fz,
            "channel": fz_name,
        }

    # P300 (300-500ms) - Positive, parietal
    p300_window = (time_ms >= 300) & (time_ms <= 500)
    if p300_window.sum() > 0:
        p300_data = grand_avg[pz_idx, p300_window]
        p300_peak = p300_data.max()
        p300_lat = time_ms[p300_window][np.argmax(p300_data)]
        components["P300"] = {"peak": p300_peak, "latency": p300_lat, "channel": "PZ"}

    # ===== PLOT 3: Component peaks marked =====
    ax = axes[1, 0]
    ax.plot(time_ms, grand_avg[pz_idx, :], "b-", linewidth=2.5, alpha=0.7, label="Pz")

    colors = {
        "P1": "green",
        "N1": "purple",
        "P2": "orange",
        "N2": "brown",
        "P300": "red",
    }

    for comp_name, comp_data in components.items():
        lat = comp_data["latency"]
        peak = comp_data["peak"]
        color = colors.get(comp_name, "black")

        ax.plot(
            lat,
            peak,
            "o",
            markersize=12,
            color=color,
            label=f"{comp_name}: {peak:.2f}μV @ {lat:.0f}ms",
            zorder=5,
        )
        ax.axvline(x=lat, color=color, linestyle="--", alpha=0.3)

    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Amplitude (μV)", fontsize=12, fontweight="bold")
    ax.set_title("Identified Components at Pz", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 600])

    # ===== PLOT 4: Topography at N1 peak =====
    ax = axes[1, 1]
    if "N1" in components:
        n1_time_idx = np.argmin(np.abs(time_ms - components["N1"]["latency"]))
        voltage_at_n1 = grand_avg[:, n1_time_idx]

        colors_topo = ["blue" if v < 0 else "red" for v in voltage_at_n1]
        bars = ax.bar(
            range(len(channel_labels)),
            voltage_at_n1,
            color=colors_topo,
            alpha=0.7,
            edgecolor="black",
        )

        ax.axhline(y=0, color="k", linestyle="-", linewidth=2)
        ax.set_xticks(range(len(channel_labels)))
        ax.set_xticklabels(channel_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Amplitude (μV)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Scalp Distribution at N1 Peak ({components['N1']['latency']:.0f}ms)",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(
            0.5,
            0.5,
            "N1 not detected",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")

    # ===== PLOT 5: Comparison table =====
    ax = axes[2, 0]
    ax.axis("off")

    table_data = []
    table_data.append(["Component", "Peak (μV)", "Latency (ms)", "Channel", "Present?"])
    table_data.append(["-" * 10, "-" * 10, "-" * 12, "-" * 8, "-" * 9])

    for comp_name in ["P1", "N1", "P2", "N2", "P300"]:
        if comp_name in components:
            data = components[comp_name]
            present = "✓" if abs(data["peak"]) > 1 else "?"
            table_data.append(
                [
                    comp_name,
                    f"{data['peak']:.2f}",
                    f"{data['latency']:.0f}",
                    data["channel"],
                    present,
                ]
            )
        else:
            table_data.append([comp_name, "N/A", "N/A", "N/A", "✗"])

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.15, 0.15, 0.2, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code header
    for i in range(5):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax.set_title("Component Summary", fontsize=14, fontweight="bold", pad=20)

    # ===== PLOT 6: Interpretation =====
    ax = axes[2, 1]
    ax.axis("off")

    interpretation = []
    interpretation.append("COMPONENT INTERPRETATION:\n")

    if "N1" in components:
        n1 = components["N1"]
        interpretation.append(
            f"✓ N1 PRESENT: {n1['peak']:.2f}μV @ {n1['latency']:.0f}ms"
        )
        interpretation.append(f"   Location: {n1['channel']}")
        interpretation.append(f"   Function: Early sensory processing")
        if n1["peak"] < -1:
            interpretation.append(f"   This is the NEGATIVE peak before P300\n")
        else:
            interpretation.append(f"   ⚠️ Very weak (< 1μV)\n")
    else:
        interpretation.append(f"✗ N1 NOT DETECTED\n")

    if "P2" in components:
        p2 = components["P2"]
        interpretation.append(
            f"✓ P2 PRESENT: {p2['peak']:.2f}μV @ {p2['latency']:.0f}ms"
        )
        interpretation.append(f"   Location: {p2['channel']}")
        interpretation.append(f"   Function: Attention orienting\n")

    if "N2" in components and components["N2"]["peak"] < -1:
        n2 = components["N2"]
        interpretation.append(
            f"✓ N2 PRESENT: {n2['peak']:.2f}μV @ {n2['latency']:.0f}ms"
        )
        interpretation.append(f"   Location: {n2['channel']} (frontal)")
        interpretation.append(f"   Function: Conflict detection\n")
    else:
        interpretation.append(f"✗ N2 ABSENT (expected for target detection)\n")

    if "P300" in components:
        p300 = components["P300"]
        interpretation.append(
            f"✓ P300 PRESENT: {p300['peak']:.2f}μV @ {p300['latency']:.0f}ms"
        )
        interpretation.append(f"   Location: {p300['channel']} (parietal)")
        interpretation.append(f"   Function: Target evaluation\n")

    interpretation.append("\nCONCLUSION:")
    if "N1" in components and components["N1"]["peak"] < -1:
        interpretation.append("The negative peak before P300 is N1,")
        interpretation.append("reflecting early sensory/attention processing.")
        interpretation.append("\nN1-P300 correlation would measure:")
        interpretation.append("  Early attention → Later evaluation")
    elif "P2" in components:
        interpretation.append("Main early component is P2 (positive),")
        interpretation.append("suggesting attention-driven processing.")
    else:
        interpretation.append("No clear early negative component.")

    ax.text(
        0.05,
        0.95,
        "\n".join(interpretation),
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()

    # ===== PRINT SUMMARY =====
    print("\n" + "=" * 70)
    print("EARLY COMPONENT ANALYSIS")
    print("=" * 70)

    for comp_name, comp_data in components.items():
        print(f"\n{comp_name}:")
        print(f"  Peak: {comp_data['peak']:.2f} μV")
        print(f"  Latency: {comp_data['latency']:.0f} ms")
        print(f"  Channel: {comp_data['channel']}")

    print("\n" + "=" * 70)

    return components


def hemispheric_lateralization_analysis(
    all_subjects_data,
    condition_key,
    left_channels,
    right_channels,
    labels,
    times,
    p300_window=(300, 600),
):
    """
    Hemispheric lateralization analysis using mixed effects models with visualization.
    """
    from scipy import stats
    import pandas as pd
    from statsmodels.regression.mixed_linear_model import MixedLM
    import matplotlib.pyplot as plt
    import numpy as np

    # Find channel indices
    labels_upper = [l.upper() for l in labels]
    left_idx = [
        labels_upper.index(ch.upper())
        for ch in left_channels
        if ch.upper() in labels_upper
    ]
    right_idx = [
        labels_upper.index(ch.upper())
        for ch in right_channels
        if ch.upper() in labels_upper
    ]

    # Prepare data for MEM and collect waveforms
    data_list = []
    p300_mask = (times >= p300_window[0]) & (times <= p300_window[1])

    # Store subject-averaged waveforms for plotting
    left_waveforms = []
    right_waveforms = []

    for subj_id, subj_data in all_subjects_data.items():
        # Extract condition data: (time, channels, trials)
        epochs = subj_data[condition_key]

        # Average left and right hemispheres
        left_data = epochs[:, left_idx, :].mean(axis=1)  # (time, trials)
        right_data = epochs[:, right_idx, :].mean(axis=1)

        # Store subject-averaged waveform
        left_waveforms.append(left_data.mean(axis=1))
        right_waveforms.append(right_data.mean(axis=1))

        n_trials = left_data.shape[1]

        for trial in range(n_trials):
            # Extract P300 peak for this trial
            left_p300 = np.max(left_data[p300_mask, trial])
            right_p300 = np.max(right_data[p300_mask, trial])

            # Compute lateralization index
            li = (left_p300 - right_p300) / (left_p300 + right_p300 + 1e-10)

            data_list.append(
                {
                    "subject": subj_id,
                    "trial": trial,
                    "left_p300": left_p300,
                    "right_p300": right_p300,
                    "lateralization_index": li,
                }
            )

    df = pd.DataFrame(data_list)

    # Convert waveforms to arrays
    left_waveforms = np.array(left_waveforms)  # (n_subjects, timepoints)
    right_waveforms = np.array(right_waveforms)

    # Grand averages across subjects
    left_grand_avg = left_waveforms.mean(axis=0)
    right_grand_avg = right_waveforms.mean(axis=0)

    # SEM across subjects
    left_sem = left_waveforms.std(axis=0) / np.sqrt(left_waveforms.shape[0])
    right_sem = right_waveforms.std(axis=0) / np.sqrt(right_waveforms.shape[0])

    # Reshape to long format for MEM
    df_long = []
    for _, row in df.iterrows():
        df_long.append(
            {
                "subject": row["subject"],
                "trial": row["trial"],
                "amplitude": row["left_p300"],
                "hemisphere": "left",
            }
        )
        df_long.append(
            {
                "subject": row["subject"],
                "trial": row["trial"],
                "amplitude": row["right_p300"],
                "hemisphere": "right",
            }
        )

    df_long = pd.DataFrame(df_long)

    # Mixed effects model
    print("\n=== Mixed Effects Model: Hemisphere Effect ===")
    model = MixedLM.from_formula(
        "amplitude ~ hemisphere", data=df_long, groups=df_long["subject"]
    )
    result = model.fit(reml=True)
    print(result.summary())

    # Extract results
    hem_effect = result.params["hemisphere[T.right]"]
    hem_pval = result.pvalues["hemisphere[T.right]"]

    # One-sample test on LI
    subject_li = df.groupby("subject")["lateralization_index"].mean()
    t_stat_li, p_val_li = stats.ttest_1samp(subject_li, 0)

    return {
        "df": df,
        "df_long": df_long,
        "mem_result": result,
        "subject_li": subject_li,
        "mean_li": subject_li.mean(),
        "t_stat_li": t_stat_li,
        "p_val_li": p_val_li,
        "hemisphere_effect": hem_effect,
        "hemisphere_pval": hem_pval,
        "left_grand_avg": left_grand_avg,
        "right_grand_avg": right_grand_avg,
        "left_sem": left_sem,
        "right_sem": right_sem,
    }


def plot_hemisphere_waveforms_comparison(results_dict, times):
    """
    Plot left vs right hemisphere waveforms for PRE, POST, ONLINE

    Parameters:
    -----------
    results_dict : dict
        {'pre': results_pre, 'post': results_post, 'online': results_online}
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    conditions = ["pre", "post", "online"]
    titles = ["PRE: Baseline", "POST: After Training", "ONLINE: Real-time Feedback"]

    for idx, (ax, condition, title) in enumerate(zip(axes, conditions, titles)):
        results = results_dict[condition]

        # Get waveform data from results
        left_avg = results["left_grand_avg"]
        right_avg = results["right_grand_avg"]
        left_sem = results["left_sem"]
        right_sem = results["right_sem"]

        # Plot
        ax.plot(
            times, left_avg, "b-", linewidth=2.5, label="Left Hemisphere", alpha=0.8
        )
        ax.fill_between(
            times, left_avg - left_sem, left_avg + left_sem, color="b", alpha=0.2
        )

        ax.plot(
            times, right_avg, "r-", linewidth=2.5, label="Right Hemisphere", alpha=0.8
        )
        ax.fill_between(
            times, right_avg - right_sem, right_avg + right_sem, color="r", alpha=0.2
        )

        # P300 window
        ax.axvspan(300, 600, alpha=0.1, color="yellow", label="P300 window")

        # Add statistics
        hem_pval = results["hemisphere_pval"]
        hem_effect = results["hemisphere_effect"]

        if hem_pval < 0.001:
            sig_str = "***"
        elif hem_pval < 0.01:
            sig_str = "**"
        elif hem_pval < 0.05:
            sig_str = "*"
        elif hem_pval < 0.10:
            sig_str = "†"  # Trend
        else:
            sig_str = "ns"

        ax.text(
            0.98,
            0.95,
            f"{sig_str}\nβ = {hem_effect:.3f} µV\np = {hem_pval:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=11,
        )

        ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
        ax.axvline(0, color="k", linestyle="--", linewidth=1)
        ax.set_ylabel("Amplitude (µV)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)

        if idx == 2:  # Last panel
            ax.set_xlabel("Time (ms)", fontsize=12)

    plt.suptitle(
        "Hemispheric P300 Comparison Across Conditions",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    return fig


def plot_hemisphere_effects_barplot(results_dict):
    """
    Bar plot showing hemisphere effect sizes across conditions with error bars.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    conditions = ["PRE", "POST", "ONLINE"]
    effects = []
    errors = []
    pvals = []

    for cond in ["pre", "post", "online"]:
        result = results_dict[cond]["mem_result"]
        beta = result.params["hemisphere[T.right]"]
        se = result.bse["hemisphere[T.right]"]
        pval = result.pvalues["hemisphere[T.right]"]

        effects.append(beta)
        errors.append(se)
        pvals.append(pval)

    # Create bar plot
    x = np.arange(len(conditions))
    colors = [
        "#d62728" if p < 0.05 else "#ff7f0e" if p < 0.10 else "#7f7f7f" for p in pvals
    ]

    bars = ax.bar(
        x,
        effects,
        yerr=errors,
        capsize=10,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.7,
    )

    # Add significance stars
    for i, (eff, err, pval) in enumerate(zip(effects, errors, pvals)):
        if pval < 0.001:
            sig = "***"
        elif pval < 0.01:
            sig = "**"
        elif pval < 0.05:
            sig = "*"
        elif pval < 0.10:
            sig = "†"
        else:
            sig = "ns"

        y_pos = eff + err + 0.1 if eff > 0 else eff - err - 0.1
        ax.text(i, y_pos, sig, ha="center", fontsize=16, fontweight="bold")

        # Add p-value below
        ax.text(i, -1.4, f"p={pval:.3f}", ha="center", fontsize=9)

    # Zero line
    ax.axhline(0, color="black", linestyle="-", linewidth=1.5)

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=12, fontweight="bold")
    ax.set_ylabel("Hemisphere Effect (Right - Left) [µV]", fontsize=12)
    ax.set_title(
        "Hemispheric Lateralization Across Conditions\n(Negative = Left-dominant)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(-1.6, 0.5)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#d62728", alpha=0.7, label="p < 0.05 (Significant)"),
        Patch(facecolor="#ff7f0e", alpha=0.7, label="0.05 ≤ p < 0.10 (Trend)"),
        Patch(facecolor="#7f7f7f", alpha=0.7, label="p ≥ 0.10 (Not significant)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    return fig


def plot_subject_hemisphere_comparison(results_dict):
    """
    Box plots showing left vs right for each condition with individual subjects.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    conditions = ["pre", "post", "online"]
    titles = ["PRE", "POST", "ONLINE"]

    for ax, cond, title in zip(axes, conditions, titles):
        results = results_dict[cond]
        df = results["df"]

        # Get subject-averaged data
        subject_left = df.groupby("subject")["left_p300"].mean()
        subject_right = df.groupby("subject")["right_p300"].mean()

        # Box plot
        bp = ax.boxplot(
            [subject_left, subject_right],
            labels=["Left", "Right"],
            patch_artist=True,
            widths=0.6,
        )
        bp["boxes"][0].set_facecolor("lightblue")
        bp["boxes"][1].set_facecolor("lightcoral")

        # Overlay individual subjects
        for left, right in zip(subject_left, subject_right):
            ax.plot(
                [1, 2],
                [left, right],
                "o-",
                color="gray",
                alpha=0.6,
                linewidth=1.5,
                markersize=8,
            )

        # Add significance
        pval = results["hemisphere_pval"]
        y_max = max(subject_left.max(), subject_right.max())

        ax.plot([1, 2], [y_max * 1.1, y_max * 1.1], "k-", linewidth=1.5)

        if pval < 0.001:
            sig_str = "***"
        elif pval < 0.01:
            sig_str = "**"
        elif pval < 0.05:
            sig_str = "*"
        elif pval < 0.10:
            sig_str = "†"
        else:
            sig_str = "ns"

        ax.text(1.5, y_max * 1.15, sig_str, ha="center", fontsize=18)
        ax.text(1.5, y_max * 1.25, f"p = {pval:.4f}", ha="center", fontsize=10)

        ax.set_ylabel("P300 Amplitude (µV)", fontsize=12)
        ax.set_title(
            f"{title}\n(n={len(subject_left)} subjects)", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Individual Subject Hemisphere Comparison", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    return fig
