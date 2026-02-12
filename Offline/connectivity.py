"""
© 2026 Alexandra Mikhael. All Rights Reserved.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, coherence as scipy_coherence
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# STEP 1: DEFINE CHANNEL CONFIGURATION
# ============================================================================


def setup_channels(labels):
    """
    Define channel groups and pairs for connectivity analysis.
    """
    # Convert labels to uppercase for matching
    labels_upper = [l.upper() for l in labels]

    # Find channel indices
    def get_idx(ch_name):
        try:
            return labels_upper.index(ch_name.upper())
        except ValueError:
            return None

    # Define channel groups
    frontal_channels = ["FZ", "F3", "F4", "FC1", "FC2"]
    central_channels = ["CZ", "C3", "C4"]
    parietal_channels = ["PZ", "P3", "P4", "CP1", "CP2"]

    frontal_idx = [get_idx(ch) for ch in frontal_channels if get_idx(ch) is not None]
    central_idx = [get_idx(ch) for ch in central_channels if get_idx(ch) is not None]
    parietal_idx = [get_idx(ch) for ch in parietal_channels if get_idx(ch) is not None]

    # Define key channel pairs for connectivity
    channel_pairs = {
        "frontal_parietal": [
            (get_idx("FZ"), get_idx("PZ")),
            (get_idx("F3"), get_idx("P3")),
            (get_idx("F4"), get_idx("P4")),
            (get_idx("FC1"), get_idx("CP1")),
            (get_idx("FC2"), get_idx("CP2")),
        ],
        "frontal_central": [
            (get_idx("FZ"), get_idx("CZ")),
            (get_idx("F3"), get_idx("C3")),
            (get_idx("F4"), get_idx("C4")),
        ],
        "central_parietal": [
            (get_idx("CZ"), get_idx("PZ")),
            (get_idx("C3"), get_idx("P3")),
            (get_idx("C4"), get_idx("P4")),
        ],
        "interhemispheric": [
            (get_idx("F3"), get_idx("F4")),
            (get_idx("C3"), get_idx("C4")),
            (get_idx("P3"), get_idx("P4")),
        ],
    }

    # Remove None pairs
    for key in channel_pairs:
        channel_pairs[key] = [
            (ch1, ch2)
            for ch1, ch2 in channel_pairs[key]
            if ch1 is not None and ch2 is not None
        ]

    return {
        "frontal": frontal_idx,
        "central": central_idx,
        "parietal": parietal_idx,
        "pairs": channel_pairs,
    }


# ============================================================================
# STEP 2: COMPUTE ALL CONNECTIVITY METRICS
# ============================================================================


def compute_plv_all_pairs(
    all_subjects_data,
    condition_key,
    times,
    freq_band,
    channel_pairs_dict,
    freq_band_definitions,
):
    """Compute PLV for all channel pair groups."""
    from scipy.signal import butter, filtfilt, hilbert

    fs = 512
    lowcut, highcut = freq_band_definitions[freq_band]
    nyq = 0.5 * fs
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype="band")

    all_results = []

    # Iterate through all channel pair groups
    for pair_type, pairs in channel_pairs_dict.items():
        for ch1, ch2 in pairs:
            for subj_id, subj_data in all_subjects_data.items():
                if condition_key not in subj_data:
                    continue

                epochs = subj_data[condition_key]
                n_trials = min(epochs.shape[2], 100)  # Limit for speed

                plv_trials = []

                for trial in range(n_trials):
                    try:
                        # Filter
                        sig1 = filtfilt(b, a, epochs[:, ch1, trial])
                        sig2 = filtfilt(b, a, epochs[:, ch2, trial])

                        # Hilbert
                        phase1 = np.angle(hilbert(sig1))
                        phase2 = np.angle(hilbert(sig2))

                        # PLV
                        plv = np.abs(np.mean(np.exp(1j * (phase1 - phase2))))
                        plv_trials.append(plv)
                    except:
                        continue

                if len(plv_trials) > 0:
                    all_results.append(
                        {
                            "subject": subj_id,
                            "pair_type": pair_type,
                            "ch1": ch1,
                            "ch2": ch2,
                            "plv": np.mean(plv_trials),
                            "plv_std": np.std(plv_trials),
                        }
                    )

    return pd.DataFrame(all_results)


def compute_wpli_all_pairs(
    all_subjects_data,
    condition_key,
    times,
    freq_band,
    channel_pairs_dict,
    freq_band_definitions,
):
    """Compute wPLI for all channel pair groups."""
    from scipy.signal import butter, filtfilt, hilbert

    fs = 512
    lowcut, highcut = freq_band_definitions[freq_band]
    nyq = 0.5 * fs
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype="band")

    all_results = []

    for pair_type, pairs in channel_pairs_dict.items():
        for ch1, ch2 in pairs:
            for subj_id, subj_data in all_subjects_data.items():
                if condition_key not in subj_data:
                    continue

                epochs = subj_data[condition_key]
                n_trials = min(epochs.shape[2], 100)

                wpli_trials = []

                for trial in range(n_trials):
                    try:
                        sig1 = filtfilt(b, a, epochs[:, ch1, trial])
                        sig2 = filtfilt(b, a, epochs[:, ch2, trial])

                        analytic1 = hilbert(sig1)
                        analytic2 = hilbert(sig2)

                        cross_spec = analytic1 * np.conj(analytic2)
                        imag_cross = np.imag(cross_spec)

                        numerator = np.abs(
                            np.mean(np.abs(imag_cross) * np.sign(imag_cross))
                        )
                        denominator = np.mean(np.abs(imag_cross))

                        wpli = numerator / (denominator + 1e-10)
                        wpli_trials.append(wpli)
                    except:
                        continue

                if len(wpli_trials) > 0:
                    all_results.append(
                        {
                            "subject": subj_id,
                            "pair_type": pair_type,
                            "ch1": ch1,
                            "ch2": ch2,
                            "wpli": np.mean(wpli_trials),
                        }
                    )

    return pd.DataFrame(all_results)


def compute_coherence_all_pairs(
    all_subjects_data,
    condition_key,
    times,
    freq_band,
    channel_pairs_dict,
    freq_band_definitions,
):
    """Compute coherence for all channel pair groups."""
    from scipy.signal import coherence as scipy_coherence

    fs = 512
    lowcut, highcut = freq_band_definitions[freq_band]

    all_results = []

    for pair_type, pairs in channel_pairs_dict.items():
        for ch1, ch2 in pairs:
            for subj_id, subj_data in all_subjects_data.items():
                if condition_key not in subj_data:
                    continue

                epochs = subj_data[condition_key]
                n_trials = min(epochs.shape[2], 100)

                coh_values = []

                for trial in range(n_trials):
                    try:
                        sig1 = epochs[:, ch1, trial]
                        sig2 = epochs[:, ch2, trial]

                        freqs, coh = scipy_coherence(sig1, sig2, fs=fs, nperseg=256)

                        # Extract band-specific coherence
                        band_mask = (freqs >= lowcut) & (freqs <= highcut)
                        band_coh = np.mean(coh[band_mask])
                        coh_values.append(band_coh)
                    except:
                        continue

                if len(coh_values) > 0:
                    all_results.append(
                        {
                            "subject": subj_id,
                            "pair_type": pair_type,
                            "ch1": ch1,
                            "ch2": ch2,
                            "coherence": np.mean(coh_values),
                        }
                    )

    return pd.DataFrame(all_results)


# ============================================================================
# STEP 3: STATISTICAL ANALYSIS WITH MIXED EFFECTS MODELS
# ============================================================================


def run_connectivity_statistics(
    results_pre, results_post, results_online, metric="plv", freq_band="alpha"
):
    """
    Run mixed effects models comparing connectivity across conditions.
    """
    # Combine results
    df_pre = results_pre[metric][freq_band].copy()
    df_pre["condition"] = "PRE"

    df_post = results_post[metric][freq_band].copy()
    df_post["condition"] = "POST"

    df_online = results_online[metric][freq_band].copy()
    df_online["condition"] = "ONLINE"

    df_combined = pd.concat([df_pre, df_post, df_online], ignore_index=True)

    df_combined["condition"] = pd.Categorical(
        df_combined["condition"],
        categories=["PRE", "POST", "ONLINE"],  # PRE first = reference
        ordered=False,
    )

    # Get metric column name
    metric_col = (
        metric if metric in df_combined.columns else list(df_combined.columns)[4]
    )

    print(f"\n{'='*70}")
    print(f"MIXED EFFECTS MODEL: {metric.upper()} - {freq_band.upper()} BAND")
    print(f"{'='*70}")

    # Run model for each pair type
    results_dict = {}

    for pair_type in df_combined["pair_type"].unique():
        df_pair = df_combined[df_combined["pair_type"] == pair_type]

        print(f"\n--- {pair_type.upper()} CONNECTIVITY ---")

        try:
            model = MixedLM.from_formula(
                f"{metric_col} ~ condition", data=df_pair, groups=df_pair["subject"]
            )
            result = model.fit(reml=True)
            print(result.summary())

            results_dict[pair_type] = result
        except Exception as e:
            print(f"Failed: {e}")
            continue

    return df_combined, results_dict


# ============================================================================
# STEP 4: VISUALIZATION
# ============================================================================


def create_connectivity_summary_figure(df_combined, metric="plv", freq_band="alpha"):
    """
    Create comprehensive connectivity visualization.
    """
    metric_col = (
        metric if metric in df_combined.columns else list(df_combined.columns)[4]
    )

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    pair_types = df_combined["pair_type"].unique()
    conditions = ["PRE", "POST", "ONLINE"]
    colors = ["lightblue", "lightcoral", "lightgreen"]

    # Plot 1-4: Box plots for each pair type
    for idx, pair_type in enumerate(pair_types[:4]):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])

        df_pair = df_combined[df_combined["pair_type"] == pair_type]

        data_for_box = [
            df_pair[df_pair["condition"] == c][metric_col].values for c in conditions
        ]

        bp = ax.boxplot(data_for_box, labels=conditions, patch_artist=True, widths=0.6)

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        # Add individual points
        for i, cond in enumerate(conditions):
            y = df_pair[df_pair["condition"] == cond][metric_col].values
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.4, s=30, color="black")

        ax.set_ylabel(f"{metric.upper()} ({freq_band})", fontsize=11)
        ax.set_title(
            f'{pair_type.replace("_", " ").title()}', fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="y")

    # Plot 5: Heatmap of connectivity by pair type and condition
    ax = fig.add_subplot(gs[2, :2])

    # Prepare data for heatmap
    heatmap_data = (
        df_combined.groupby(["pair_type", "condition"])[metric_col].mean().unstack()
    )

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": f"{metric.upper()}"},
    )
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Connection Type", fontsize=12)
    ax.set_title(
        f"{metric.upper()} Connectivity Heatmap ({freq_band} band)",
        fontsize=13,
        fontweight="bold",
    )

    # Plot 6: Summary statistics table
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")

    # Compute summary stats
    summary_data = []
    summary_data.append(["Pair Type", "PRE", "POST", "ONLINE", "Change"])

    for pair_type in pair_types:
        df_pair = df_combined[df_combined["pair_type"] == pair_type]

        pre_mean = df_pair[df_pair["condition"] == "PRE"][metric_col].mean()
        post_mean = df_pair[df_pair["condition"] == "POST"][metric_col].mean()
        online_mean = df_pair[df_pair["condition"] == "ONLINE"][metric_col].mean()

        change = ((post_mean - pre_mean) / pre_mean) * 100

        summary_data.append(
            [
                pair_type[:12],  # Truncate
                f"{pre_mean:.3f}",
                f"{post_mean:.3f}",
                f"{online_mean:.3f}",
                f"{change:+.1f}%",
            ]
        )

    table = ax.table(
        cellText=summary_data, cellLoc="center", loc="center", bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for j in range(5):
        table[(0, j)].set_facecolor("#4CAF50")
        table[(0, j)].set_text_props(weight="bold", color="white")

    ax.set_title("Summary Statistics", fontsize=12, fontweight="bold", pad=20)

    plt.suptitle(
        f"{metric.upper()} Connectivity Analysis: {freq_band.capitalize()} Band",
        fontsize=15,
        fontweight="bold",
    )

    return fig


def create_frequency_band_comparison(
    results_pre,
    results_post,
    results_online,
    metric="plv",
    pair_type="frontal_parietal",
):
    """
    Compare connectivity across frequency bands for a specific connection type.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    freq_bands = ["theta", "alpha", "beta"]
    conditions = ["PRE", "POST", "ONLINE"]
    colors = ["lightblue", "lightcoral", "lightgreen"]

    for ax, cond, color in zip(axes, conditions, colors):
        means = []
        sems = []

        for freq_band in freq_bands:
            if cond == "PRE":
                df = results_pre[metric][freq_band]
            elif cond == "POST":
                df = results_post[metric][freq_band]
            else:
                df = results_online[metric][freq_band]

            df_pair = df[df["pair_type"] == pair_type]
            metric_col = metric if metric in df.columns else list(df.columns)[4]

            means.append(df_pair[metric_col].mean())
            sems.append(df_pair[metric_col].sem())

        x = np.arange(len(freq_bands))
        ax.bar(
            x,
            means,
            yerr=sems,
            capsize=10,
            color=color,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.7,
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f.capitalize() for f in freq_bands], fontsize=11)
        ax.set_ylabel(f"{metric.upper()}", fontsize=12)
        ax.set_title(f"{cond}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f'{metric.upper()} Across Frequency Bands: {pair_type.replace("_", " ").title()}',
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig


# ============================================================================
# STEP 5: MAIN EXECUTION FUNCTION
# ============================================================================


def create_all_connectivity_figures(
    results_pre, results_post, results_online, stats_results, freq_bands
):
    """
    Create comprehensive figure set for all frequency bands.
    """
    figures = {}

    # Figure 1: Alpha band summary (main figure)
    print("  - Creating Alpha band summary...")
    figures["alpha_summary"] = create_connectivity_summary_figure(
        stats_results["plv"]["alpha"]["data"], "plv", "alpha"
    )

    # Figure 2: Theta band summary (meditation-relevant)
    print("  - Creating Theta band summary...")
    figures["theta_summary"] = create_connectivity_summary_figure(
        stats_results["plv"]["theta"]["data"], "plv", "theta"
    )

    # Figure 3: Gamma band summary (if available)
    if "gamma" in freq_bands:
        print("  - Creating Gamma band summary...")
        figures["gamma_summary"] = create_connectivity_summary_figure(
            stats_results["plv"]["gamma"]["data"], "plv", "gamma"
        )

    # Figure 4: All frequency bands comparison
    print("  - Creating frequency band comparison...")
    figures["all_bands_comparison"] = create_all_bands_comparison(
        results_pre, results_post, results_online, freq_bands
    )

    # Figure 5: Comprehensive heatmap across all bands
    print("  - Creating comprehensive heatmap...")
    figures["comprehensive_heatmap"] = create_comprehensive_heatmap(
        stats_results, freq_bands
    )

    # Figure 6 : Comprehensive heatmap across all bands per pair
    print(" - Creating comprehensive heatmap per channels pairs")
    # PLV by pair type
    figures["plv_heatmap_by_pair"] = create_comprehensive_heatmap_by_pair_type(
        stats_results, freq_bands, metric="plv"
    )

    # wPLI by pair type
    figures["wpli_heatmap_by_pair"] = create_comprehensive_heatmap_by_pair_type(
        stats_results, freq_bands, metric="wpli"
    )

    # Coherence by pair type
    figures["coherence_heatmap_by_pair"] = create_comprehensive_heatmap_by_pair_type(
        stats_results, freq_bands, metric="coherence"
    )

    # Figure 6: wPLI vs PLV comparison
    print("  - Creating metric comparison...")
    figures["metric_comparison"] = create_metric_comparison(stats_results, "alpha")

    return figures


def create_all_bands_comparison(results_pre, results_post, results_online, freq_bands):
    """
    Compare all frequency bands side-by-side.
    """
    fig, axes = plt.subplots(len(freq_bands), 3, figsize=(15, 4 * len(freq_bands)))

    # Handle single frequency band case
    if len(freq_bands) == 1:
        axes = axes.reshape(1, -1)

    conditions = ["PRE", "POST", "ONLINE"]
    pair_type = "frontal_parietal"  # Focus on main connection

    for row, freq_band in enumerate(freq_bands):
        for col, (cond, result) in enumerate(
            zip(conditions, [results_pre, results_post, results_online])
        ):
            ax = axes[row, col]

            # Get PLV data for this band
            df = result["plv"][freq_band]
            df_pair = df[df["pair_type"] == pair_type]

            # Plot distribution
            ax.hist(
                df_pair["plv"].values,
                bins=15,
                edgecolor="black",
                alpha=0.7,
                color="steelblue",
            )
            ax.axvline(
                df_pair["plv"].mean(),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f'Mean: {df_pair["plv"].mean():.3f}',
            )

            ax.set_xlabel("PLV", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(
                f"{freq_band.upper()} - {cond}", fontsize=11, fontweight="bold"
            )
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Frontal-Parietal Connectivity Across All Frequency Bands",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig


def create_comprehensive_heatmap(stats_results, freq_bands):
    """
    Heatmap showing connectivity across all bands and conditions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ["plv", "wpli", "coherence"]
    metric_names = ["PLV", "wPLI", "Coherence"]

    for ax, metric, metric_name in zip(axes, metrics, metric_names):
        # Prepare data: rows = freq bands, cols = conditions
        data_matrix = np.zeros((len(freq_bands), 3))

        for i, freq_band in enumerate(freq_bands):
            df = stats_results[metric][freq_band]["data"]

            # Get metric column name
            metric_col = metric if metric in df.columns else list(df.columns)[4]

            # Average across all pair types
            pre_mean = df[df["condition"] == "PRE"][metric_col].mean()
            post_mean = df[df["condition"] == "POST"][metric_col].mean()
            online_mean = df[df["condition"] == "ONLINE"][metric_col].mean()

            data_matrix[i, :] = [pre_mean, post_mean, online_mean]

        # Plot heatmap
        im = ax.imshow(data_matrix, cmap="YlOrRd", aspect="auto")

        # Set ticks
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["PRE", "POST", "ONLINE"], fontsize=11)
        ax.set_yticks(range(len(freq_bands)))
        ax.set_yticklabels([f.capitalize() for f in freq_bands], fontsize=11)

        # Add values
        for i in range(len(freq_bands)):
            for j in range(3):
                text = ax.text(
                    j,
                    i,
                    f"{data_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

        ax.set_title(f"{metric_name} Connectivity", fontsize=13, fontweight="bold")

        # Colorbar
        plt.colorbar(im, ax=ax, label=metric_name)

    plt.suptitle(
        "Connectivity Across All Frequency Bands and Conditions",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig


def create_comprehensive_heatmap_by_pair_type(stats_results, freq_bands, metric="plv"):
    """
    Heatmap showing connectivity for each pair type separately.
    """
    # Get all pair types
    sample_df = stats_results[metric][freq_bands[0]]["data"]
    pair_types = sample_df["pair_type"].unique()

    fig, axes = plt.subplots(len(pair_types), 1, figsize=(12, 4 * len(pair_types)))

    if len(pair_types) == 1:
        axes = [axes]

    for idx, pair_type in enumerate(pair_types):
        ax = axes[idx]

        # Prepare data: rows = freq bands, cols = conditions
        data_matrix = np.zeros((len(freq_bands), 3))

        for i, freq_band in enumerate(freq_bands):
            df = stats_results[metric][freq_band]["data"]
            df_pair = df[df["pair_type"] == pair_type]

            # Get metric column name
            metric_col = metric if metric in df.columns else list(df.columns)[4]

            # Average only this pair type
            pre_mean = df_pair[df_pair["condition"] == "PRE"][metric_col].mean()
            post_mean = df_pair[df_pair["condition"] == "POST"][metric_col].mean()
            online_mean = df_pair[df_pair["condition"] == "ONLINE"][metric_col].mean()

            data_matrix[i, :] = [pre_mean, post_mean, online_mean]

        # Plot heatmap
        im = ax.imshow(data_matrix, cmap="YlOrRd", aspect="auto")

        # Set ticks
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["PRE", "POST", "ONLINE"], fontsize=11)
        ax.set_yticks(range(len(freq_bands)))
        ax.set_yticklabels([f.capitalize() for f in freq_bands], fontsize=11)

        # Add values
        for i in range(len(freq_bands)):
            for j in range(3):
                text = ax.text(
                    j,
                    i,
                    f"{data_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

        ax.set_title(
            f'{metric.upper()} Connectivity: {pair_type.replace("_", " ").title()}',
            fontsize=13,
            fontweight="bold",
        )

        # Colorbar
        plt.colorbar(im, ax=ax, label=metric.upper())

    plt.suptitle(
        f"{metric.upper()} Connectivity by Connection Type",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig


def create_metric_comparison(stats_results, freq_band="alpha"):
    """
    Compare PLV, wPLI, and Coherence for a specific frequency band.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["plv", "wpli", "coherence"]
    metric_names = ["PLV", "wPLI", "Coherence"]
    conditions = ["PRE", "POST", "ONLINE"]
    colors = ["lightblue", "lightcoral", "lightgreen"]

    for ax, metric, metric_name in zip(axes, metrics, metric_names):
        df = stats_results[metric][freq_band]["data"]

        # Get metric column name
        metric_col = metric if metric in df.columns else list(df.columns)[4]

        # Average across all pair types
        means = []
        sems = []

        for cond in conditions:
            df_cond = df[df["condition"] == cond]
            means.append(df_cond[metric_col].mean())
            sems.append(df_cond[metric_col].sem())

        x = np.arange(len(conditions))
        bars = ax.bar(
            x,
            means,
            yerr=sems,
            capsize=10,
            color=colors,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.7,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=11)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f"{metric_name}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Connectivity Metrics Comparison: {freq_band.capitalize()} Band",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig


def save_all_results(figures, stats_results, freq_bands, output_dir):
    """
    Save all figures and data files.
    """
    # Save figures
    for name, fig in figures.items():
        filename = f"{output_dir}/{name}.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"  - Saved {name}.png")

    # Save data files
    for metric in ["plv", "wpli", "coherence"]:
        for freq_band in freq_bands:
            df = stats_results[metric][freq_band]["data"]
            filename = f"{output_dir}/{metric}_{freq_band}_data.csv"
            df.to_csv(filename, index=False)
            print(f"  - Saved {metric}_{freq_band}_data.csv")


def generate_comprehensive_report(stats_results, freq_bands, output_dir=None):
    """
    Generate a comprehensive text summary report of connectivity findings.
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("COMPREHENSIVE CONNECTIVITY ANALYSIS SUMMARY REPORT")
    report_lines.append("=" * 70)

    # Overall summary
    report_lines.append("\nFREQUENCY BANDS ANALYZED:")
    for freq_band in freq_bands:
        report_lines.append(f"  - {freq_band.capitalize()}")

    report_lines.append("\nMETRICS COMPUTED:")
    report_lines.append("  - PLV (Phase Locking Value)")
    report_lines.append("  - wPLI (Weighted Phase Lag Index)")
    report_lines.append("  - Coherence")

    # Detailed results for each metric and band
    for metric in ["plv", "wpli", "coherence"]:
        report_lines.append(f"\n{'='*70}")
        report_lines.append(f"{metric.upper()} RESULTS:")
        report_lines.append("=" * 70)

        for freq_band in freq_bands:
            report_lines.append(
                f"\n  {freq_band.capitalize()} Band ({metric.upper()}):"
            )
            report_lines.append("  " + "-" * 66)

            df = stats_results[metric][freq_band]["data"]
            metric_col = metric if metric in df.columns else list(df.columns)[4]

            for pair_type in df["pair_type"].unique():
                df_pair = df[df["pair_type"] == pair_type]

                pre_mean = df_pair[df_pair["condition"] == "PRE"][metric_col].mean()
                post_mean = df_pair[df_pair["condition"] == "POST"][metric_col].mean()
                online_mean = df_pair[df_pair["condition"] == "ONLINE"][
                    metric_col
                ].mean()

                change_pre_post = ((post_mean - pre_mean) / pre_mean) * 100
                change_pre_online = ((online_mean - pre_mean) / pre_mean) * 100

                report_lines.append(f"\n    {pair_type.replace('_', ' ').title()}:")
                report_lines.append(f"      PRE:    {pre_mean:.3f}")
                report_lines.append(
                    f"      POST:   {post_mean:.3f} ({change_pre_post:+.1f}%)"
                )
                report_lines.append(
                    f"      ONLINE: {online_mean:.3f} ({change_pre_online:+.1f}%)"
                )

    # Key findings summary
    report_lines.append(f"\n{'='*70}")
    report_lines.append("KEY FINDINGS SUMMARY:")
    report_lines.append("=" * 70)

    # Find largest increases/decreases
    max_increase = {"value": -float("inf"), "metric": "", "band": "", "pair": ""}
    max_decrease = {"value": float("inf"), "metric": "", "band": "", "pair": ""}

    for metric in ["plv", "wpli", "coherence"]:
        for freq_band in freq_bands:
            df = stats_results[metric][freq_band]["data"]
            metric_col = metric if metric in df.columns else list(df.columns)[4]

            for pair_type in df["pair_type"].unique():
                df_pair = df[df["pair_type"] == pair_type]

                pre_mean = df_pair[df_pair["condition"] == "PRE"][metric_col].mean()
                post_mean = df_pair[df_pair["condition"] == "POST"][metric_col].mean()

                change = ((post_mean - pre_mean) / pre_mean) * 100

                if change > max_increase["value"]:
                    max_increase = {
                        "value": change,
                        "metric": metric,
                        "band": freq_band,
                        "pair": pair_type,
                    }

                if change < max_decrease["value"]:
                    max_decrease = {
                        "value": change,
                        "metric": metric,
                        "band": freq_band,
                        "pair": pair_type,
                    }

    report_lines.append(f"\nLargest Increase (PRE to POST):")
    report_lines.append(f"  {max_increase['pair'].replace('_', ' ').title()}")
    report_lines.append(
        f"  {max_increase['band'].capitalize()} band - {max_increase['metric'].upper()}"
    )
    report_lines.append(f"  Change: +{max_increase['value']:.1f}%")

    report_lines.append(f"\nLargest Decrease (PRE to POST):")
    report_lines.append(f"  {max_decrease['pair'].replace('_', ' ').title()}")
    report_lines.append(
        f"  {max_decrease['band'].capitalize()} band - {max_decrease['metric'].upper()}"
    )
    report_lines.append(f"  Change: {max_decrease['value']:.1f}%")

    report_lines.append("\n" + "=" * 70)

    report_text = "\n".join(report_lines)
    print(report_text)

    if output_dir:
        with open(f"{output_dir}/connectivity_comprehensive_report.txt", "w") as f:
            f.write(report_text)
        print(
            f"\nComprehensive report saved to {output_dir}/connectivity_comprehensive_report.txt"
        )


def save_mem_results_to_text(stats_results, freq_bands, output_dir):
    """
    Save all MEM model results to organized text files.
    """
    import os

    # Create subdirectory for MEM results
    mem_dir = os.path.join(output_dir, "mem_results")
    if not os.path.exists(mem_dir):
        os.makedirs(mem_dir)

    print(f"\nSaving MEM results to {mem_dir}/...")

    for metric in ["plv", "wpli", "coherence"]:
        for freq_band in freq_bands:
            # Create filename
            filename = os.path.join(mem_dir, f"{metric}_{freq_band}_mem_results.txt")

            with open(filename, "w") as f:
                f.write("=" * 70 + "\n")
                f.write(f"MIXED EFFECTS MODEL RESULTS\n")
                f.write(f"Metric: {metric.upper()}\n")
                f.write(f"Frequency Band: {freq_band.upper()}\n")
                f.write("=" * 70 + "\n\n")

                # Get models for this metric and band
                models = stats_results[metric][freq_band]["models"]

                for pair_type, model_result in models.items():
                    f.write(f"\n{'='*70}\n")
                    f.write(f"{pair_type.upper().replace('_', ' ')} CONNECTIVITY\n")
                    f.write(f"{'='*70}\n\n")

                    # Write full model summary
                    f.write(str(model_result.summary()))
                    f.write("\n\n")

            print(f"  - Saved {metric}_{freq_band}_mem_results.txt")


def run_complete_connectivity_analysis_unfiltered(
    all_subjects_data,
    times,
    labels,
    save_results=True,
    output_dir="connectivity_results",
):
    """
    Connectivity analysis using UNFILTERED data stored in 'connectivity_*' keys.
    This allows analysis of ALL frequency bands including gamma.
    """
    import os

    output_dir = (
        "/home/alexandra-admin/Documents/Offline/offline_logs/connectivity_results"
    )
    os.makedirs(output_dir)

    print("=" * 70)
    print("CONNECTIVITY ANALYSIS USING UNFILTERED DATA")
    print("=" * 70)
    print("Using unfiltered epochs from 'connectivity_*' keys")
    print("Analyzing: Delta, Theta, Alpha, Beta, Gamma bands")
    print("=" * 70)

    # Step 1: Setup
    print("\n[1/6] Setting up channel configuration...")
    channel_config = setup_channels(labels)

    # Step 2: Compute connectivity for ALL frequency bands
    print("\n[2/6] Computing connectivity metrics for all conditions...")

    # NOW we can use ALL frequency bands!
    freq_bands = ["delta", "theta", "alpha", "beta", "gamma"]

    # Define ALL frequency bands (now possible with unfiltered data)
    freq_band_definitions = {
        "delta": (1, 4),  # Slow waves
        "theta": (4, 8),  # Meditation, working memory
        "alpha": (8, 13),  # Attention, relaxation
        "beta": (13, 30),  # Active thinking
        "gamma": (30, 50),  # High-level processing
    }

    print("\n  PRE condition:")
    results_pre = compute_all_connectivity_metrics_unfiltered(
        all_subjects_data,
        "connectivity_pre_target_all",
        times,
        channel_config,
        freq_bands,
        freq_band_definitions,
    )

    print("\n  POST condition:")
    results_post = compute_all_connectivity_metrics_unfiltered(
        all_subjects_data,
        "connectivity_post_target_all",
        times,
        channel_config,
        freq_bands,
        freq_band_definitions,
    )

    print("\n  ONLINE condition:")
    results_online = compute_all_connectivity_metrics_unfiltered(
        all_subjects_data,
        "connectivity_online_target_all",
        times,
        channel_config,
        freq_bands,
        freq_band_definitions,
    )

    # Step 3: Statistical analysis
    print("\n[3/6] Running statistical analyses...")
    stats_results = {}

    for metric in ["plv", "wpli", "coherence"]:
        stats_results[metric] = {}
        for freq_band in freq_bands:  # All bands!
            print(f"\n  Analyzing {metric.upper()} - {freq_band} band...")
            df_combined, model_results = run_connectivity_statistics(
                results_pre,
                results_post,
                results_online,
                metric=metric,
                freq_band=freq_band,
            )
            stats_results[metric][freq_band] = {
                "data": df_combined,
                "models": model_results,
            }

    # Step 4: Create comprehensive visualizations
    print("\n[4/6] Creating visualizations...")
    figures = create_all_connectivity_figures(
        results_pre, results_post, results_online, stats_results, freq_bands
    )

    # Step 5: Save everything
    if save_results:
        print(f"\n[5/6] Saving results to {output_dir}/...")
        save_all_results(figures, stats_results, freq_bands, output_dir)
        save_mem_results_to_text(stats_results, freq_bands, output_dir)

    # Step 6: Generate comprehensive report
    print("\n[6/6] Generating summary report...")
    generate_comprehensive_report(stats_results, freq_bands, output_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    
    print("=" * 70)

    return {
        "pre": results_pre,
        "post": results_post,
        "online": results_online,
        "statistics": stats_results,
        "figures": figures,
    }


def compute_all_connectivity_metrics_unfiltered(
    all_subjects_data,
    condition_key,
    times,
    channel_config,
    freq_bands,
    freq_band_definitions,
):
    """
    Compute connectivity metrics using UNFILTERED data.
    Now we can analyze ALL frequency bands properly!

    """
    results = {"plv": {}, "wpli": {}, "coherence": {}}

    for freq_band in freq_bands:
        lowcut, highcut = freq_band_definitions[freq_band]
        print(f"\n  Computing {freq_band} band ({lowcut}-{highcut} Hz)...")

        # PLV - NOW PASSES freq_band_definitions
        results["plv"][freq_band] = compute_plv_all_pairs(
            all_subjects_data,
            condition_key,
            times,
            freq_band,
            channel_config["pairs"],
            freq_band_definitions,  # PASS THIS
        )

        # wPLI - NOW PASSES freq_band_definitions
        results["wpli"][freq_band] = compute_wpli_all_pairs(
            all_subjects_data,
            condition_key,
            times,
            freq_band,
            channel_config["pairs"],
            freq_band_definitions,  # PASS THIS
        )

        # Coherence - NOW PASSES freq_band_definitions
        results["coherence"][freq_band] = compute_coherence_all_pairs(
            all_subjects_data,
            condition_key,
            times,
            freq_band,
            channel_config["pairs"],
            freq_band_definitions,  # PASS THIS
        )

    return results
