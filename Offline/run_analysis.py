import numpy as np
from scipy.stats import spearmanr
import matplotlib as plt
from CPP_Analysis import (
    compare_cpp_conditions,
    cpp_buildup_rate_analysis,
    cpp_latency_distribution_analysis,
    cpp_p300_relationship,
    detect_cpp_onset,
    cpp_onset_analysis,
    compare_cpp_slopes,
    comprehensive_cpp_report,
)
from sklearn.metrics import silhouette_score


def p3_metrics(
    t_ms,
    erp,
    p3_window=(300, 600),
    baseline_window=(-200, 0),
    area_mode="positive",
    snr_method="auto",
):
    """
    Compute P3 metrics at a single channel (e.g., Pz).

    Returns dict:
        peak_uv, latency_ms, fwhm_ms, area_uv_ms, snr, com_ms, lobe
    """
    t = np.asarray(t_ms).astype(float)
    x = np.asarray(erp).astype(float)

    w = (t >= p3_window[0]) & (t <= p3_window[1])
    b = (t >= baseline_window[0]) & (t <= baseline_window[1])

    if not np.any(w):
        raise ValueError("p3_window has no samples in t_ms")

    # dominant lobe (positive vs negative)
    xw = x[w]
    t_w = t[w]
    max_idx = np.argmax(xw)
    min_idx = np.argmin(xw)
    use_positive = abs(xw[max_idx]) >= abs(xw[min_idx])

    if use_positive:
        peak_amp = xw[max_idx]
        peak_lat = t_w[max_idx]
        half = 0.5 * peak_amp
        mask_half = x >= half
    else:
        peak_amp = xw[min_idx]
        peak_lat = t_w[min_idx]
        half = 0.5 * peak_amp
        mask_half = x <= half

    # --- FWHM with interpolation
    left_inds = np.where((t <= peak_lat) & mask_half)[0]
    right_inds = np.where((t >= peak_lat) & mask_half)[0]

    def interp_crossing(side_inds, ascend=True):
        if side_inds.size == 0:
            return np.nan
        i0 = side_inds[0] if ascend else side_inds[-1]
        i1 = i0 - 1 if ascend else i0 + 1
        if i1 < 0 or i1 >= len(t):
            return np.nan
        x0, x1 = x[i0], x[i1]
        t0, t1 = t[i0], t[i1]
        if x1 == x0:
            return t0
        frac = (half - x0) / (x1 - x0)
        return t0 + frac * (t1 - t0)

    left_t = interp_crossing(left_inds, ascend=True)
    right_t = interp_crossing(right_inds, ascend=False)
    fwhm = (
        (right_t - left_t) if np.isfinite(left_t) and np.isfinite(right_t) else np.nan
    )

    # --- Area
    xw_for_area = xw.copy()
    if area_mode == "positive":
        if use_positive:
            xw_for_area = np.clip(xw_for_area, 0, None)
        else:
            xw_for_area = -np.clip(-xw_for_area, 0, None)
    area = float(np.trapz(xw_for_area, t_w))

    # --- Center of mass
    xw_pos = np.clip(xw, 0, None) if use_positive else np.clip(-xw, 0, None)
    com_ms = (
        float(np.sum(t_w * xw_pos) / np.sum(xw_pos)) if np.sum(xw_pos) > 0 else np.nan
    )

    # --- Baseline noise
    if np.any(b):
        base = x[b]
        sd = np.std(base)
        if (snr_method == "auto" and sd == 0) or snr_method == "mad":
            mad = np.median(np.abs(base - np.median(base)))
            noise = 1.4826 * mad if mad > 0 else np.nan
        else:
            noise = sd if sd > 0 else np.nan
    else:
        noise = np.nan
    snr = float(peak_amp / noise) if np.isfinite(noise) and noise > 0 else np.nan

    return {
        "peak_uv": float(peak_amp),
        "latency_ms": float(peak_lat),
        "fwhm_ms": float(fwhm),
        "area_uv_ms": float(area),
        "snr": snr,
        "com_ms": com_ms,
        "lobe": "positive" if use_positive else "negative",
    }


def n2_p3_ptp(t_ms, erp, n2_w=(180, 300), p3_w=(300, 600)):
    """Peak-to-peak N2–P3 amplitude (µV)."""
    t = np.asarray(t_ms)
    x = np.asarray(erp)
    n2 = np.min(x[(t >= n2_w[0]) & (t <= n2_w[1])])
    p3 = np.max(x[(t >= p3_w[0]) & (t <= p3_w[1])])
    return float(p3 - n2)


def single_trial_latency_jitter(t_ms, X, p3_window=(300, 600)):
    """
    Compute single-trial latency jitter (SD in ms) and all per-trial latencies.

    Parameters
    ----------
    t_ms : array-like, shape (n_times,)
    X : array-like, shape (n_trials, n_times)
    """
    t = np.asarray(t_ms)
    X = np.asarray(X)
    w = (t >= p3_window[0]) & (t <= p3_window[1])
    peaks = t[w][np.argmax(X[:, w], axis=1)]
    return float(np.std(peaks)), peaks


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sourcelocal import run_source_localization_analysis


def trials_time_from_tc_tr(data_tc_tr, ch_idx):
    """data_tc_tr: (time, chan, trials) → returns X: (trials, time) at ch_idx"""
    return data_tc_tr[:, ch_idx, :].T


def sem(a, axis=0):
    a = np.asarray(a)
    ddof = 1 if a.shape[axis] > 1 else 0
    return np.std(a, axis=axis, ddof=ddof) / np.sqrt(max(a.shape[axis], 1))


def temporal_cluster_analysis(features, run_labels, trial_indices, n_clusters=None):
    """
    Cluster P300 trials and analyze temporal evolution

    Parameters:
        features: (n_trials, n_features) - all P300 trial features
        run_labels: (n_trials,) - which run each trial came from
        trial_indices: (n_trials,) - sequential trial number within run
        n_clusters: int or None - if None, finds optimal automatically

    Returns:
        cluster_labels: (n_trials,) cluster assignment
        cluster_info: dict with detailed analysis
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from collections import Counter

    # Standardize features (important for clustering!)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Find optimal clusters if not specified
    if n_clusters is None:
        print("\n=== Finding Optimal Number of Clusters ===")
        inertias = []
        silhouette_scores_list = []
        K_range = range(2, 7)  # Test 2-6 clusters

        for k in K_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_temp = kmeans_temp.fit_predict(features_scaled)

            inertias.append(kmeans_temp.inertia_)
            silhouette_scores_list.append(
                silhouette_score(features_scaled, labels_temp)
            )

        # Plot elbow curve
        fig_elbow, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Elbow method
        ax1.plot(list(K_range), inertias, "bo-", linewidth=2, markersize=8)
        ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
        ax1.set_ylabel("Inertia", fontsize=12)
        ax1.set_title("Elbow Method", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Silhouette score
        ax2.plot(
            list(K_range), silhouette_scores_list, "ro-", linewidth=2, markersize=8
        )
        ax2.set_xlabel("Number of Clusters (k)", fontsize=12)
        ax2.set_ylabel("Silhouette Score", fontsize=12)
        ax2.set_title(
            "Silhouette Score (Higher = Better)", fontsize=14, fontweight="bold"
        )
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.5, color="g", linestyle="--", alpha=0.5, label="Good threshold")
        ax2.legend()

        plt.tight_layout()
        plt.show()

        # Find optimal k (highest silhouette score)
        optimal_idx = np.argmax(silhouette_scores_list)
        n_clusters = list(K_range)[optimal_idx]

        print(f"\n{'='*60}")
        print(f"OPTIMAL NUMBER OF CLUSTERS")
        print(f"{'='*60}")
        print(f"Recommended k: {n_clusters}")
        print(f"Silhouette score: {silhouette_scores_list[optimal_idx]:.3f}")
        print(f"\nAll scores:")
        for k, inertia, sil in zip(K_range, inertias, silhouette_scores_list):
            marker = " ← BEST" if k == n_clusters else ""
            print(f"  k={k}: Inertia={inertia:.1f}, Silhouette={sil:.3f}{marker}")
        print(f"{'='*60}\n")

    # Perform clustering with optimal/specified k
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(features_scaled)

    print(f"\n{'='*60}")
    print(f"CLUSTERING RESULTS")
    print(f"{'='*60}")
    print(f"Total P300 trials: {len(features)}")
    print(f"Number of clusters: {n_clusters}")
    print(f"\nCluster sizes:")
    for cluster_id in range(n_clusters):
        count = np.sum(cluster_labels == cluster_id)
        pct = 100 * count / len(features)
        print(f"  Cluster {cluster_id}: {count} trials ({pct:.1f}%)")
    print(f"{'='*60}\n")

    # Analyze temporal distribution
    unique_runs = np.unique(run_labels)
    n_runs = len(unique_runs)

    # Create temporal evolution matrix
    cluster_by_run = np.zeros((n_runs, n_clusters))

    for run_idx, run_num in enumerate(unique_runs):
        run_mask = run_labels == run_num
        run_clusters = cluster_labels[run_mask]

        for cluster_id in range(n_clusters):
            cluster_by_run[run_idx, cluster_id] = np.sum(run_clusters == cluster_id)

    # Plot temporal evolution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Stacked bar chart - Cluster composition per run
    ax1 = axes[0, 0]
    bottom = np.zeros(n_runs)
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

    for cluster_id in range(n_clusters):
        ax1.bar(
            unique_runs,
            cluster_by_run[:, cluster_id],
            bottom=bottom,
            label=f"Cluster {cluster_id}",
            color=colors[cluster_id],
            alpha=0.8,
        )
        bottom += cluster_by_run[:, cluster_id]

    ax1.set_xlabel("Run Number", fontsize=12)
    ax1.set_ylabel("Number of Trials", fontsize=12)
    ax1.set_title("Cluster Composition by Run", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. Line plot - Cluster proportion over time
    ax2 = axes[0, 1]
    for cluster_id in range(n_clusters):
        proportions = cluster_by_run[:, cluster_id] / cluster_by_run.sum(axis=1)
        ax2.plot(
            unique_runs,
            proportions,
            "o-",
            linewidth=2,
            markersize=8,
            label=f"Cluster {cluster_id}",
            color=colors[cluster_id],
        )

    ax2.set_xlabel("Run Number", fontsize=12)
    ax2.set_ylabel("Proportion of Trials", fontsize=12)
    ax2.set_title("Cluster Proportion Evolution", fontsize=14, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # 3. Heatmap - Cluster distribution across runs
    ax3 = axes[1, 0]
    im = ax3.imshow(
        cluster_by_run.T, aspect="auto", cmap="YlOrRd", interpolation="nearest"
    )
    ax3.set_xlabel("Run Number", fontsize=12)
    ax3.set_ylabel("Cluster ID", fontsize=12)
    ax3.set_title("Cluster Heatmap (Counts)", fontsize=14, fontweight="bold")
    ax3.set_xticks(range(n_runs))
    ax3.set_xticklabels(unique_runs.astype(int))
    ax3.set_yticks(range(n_clusters))
    ax3.set_yticklabels([f"C{i}" for i in range(n_clusters)])
    plt.colorbar(im, ax=ax3, label="Trial Count")

    # 4. Cluster transitions - Do trials shift clusters over runs?
    ax4 = axes[1, 1]

    # Calculate "center of mass" for each cluster (average run number)
    cluster_centroids_time = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        avg_run = np.mean(run_labels[cluster_mask])
        cluster_centroids_time.append(avg_run)

    ax4.bar(range(n_clusters), cluster_centroids_time, color=colors, alpha=0.8)
    ax4.set_xlabel("Cluster ID", fontsize=12)
    ax4.set_ylabel("Average Run Number", fontsize=12)
    ax4.set_title("Temporal Center of Each Cluster", fontsize=14, fontweight="bold")
    ax4.set_xticks(range(n_clusters))
    ax4.set_xticklabels([f"C{i}" for i in range(n_clusters)])
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.axhline(
        y=np.mean(unique_runs),
        color="r",
        linestyle="--",
        linewidth=2,
        label="Overall average",
    )
    ax4.legend()

    plt.tight_layout()
    plt.show()

    # Statistical summary
    print(f"\n{'='*60}")
    print(f"TEMPORAL ANALYSIS")
    print(f"{'='*60}")
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_runs = run_labels[cluster_mask]

        print(f"\nCluster {cluster_id}:")
        print(f"  Average run: {np.mean(cluster_runs):.2f}")
        print(f"  Run range: {int(cluster_runs.min())}-{int(cluster_runs.max())}")
        print(f"  Most common run: {Counter(cluster_runs).most_common(1)[0][0]}")

        # Early vs late bias
        early_runs = [1, 2, 3]
        late_runs = [6, 7, 8]
        early_count = np.sum(np.isin(cluster_runs, early_runs))
        late_count = np.sum(np.isin(cluster_runs, late_runs))

        if early_count > late_count * 1.5:
            print(f"  → Early-run cluster (learning phase)")
        elif late_count > early_count * 1.5:
            print(f"  → Late-run cluster (expert phase)")
        else:
            print(f"  → Distributed across runs")

    print(f"{'='*60}\n")

    return cluster_labels, {
        "n_clusters": n_clusters,
        "cluster_by_run": cluster_by_run,
        "unique_runs": unique_runs,
        "kmeans": kmeans,
        "scaler": scaler,
    }


def compare_all_three_conditions(
    results_pre, results_post, results_online, save_dir=None
):
    """
    Compare Pre vs Post vs Online clustering patterns
    Creates comprehensive comparison plots showing learning and online performance
    """

    print("\n" + "=" * 70)
    print("THREE-WAY COMPARISON: PRE vs POST vs ONLINE")
    print("=" * 70)

    # Check which conditions are available
    conditions = {}
    if results_pre is not None:
        conditions["Pre"] = results_pre
    if results_post is not None:
        conditions["Post"] = results_post
    if results_online is not None:
        conditions["Online"] = results_online

    if len(conditions) < 2:
        print("⚠️ At least 2 conditions required for comparison")
        return

    # ============================================================
    # CREATE 2x3 COMPARISON FIGURE
    # ============================================================

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Color scheme
    colors_dict = {"Pre": "steelblue", "Post": "coral", "Online": "mediumseagreen"}

    # ============================================================
    # 1. OVERALL LATENCY COMPARISON (BOX PLOTS)
    # ============================================================

    ax1 = fig.add_subplot(gs[0, 0])

    latency_data = []
    condition_names = []
    condition_colors = []

    for cond_name, result in conditions.items():
        latency_data.append(result["latencies_positive"])
        condition_names.append(cond_name)
        condition_colors.append(colors_dict[cond_name])

    # Box plots
    bp = ax1.boxplot(
        latency_data,
        labels=condition_names,
        widths=0.5,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=8),
    )

    # Color boxes
    for patch, color in zip(bp["boxes"], condition_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add individual points
    for i, (lats, color) in enumerate(zip(latency_data, condition_colors)):
        x = np.random.normal(i + 1, 0.04, size=len(lats))
        ax1.scatter(x, lats, alpha=0.3, s=15, color=color)

    ax1.set_ylabel("P300 Latency (ms)", fontsize=12)
    ax1.set_title("Overall P300 Latency Comparison", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add statistics
    means = [np.mean(lats) for lats in latency_data]
    stds = [np.std(lats) for lats in latency_data]

    stats_text = "\n".join(
        [
            f"{name}: {mean:.1f}±{std:.1f} ms"
            for name, mean, std in zip(condition_names, means, stds)
        ]
    )
    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=9,
    )

    # ============================================================
    # 2. CLUSTER DISTRIBUTION COMPARISON (STACKED BAR)
    # ============================================================

    ax2 = fig.add_subplot(gs[0, 1])

    # Get max number of clusters across conditions
    max_clusters = max([result["optimal_k"] for result in conditions.values()])

    # Prepare data for stacked bar
    cluster_props = {cond: [] for cond in condition_names}
    cluster_labels_all = []

    for cluster_num in range(max_clusters):
        if cluster_num == 0:
            cluster_labels_all.append("FAST")
        elif cluster_num == max_clusters - 1:
            cluster_labels_all.append("MEDIUM")
        else:
            cluster_labels_all.append("SLOW")

        for cond_name, result in conditions.items():
            clusters = result["cluster_labels"]
            cluster_names = result["cluster_names"]

            # Find which cluster ID corresponds to this position
            sorted_ids = sorted(
                set(clusters),
                key=lambda x: result["latencies_positive"][clusters == x].mean(),
            )

            if cluster_num < len(sorted_ids):
                cid = sorted_ids[cluster_num]
                prop = 100 * np.sum(clusters == cid) / len(clusters)
                cluster_props[cond_name].append(prop)
            else:
                cluster_props[cond_name].append(0)

    # Create stacked bar
    x = np.arange(len(condition_names))
    width = 0.6
    bottom = np.zeros(len(condition_names))

    colors_clusters = plt.cm.RdYlBu_r(np.linspace(0, 1, max_clusters))

    for cluster_idx, cluster_label in enumerate(cluster_labels_all):
        heights = [cluster_props[cond][cluster_idx] for cond in condition_names]
        ax2.bar(
            x,
            heights,
            width,
            bottom=bottom,
            label=cluster_label,
            color=colors_clusters[cluster_idx],
            alpha=0.8,
        )

        # Add percentage labels
        for i, (height, cond) in enumerate(zip(heights, condition_names)):
            if height > 5:  # Only show if > 5%
                ax2.text(
                    i,
                    bottom[i] + height / 2,
                    f"{height:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

        bottom += heights

    ax2.set_ylabel("Percentage of Trials (%)", fontsize=12)
    ax2.set_title("Cluster Distribution", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(condition_names)
    ax2.set_ylim([0, 100])
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    # ============================================================
    # 3. AMPLITUDE COMPARISON
    # ============================================================

    ax3 = fig.add_subplot(gs[0, 2])

    amplitude_data = []
    for result in conditions.values():
        amplitude_data.append(result["amplitudes_positive"])

    # Box plots
    bp_amp = ax3.boxplot(
        amplitude_data,
        labels=condition_names,
        widths=0.5,
        patch_artist=True,
        showmeans=True,
    )

    # Color boxes
    for patch, color in zip(bp_amp["boxes"], condition_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add individual points
    for i, (amps, color) in enumerate(zip(amplitude_data, condition_colors)):
        x = np.random.normal(i + 1, 0.04, size=len(amps))
        ax3.scatter(x, amps, alpha=0.3, s=15, color=color)

    ax3.set_ylabel("P300 Amplitude (µV)", fontsize=12)
    ax3.set_title("P300 Amplitude Comparison", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    # ============================================================
    # 4. LATENCY BY CLUSTER (GROUPED BAR CHART)
    # ============================================================

    ax4 = fig.add_subplot(gs[1, 0])

    # Get latencies for each cluster type (FAST, MEDIUM, SLOW)
    cluster_types = ["FAST", "MEDIUM", "SLOW"]

    cluster_latencies = {ctype: [] for ctype in cluster_types}

    for cond_name, result in conditions.items():
        clusters = result["cluster_labels"]
        lats = result["latencies_positive"]
        cluster_names_dict = result["cluster_names"]

        # Sort clusters by latency
        sorted_ids = sorted(set(clusters), key=lambda x: lats[clusters == x].mean())

        # Map to cluster types
        for i, cid in enumerate(sorted_ids):
            cluster_mask = clusters == cid
            mean_lat = lats[cluster_mask].mean()

            if i == 0:
                cluster_latencies["FAST"].append(mean_lat)
            elif i == len(sorted_ids) - 1:
                cluster_latencies["SLOW"].append(mean_lat)
            else:
                cluster_latencies["MEDIUM"].append(mean_lat)

        # Fill missing cluster types
        if len(sorted_ids) == 2:
            cluster_latencies["MEDIUM"].append(np.nan)

    # Grouped bar chart
    x = np.arange(len(cluster_types))
    width = 0.25

    for i, (cond_name, color) in enumerate(zip(condition_names, condition_colors)):
        heights = [
            cluster_latencies[ctype][i] if i < len(cluster_latencies[ctype]) else np.nan
            for ctype in cluster_types
        ]
        offset = width * (i - len(condition_names) / 2 + 0.5)
        ax4.bar(x + offset, heights, width, label=cond_name, color=color, alpha=0.7)

    ax4.set_ylabel("Mean Latency (ms)", fontsize=12)
    ax4.set_title("Mean Latency by Cluster Type", fontsize=13, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(cluster_types)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    # ============================================================
    # 5. POSITIVE vs NEGATIVE P300 PROPORTION
    # ============================================================

    ax5 = fig.add_subplot(gs[1, 1])

    positive_props = []
    negative_props = []

    for result in conditions.values():
        pos_prop = 100 * result["n_positive"] / result["n_trials"]
        neg_prop = 100 * result["n_negative"] / result["n_trials"]
        positive_props.append(pos_prop)
        negative_props.append(neg_prop)

    x = np.arange(len(condition_names))
    width = 0.6

    bars1 = ax5.bar(
        x, positive_props, width, label="Positive P300", color="lightgreen", alpha=0.7
    )
    bars2 = ax5.bar(
        x,
        negative_props,
        width,
        bottom=positive_props,
        label="Negative/Artifact",
        color="lightcoral",
        alpha=0.7,
    )

    # Add labels
    for i, (pos, neg) in enumerate(zip(positive_props, negative_props)):
        ax5.text(
            i,
            pos / 2,
            f"{pos:.0f}%",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        ax5.text(
            i,
            pos + neg / 2,
            f"{neg:.0f}%",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    ax5.set_ylabel("Percentage of Trials (%)", fontsize=12)
    ax5.set_title("Trial Quality Distribution", fontsize=13, fontweight="bold")
    ax5.set_xticks(x)
    ax5.set_xticklabels(condition_names)
    ax5.set_ylim([0, 100])
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis="y")

    # ============================================================
    # 6. STATISTICAL SUMMARY TABLE
    # ============================================================

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    # Create summary table
    table_data = []
    table_data.append(["Metric", "Pre", "Post", "Online"])

    # Total trials
    row = ["Total Trials"]
    for cond_name in ["Pre", "Post", "Online"]:
        if cond_name in conditions:
            row.append(str(conditions[cond_name]["n_trials"]))
        else:
            row.append("-")
    table_data.append(row)

    # Positive P300s
    row = ["Positive P300s"]
    for cond_name in ["Pre", "Post", "Online"]:
        if cond_name in conditions:
            n_pos = conditions[cond_name]["n_positive"]
            n_tot = conditions[cond_name]["n_trials"]
            row.append(f"{n_pos} ({100*n_pos/n_tot:.0f}%)")
        else:
            row.append("-")
    table_data.append(row)

    # Mean latency
    row = ["Mean Latency (ms)"]
    for cond_name in ["Pre", "Post", "Online"]:
        if cond_name in conditions:
            lat = conditions[cond_name]["latencies_positive"].mean()
            row.append(f"{lat:.1f}")
        else:
            row.append("-")
    table_data.append(row)

    # SD latency
    row = ["SD Latency (ms)"]
    for cond_name in ["Pre", "Post", "Online"]:
        if cond_name in conditions:
            lat_sd = conditions[cond_name]["latencies_positive"].std()
            row.append(f"{lat_sd:.1f}")
        else:
            row.append("-")
    table_data.append(row)

    # Mean amplitude
    row = ["Mean Amplitude (µV)"]
    for cond_name in ["Pre", "Post", "Online"]:
        if cond_name in conditions:
            amp = conditions[cond_name]["amplitudes_positive"].mean()
            row.append(f"{amp:.2f}")
        else:
            row.append("-")
    table_data.append(row)

    # Number of clusters
    row = ["# Clusters"]
    for cond_name in ["Pre", "Post", "Online"]:
        if cond_name in conditions:
            row.append(str(conditions[cond_name]["optimal_k"]))
        else:
            row.append("-")
    table_data.append(row)

    # Create table
    table = ax6.table(
        cellText=table_data, cellLoc="center", loc="center", bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style metric column
    for i in range(1, len(table_data)):
        table[(i, 0)].set_facecolor("#E7E6E6")
        table[(i, 0)].set_text_props(weight="bold")

    # Color condition columns
    for i in range(1, len(table_data)):
        if "Pre" in conditions:
            table[(i, 1)].set_facecolor("#D6E4F5")
        if "Post" in conditions:
            table[(i, 2)].set_facecolor("#FCE4D6")
        if "Online" in conditions:
            table[(i, 3)].set_facecolor("#D5E8D4")

    ax6.set_title("Summary Statistics", fontsize=13, fontweight="bold", pad=20)

    # ============================================================
    # SAVE FIGURE
    # ============================================================

    plt.suptitle(
        "Pre vs Post vs Online: Comprehensive Comparison",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )

    if save_dir:
        fig_path = os.path.join(save_dir, "comparison_all_three_conditions.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {fig_path}")

    plt.show()
    plt.close(fig)

    # ============================================================
    # PRINT STATISTICAL COMPARISONS
    # ============================================================

    print(f"\n{'='*70}")
    print("PAIRWISE STATISTICAL COMPARISONS")
    print(f"{'='*70}")

    from scipy.stats import ttest_ind

    # All pairwise comparisons
    pairs = [("Pre", "Post"), ("Pre", "Online"), ("Post", "Online")]

    for cond1, cond2 in pairs:
        if cond1 in conditions and cond2 in conditions:
            print(f"\n{cond1} vs {cond2}:")

            lats1 = conditions[cond1]["latencies_positive"]
            lats2 = conditions[cond2]["latencies_positive"]

            mean1 = lats1.mean()
            mean2 = lats2.mean()
            diff = mean1 - mean2

            t_stat, p_val = ttest_ind(lats1, lats2)

            print(f"  Latency:")
            print(f"    {cond1}: {mean1:.1f} ms")
            print(f"    {cond2}: {mean2:.1f} ms")
            print(f"    Difference: {diff:+.1f} ms ({100*diff/mean1:+.1f}%)")
            print(f"    t-test: t={t_stat:.3f}, p={p_val:.4f}")

            if p_val < 0.001:
                print(f"    Significance: *** (p < 0.001)")
            elif p_val < 0.01:
                print(f"    Significance: ** (p < 0.01)")
            elif p_val < 0.05:
                print(f"    Significance: * (p < 0.05)")
            else:
                print(f"    Significance: ns (not significant)")

    print(f"\n{'='*70}")

    # ============================================================
    # LEARNING PROGRESSION SUMMARY
    # ============================================================

    if "Pre" in conditions and "Post" in conditions:
        print(f"\n{'='*70}")
        print("LEARNING EFFECT (PRE → POST)")
        print(f"{'='*70}")

        pre_fast = np.sum(
            conditions["Pre"]["cluster_labels"]
            == sorted(
                set(conditions["Pre"]["cluster_labels"]),
                key=lambda x: conditions["Pre"]["latencies_positive"][
                    conditions["Pre"]["cluster_labels"] == x
                ].mean(),
            )[0]
        )
        pre_fast_pct = 100 * pre_fast / len(conditions["Pre"]["cluster_labels"])

        post_fast = np.sum(
            conditions["Post"]["cluster_labels"]
            == sorted(
                set(conditions["Post"]["cluster_labels"]),
                key=lambda x: conditions["Post"]["latencies_positive"][
                    conditions["Post"]["cluster_labels"] == x
                ].mean(),
            )[0]
        )
        post_fast_pct = 100 * post_fast / len(conditions["Post"]["cluster_labels"])

        print(f"\nFAST cluster proportion:")
        print(f"  Pre → Post: {pre_fast_pct:.1f}% → {post_fast_pct:.1f}%")
        print(f"  Change: {post_fast_pct - pre_fast_pct:+.1f} percentage points")

        if post_fast_pct > pre_fast_pct * 1.2:
            print(f"  → ✓ FAST responses INCREASED with training")

        # SLOW cluster
        pre_slow = np.sum(
            conditions["Pre"]["cluster_labels"]
            == sorted(
                set(conditions["Pre"]["cluster_labels"]),
                key=lambda x: conditions["Pre"]["latencies_positive"][
                    conditions["Pre"]["cluster_labels"] == x
                ].mean(),
            )[-1]
        )
        pre_slow_pct = 100 * pre_slow / len(conditions["Pre"]["cluster_labels"])

        post_slow = np.sum(
            conditions["Post"]["cluster_labels"]
            == sorted(
                set(conditions["Post"]["cluster_labels"]),
                key=lambda x: conditions["Post"]["latencies_positive"][
                    conditions["Post"]["cluster_labels"] == x
                ].mean(),
            )[-1]
        )
        post_slow_pct = 100 * post_slow / len(conditions["Post"]["cluster_labels"])

        print(f"\nSLOW cluster proportion:")
        print(f"  Pre → Post: {pre_slow_pct:.1f}% → {post_slow_pct:.1f}%")
        print(f"  Change: {post_slow_pct - pre_slow_pct:+.1f} percentage points")

        if post_slow_pct < pre_slow_pct * 0.5:
            print(f"  → ✓ SLOW responses DECREASED with training")

        print(f"\n{'='*70}")


def cluster_latency_and_plot(
    t_ms,
    X,
    title,
    fast_window=(250, 400),
    slow_window=(420, 620),
    n_clusters=2,
    random_state=0,
    show=True,
):
    """
    X: (trials, time) at one channel.
    Returns: dict with jitter_sd, per_trial_lat, labels, centers, n_per_cluster
    and the cluster ERPs (erp0, erp1, sem0, sem1).
    """
    # latencies & jitter
    w = (t_ms >= 300) & (t_ms <= 600)
    per_trial_lat = t_ms[w][np.argmax(X[:, w], axis=1)]
    jitter_sd = float(np.std(per_trial_lat))

    # cluster on latency
    lat = per_trial_lat.reshape(-1, 1)
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state).fit(lat)
    labels = km.labels_
    centers = sorted(km.cluster_centers_.ravel())
    # cluster splits
    X0, X1 = X[labels == 0], X[labels == 1]
    erp0, erp1 = X0.mean(axis=0), X1.mean(axis=0)
    sem0 = sem(X0, axis=0) if len(X0) > 1 else np.zeros_like(erp0)
    sem1 = sem(X1, axis=0) if len(X1) > 1 else np.zeros_like(erp1)

    if show:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t_ms, erp0, label=f"Cluster 0 (n={len(X0)})")
        ax.fill_between(t_ms, erp0 - sem0, erp0 + sem0, alpha=0.2)
        ax.plot(t_ms, erp1, label=f"Cluster 1 (n={len(X1)})")
        ax.fill_between(t_ms, erp1 - sem1, erp1 + sem1, alpha=0.2)
        ax.axvspan(
            fast_window[0],
            fast_window[1],
            alpha=0.10,
            label=f"FAST window ({fast_window[0]}–{fast_window[1]} ms)",
        )
        ax.axvspan(
            slow_window[0],
            slow_window[1],
            alpha=0.10,
            label=f"SLOW window ({slow_window[0]}–{slow_window[1]} ms)",
        )
        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {
        "jitter_sd": jitter_sd,
        "per_trial_lat": per_trial_lat,
        "labels": labels,
        "centers_ms": centers,
        "n0": int((labels == 0).sum()),
        "n1": int((labels == 1).sum()),
        "erp0": erp0,
        "erp1": erp1,
        "sem0": sem0,
        "sem1": sem1,
    }


from scipy.signal import correlate


def shift_no_wrap(x, shift, pad=0.0):
    """Shift 1D array by integer samples; pad with `pad` instead of wrap."""
    y = np.full_like(x, pad)
    if shift > 0:
        y[shift:] = x[:-shift]
    elif shift < 0:
        y[:shift] = x[-shift:]
    else:
        y[:] = x
    return y


def woody_align_window(
    t, X, window=(300, 600), max_shift_ms=80, pad=0.0, rebaseline=(-200, 0)
):
    """
    X: (trials × time). Aligns only using the window; shifts the WHOLE trial with no wrap.
    Then re-baselines to keep amplitude comparable.
    """
    t = np.asarray(t)
    X = np.asarray(X)
    dt = t[1] - t[0]
    w = (t >= window[0]) & (t <= window[1])
    ref = np.nanmean(X, axis=0)  # reference ERP
    ref_w = ref[w] - np.mean(ref[(t >= rebaseline[0]) & (t <= rebaseline[1])])

    max_shift = int(round(max_shift_ms / dt))
    shifts = np.zeros(X.shape[0], dtype=int)
    X_aligned = np.empty_like(X)

    for i in range(X.shape[0]):
        xi = X[i].copy()
        # re-baseline first
        base_mask = (t >= rebaseline[0]) & (t <= rebaseline[1])
        xi -= np.mean(xi[base_mask])

        # xcorr inside the window
        c = correlate(xi[w], ref_w, mode="full")
        lags = np.arange(-len(ref_w) + 1, len(ref_w))
        m = (lags >= -max_shift) & (lags <= max_shift)
        best_lag = lags[m][np.argmax(c[m])]
        shifts[i] = best_lag

        # shift entire trial with zero-padding (no wrap)
        X_aligned[i] = shift_no_wrap(xi, best_lag, pad=0.0)

        # re-baseline again (keeps comparability after shift)
        X_aligned[i] -= np.mean(X_aligned[i][base_mask])

    return X_aligned, shifts * dt


from Visualize_ERP import (
    time_frequency_analysis,
    difference_wave_analysis,
    latency_distribution_analysis,
    cluster_based_comparison,
    n200_p300_temporal_relationship,
    hemispheric_lateralization_analysis,
    analyze_n200_p300_correlation,
    create_n200_p300_comparison_summary,
    identify_early_components,
)

import os


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
        nt, fz, pz, cz, cpz, c1, c2 = 9, 6, 26, 15, 20, 20, 23

    prompt = "CPP[1] or P300[2] or Target[3] or Advanced[4] or Cluster[5/6] or Source[7] analysis: "
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
            try:
                lat_fig, lat_stats = cpp_latency_distribution_analysis(
                    nop300_pre, time_ms, cpp_channels, window=(200, 500)
                )
                save_path = os.path.join(output_dir, "cpp_latency_pre.png")
                lat_fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"✅ Saved: cpp_latency_pre.png")
                print(
                    f"   Mean latency: {lat_stats['mean_lat']:.1f} ± {lat_stats['std_lat']:.1f} ms"
                )
                plt.close(lat_fig)
            except Exception as e:
                print(f"❌ Error in CPP latency analysis: {e}")

            # === 3. CPP Latency Distribution (POST) ===
            print("\n=== CPP Latency Distribution (Post) ===")
            try:
                lat_fig, lat_stats = cpp_latency_distribution_analysis(
                    nop300_post, time_ms, cpp_channels, window=(200, 500)
                )
                save_path = os.path.join(output_dir, "cpp_latency_post.png")
                lat_fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"✅ Saved: cpp_latency_post.png")
                print(
                    f"   Mean latency: {lat_stats['mean_lat']:.1f} ± {lat_stats['std_lat']:.1f} ms"
                )
                plt.close(lat_fig)
            except Exception as e:
                print(f"❌ Error in CPP latency analysis: {e}")

            # === 4. CPP Buildup Rate (PRE) ===
            print("\n=== CPP Buildup Rate Analysis (Pre) ===")
            try:
                buildup_fig, buildup_stats = cpp_buildup_rate_analysis(
                    nop300_pre,
                    time_ms,
                    cpp_channels,
                    baseline_window=(-200, 0),
                    buildup_window=(0, 500),
                )
                save_path = os.path.join(output_dir, "cpp_buildup_pre.png")
                buildup_fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"✅ Saved: cpp_buildup_pre.png")
                print(
                    f"   Mean buildup rate: {buildup_stats['mean_slope']:.2f} ± {buildup_stats['std_slope']:.2f} µV/s"
                )
                plt.close(buildup_fig)
            except Exception as e:
                print(f"❌ Error in CPP buildup analysis: {e}")

                # === CPP ONSET ANALYSIS ===
            print("\n=== CPP Onset Detection ===")
            for name, data in [("Pre", nop300_pre), ("Post", nop300_post)]:
                try:
                    onset_fig, onset_stats = cpp_onset_analysis(
                        data,
                        time_ms,
                        cpp_channels,
                        baseline_window=(-200, 0),
                        search_window=(0, 400),
                    )
                    save_path = os.path.join(
                        output_dir, f"cpp_onset_{name.lower()}.png"
                    )
                    onset_fig.savefig(save_path, dpi=150, bbox_inches="tight")
                    print(f"✅ Saved: cpp_onset_{name.lower()}.png")
                    print(
                        f"   {name} CPP onset: {onset_stats['mean_onset']:.1f} ± {onset_stats['std_onset']:.1f} ms"
                    )
                    plt.close(onset_fig)
                except Exception as e:
                    print(f"❌ Error in {name} onset analysis: {e}")

            # === CPP BUILDUP RATE ===
            print("\n=== CPP Buildup Rate Analysis ===")
            for name, data in [("Pre", nop300_pre), ("Post", nop300_post)]:
                try:
                    buildup_fig, buildup_stats = cpp_buildup_rate_analysis(
                        data,
                        time_ms,
                        cpp_channels,
                        baseline_window=(-200, 0),
                        buildup_window=(0, 500),
                    )
                    save_path = os.path.join(
                        output_dir, f"cpp_buildup_{name.lower()}.png"
                    )
                    buildup_fig.savefig(save_path, dpi=150, bbox_inches="tight")
                    print(f"✅ Saved: cpp_buildup_{name.lower()}.png")
                    print(
                        f"   {name} buildup rate: {buildup_stats['mean_slope']:.2f} ± {buildup_stats['std_slope']:.2f} µV/s"
                    )
                    plt.close(buildup_fig)
                except Exception as e:
                    print(f"❌ Error in {name} buildup analysis: {e}")

            # === CPP-P300 RELATIONSHIP ===
            print("\n=== CPP-P300 Relationship Analysis ===")
            for name, cpp_data, p300_data in [
                ("Pre", nop300_pre, p300_pre),
                ("Post", nop300_post, p300_post),
            ]:
                try:
                    rel_fig, rel_stats = cpp_p300_relationship(
                        cpp_data,
                        p300_data,
                        time_ms,
                        cpp_channels,
                        pz,
                        cpp_window=(200, 500),
                        p300_window=(300, 600),
                    )
                    save_path = os.path.join(
                        output_dir, f"cpp_p300_relationship_{name.lower()}.png"
                    )
                    rel_fig.savefig(save_path, dpi=150, bbox_inches="tight")
                    print(f"✅ Saved: cpp_p300_relationship_{name.lower()}.png")
                    print(
                        f"   {name} CPP Amp → P300 Amp: r = {rel_stats['corr_amp_amp']:.3f}, p = {rel_stats['p_amp_amp']:.4f}"
                    )
                    plt.close(rel_fig)
                except Exception as e:
                    print(f"❌ Error in {name} CPP-P300 analysis: {e}")

        elif comparison == 1:
            # === WITH ONLINE DATA ===
            print("\n=== CPP Comparison: Pre vs Post vs Online ===")
            try:
                comp_fig = compare_cpp_conditions(
                    nop300_pre, nop300_post, nop300_online, time_ms, cpp_channels
                )
                save_path = os.path.join(output_dir, "cpp_three_way_comparison.png")
                comp_fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"✅ Saved: cpp_three_way_comparison.png")
                plt.close(comp_fig)
            except Exception as e:
                print(f"❌ Error in CPP comparison: {e}")

            # Latency analysis for all three
            for name, data in [
                ("Pre", nop300_pre),
                ("Post", nop300_post),
                ("Online", nop300_online),
            ]:
                print(f"\n=== CPP Latency Distribution ({name}) ===")
                try:
                    lat_fig, lat_stats = cpp_latency_distribution_analysis(
                        data, time_ms, cpp_channels, window=(200, 500)
                    )
                    save_path = os.path.join(
                        output_dir, f"cpp_latency_{name.lower()}.png"
                    )
                    lat_fig.savefig(save_path, dpi=150, bbox_inches="tight")
                    print(f"✅ Saved: cpp_latency_{name.lower()}.png")
                    print(
                        f"   Mean latency: {lat_stats['mean_lat']:.1f} ± {lat_stats['std_lat']:.1f} ms"
                    )
                    plt.close(lat_fig)
                except Exception as e:
                    print(f"❌ Error: {e}")

                    # 1. Slope comparison
            print("\n--- CPP Slope Comparison ---")
            try:
                slope_fig, slope_stats = compare_cpp_slopes(
                    nop300_pre,
                    nop300_post,
                    nop300_online,
                    time_ms,
                    cpp_channels,
                    buildup_window=(0, 500),
                )
                save_path = os.path.join(output_dir, "cpp_slope_comparison.png")
                slope_fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"✅ Saved: cpp_slope_comparison.png")
                plt.close(slope_fig)
            except Exception as e:
                print(f"❌ Error in slope comparison: {e}")

            # 2. Onset analysis for all three
            print("\n--- CPP Onset Analysis (All Conditions) ---")
            for name, data in [
                ("Pre", nop300_pre),
                ("Post", nop300_post),
                ("Online", nop300_online),
            ]:
                try:
                    onset_fig, onset_stats = cpp_onset_analysis(
                        data, time_ms, cpp_channels
                    )
                    save_path = os.path.join(
                        output_dir, f"cpp_onset_{name.lower()}.png"
                    )
                    onset_fig.savefig(save_path, dpi=150, bbox_inches="tight")
                    print(
                        f"✅ {name}: onset = {onset_stats['mean_onset']:.1f} ± {onset_stats['std_onset']:.1f} ms"
                    )
                    plt.close(onset_fig)
                except Exception as e:
                    print(f"❌ {name}: {e}")

            # 3. CPP-P300 relationships
            print("\n--- CPP-P300 Relationships (All Conditions) ---")
            for name, cpp_data, p300_data in [
                ("Pre", nop300_pre, p300_pre),
                ("Post", nop300_post, p300_post),
                ("Online", nop300_online, p300_online),
            ]:
                try:
                    rel_fig, rel_stats = cpp_p300_relationship(
                        cpp_data, p300_data, time_ms, cpp_channels, pz
                    )
                    save_path = os.path.join(
                        output_dir, f"cpp_p300_relationship_{name.lower()}.png"
                    )
                    rel_fig.savefig(save_path, dpi=150, bbox_inches="tight")
                    print(
                        f"✅ {name}: CPP Slope → P300 Amp: r = {rel_stats['corr_slope_amp']:.3f}"
                    )
                    plt.close(rel_fig)
                except Exception as e:
                    print(f"❌ {name}: {e}")

            # 4. Comprehensive report
            print("\n--- Generating Comprehensive Report ---")
            try:
                report = comprehensive_cpp_report(
                    nop300_pre,
                    nop300_post,
                    nop300_online,
                    p300_pre,
                    p300_post,
                    p300_online,
                    time_ms,
                    cpp_channels,
                    pz,
                    labels,
                )
                # Save report to text file
                report_path = os.path.join(output_dir, "cpp_comprehensive_report.txt")
                with open(report_path, "w") as f:
                    f.write("=" * 70 + "\n")
                    f.write("COMPREHENSIVE CPP ANALYSIS REPORT\n")
                    f.write("=" * 70 + "\n")
                    f.write(f"\nSubject ID: {ID}\n")
                    f.write(f"Session: {session}\n")
                    f.write(f"CPP Channels: {cpp_channel_names}\n")
                    f.write(f"\n" + str(report))
                print(f"✅ Saved comprehensive report: cpp_comprehensive_report.txt")
            except Exception as e:
                print(f"❌ Error generating report: {e}")

        print("\n" + "=" * 70)
        print("✅ ENHANCED CPP ANALYSIS COMPLETE!")
        print(f"📁 All figures saved to: {output_dir}")
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
                gaussian_filter1d(pz_pre_target, sigma=2),
                label="P300 Pre",
                color="blue",
            )
            plt.plot(
                time_ms,
                gaussian_filter1d(pz_post_target, sigma=2),
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
            plt.title("P300 Analysis at Fz")
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

            # Difference waves
            print("\n=== Difference Wave Analysis ===")
            diff_fig = difference_wave_analysis(
                p300_pre,
                p300_post,
                time_ms,
                channels_to_plot=[25, 6, cz],
                labels=labels,
            )
            plt.show()  # <-- Change this
            plt.pause(0.1)

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
            print("\n=== Statistical Comparison (Pre vs Post) ===")
            stat_fig, clusters = cluster_based_comparison(
                p300_pre, p300_post, time_ms, 25, n_permutations=1000
            )
            plt.show()  # <-- Change this
            plt.pause(0.1)"""

        elif comparison == 1:
            pz_pre_target = np.mean(p300_pre[:, pz, :], axis=1)
            pz_post_target = np.mean(p300_post[:, pz, :], axis=1)
            pz_online_target = np.mean(p300_online[:, pz, :], axis=1)

            stdpre = np.std(p300_pre[:, pz, :], axis=1) / np.sqrt(p300_pre.shape[2])
            stdpost = np.std(p300_post[:, pz, :], axis=1) / np.sqrt(p300_post.shape[2])
            stdonline = np.std(p300_online[:, pz, :], axis=1) / np.sqrt(
                p300_online.shape[2]
            )

            plt.figure()
            plt.plot(
                time_ms,
                gaussian_filter1d(pz_pre_target, sigma=2),
                label="P300 Pre Offline",
                color="blue",
            )
            plt.plot(
                time_ms,
                gaussian_filter1d(pz_post_target, sigma=2),
                label="P300 Post Offline",
                color="red",
            )
            plt.plot(
                time_ms,
                gaussian_filter1d(pz_online_target, sigma=2),
                label="P300 Online",
                color="green",
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
            plt.fill_between(
                time_ms,
                pz_online_target - stdonline,
                pz_online_target + stdonline,
                color="green",
                alpha=0.2,
            )
            plt.title("P300 Analysis Comparison at FSz")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (µV)")
            plt.legend()
            plt.grid(True)
            plt.show()

            m_pre = p3_metrics(time_ms, pz_pre_target)  # blue
            m_post = p3_metrics(time_ms, pz_post_target)  # red
            m_on = p3_metrics(time_ms, pz_online_target)  # green
            print("\n=== P300 Metrics ===")
            print(f"Pre metrics : {m_pre}")
            print(f"Post metrics : {m_post}")
            print(f"Online metrics : {m_on}")

            ptp_pre = n2_p3_ptp(time_ms, pz_pre_target)
            ptp_post = n2_p3_ptp(time_ms, pz_post_target)
            ptp_on = n2_p3_ptp(time_ms, pz_online_target)

            t_ms = np.arange(p300_online.shape[0]) * 1000 / fs

            # ensure trials×time
            X = p300_online[
                :, pz, :
            ].T  # was (n_times, n_trials) -> now (n_trials, n_times)

            print(f"X shape (trials×time): {X.shape}")

            # compute jitter
            jitter_sd, per_trial_lat = single_trial_latency_jitter(t_ms, X)

            print(
                f"Online jitter SD: {jitter_sd:.2f} ms; n={len(per_trial_lat)} trials"
            )

            # ---- Peak-to-peak amplitude (N2–P3) ----
            print("\n=== N2–P3 Peak-to-Peak Amplitude (µV) ===")
            print(f"Offline PRE:   {ptp_pre:.3f} µV")
            print(f"Offline POST:  {ptp_post:.3f} µV")
            print(f"Online (BCI):  {ptp_on:.3f} µV")

            # ---- Latency jitter ----
            print("\n=== Single-Trial Latency Jitter ===")
            print(f"Online jitter SD: {jitter_sd:.2f} ms")
            print(f"Number of trials: {len(per_trial_lat)}")

            # Optional: Show summary statistics of per-trial latencies
            print(f"Mean latency:  {np.mean(per_trial_lat):.2f} ms")
            print(f"Min latency:   {np.min(per_trial_lat):.2f} ms")
            print(f"Max latency:   {np.max(per_trial_lat):.2f} ms")

            plt.hist(per_trial_lat, bins=10, edgecolor="black")
            plt.xlabel("P3 Latency (ms)")
            plt.ylabel("Trial Count")
            plt.title("Online BCI: Single-Trial P3 Latency Distribution")
            plt.show()

            # Use with your online trials at Pz (trials×time)
            X = p300_online[:, pz, :].T
            Xw, shifts_ms = woody_align_window(
                t_ms, X, window=(300, 600), max_shift_ms=80
            )
            erp_online_aligned = Xw.mean(axis=0)

            # Recompute metrics on the aligned ERP
            m_on_aligned = p3_metrics(t_ms, erp_online_aligned)
            print("Aligned online:", m_on_aligned)
            print(
                f"Alignment SD: {np.std(shifts_ms):.1f} ms (should be close to jitter SD)"
            )
            # per_trial_lat already computed (ms)
            trial_idx = np.arange(len(per_trial_lat)) + 1
            rho, p = spearmanr(trial_idx, per_trial_lat)
            print(f"Latency drift: Spearman r={rho:.2f}, p={p:.3g}")

            plt.plot(trial_idx, per_trial_lat, ".-")
            plt.xlabel("Trial #")
            plt.ylabel("P3 latency (ms)")
            plt.title("P3 latency across trials (online)")
            plt.show()

            X = p300_online[:, pz, :].T
            Xw, shifts_ms = woody_align_window(
                t_ms, X, window=(300, 600), max_shift_ms=80
            )
            erp_online_aligned = Xw.mean(axis=0)

            erp_on_raw = X.mean(axis=0)
            erp_on_aln = Xw.mean(axis=0)

            m_on_raw = p3_metrics(
                t_ms, erp_on_raw, baseline_window=(-400, -250), snr_method="mad"
            )
            m_on_aln = p3_metrics(
                t_ms, erp_on_aln, baseline_window=(-400, -250), snr_method="mad"
            )
            print("Online (raw):    ", m_on_raw)
            print("Online (aligned):", m_on_aln)
            print(f"Alignment SD (ms): {np.std(shifts_ms):.1f}")

            from sklearn.cluster import KMeans

            lat = per_trial_lat.reshape(-1, 1)
            km = KMeans(n_clusters=2, n_init=20, random_state=0).fit(lat)
            labels = km.labels_
            print("Cluster means (ms):", sorted(km.cluster_centers_.ravel()))

            # Compare ERPs of the two latency clusters
            erp_fast = X[labels == 0].mean(axis=0)
            erp_slow = X[labels == 1].mean(axis=0)

            print("Fast cluster:", p3_metrics(t_ms, erp_fast))
            print("Slow cluster:", p3_metrics(t_ms, erp_slow))

            sign_ref = np.sign(np.max(p300_post) - abs(np.min(p300_post))) or 1.0
            erp_online_pz_fixed = sign_ref * X.mean(axis=0)  # X.mean: online ERP at Pz
            print(p3_metrics(t_ms, erp_online_pz_fixed, snr_method="mad"))

            erp_online_pz = X.mean(axis=0)  # grand-average (online) at Pz

            def sem(a, axis=0):
                a = np.asarray(a)
                return np.std(a, axis=axis, ddof=1) / np.sqrt(a.shape[axis])

            # ====== Figure 1: Pre vs Post vs Online ======
            fig1, ax1 = plt.subplots(figsize=(8, 4))

            # Online SEM from trials
            online_sem = sem(X, axis=0)

            ax1.plot(t_ms, pz_pre_target, label="Pre (offline)")
            ax1.plot(t_ms, pz_post_target, label="Post (offline)")
            ax1.plot(t_ms, erp_online_pz, label="Online day1 (BCI)")

            ax1.fill_between(
                t_ms,
                pz_pre_target - stdpre,
                pz_pre_target + stdpre,
                alpha=0.2,
                color="blue",
            )

            ax1.fill_between(
                t_ms,
                pz_post_target - stdpost,
                pz_post_target + stdpost,
                alpha=0.2,
                color="red",
            )

            ax1.fill_between(
                t_ms,
                erp_online_pz - online_sem,
                erp_online_pz + online_sem,
                alpha=0.2,
                color="green",
            )

            ax1.set_title("ERP at Fz: Pre vs Post vs Online day 1")
            ax1.set_xlabel("Time (ms)")
            ax1.set_ylabel("Amplitude (µV)")
            ax1.axvline(0, linestyle="--", linewidth=1)
            ax1.legend(loc="best")
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            # ====== Figure 2: Online FAST vs SLOW clusters at Pz ======
            # Determine which label is earlier (FAST) based on cluster mean latencies you computed
            # If you don’t have per_trial_lat here, infer FAST/SLOW by ERP center-of-mass or keep 0/1 as-is.
            # For simplicity, we’ll compute means by label 0 and 1 and you can swap if needed.

            X0 = X[labels == 0]  # cluster 0 trials
            X1 = X[labels == 1]  # cluster 1 trials

            erp0 = X0.mean(axis=0)
            erp1 = X1.mean(axis=0)
            sem0 = sem(X0, axis=0) if len(X0) > 1 else np.zeros_like(erp0)
            sem1 = sem(X1, axis=0) if len(X1) > 1 else np.zeros_like(erp1)

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(t_ms, erp0, label=f"Cluster 0 (n={len(X0)})")
            ax2.fill_between(t_ms, erp0 - sem0, erp0 + sem0, alpha=0.2)

            ax2.plot(t_ms, erp1, label=f"Cluster 1 (n={len(X1)})")
            ax2.fill_between(t_ms, erp1 - sem1, erp1 + sem1, alpha=0.2)

            # Highlight canonical P3 windows you used
            ax2.axvspan(250, 400, alpha=0.1, label="FAST window (250–400 ms)")
            ax2.axvspan(420, 620, alpha=0.1, label="SLOW window (420–620 ms)")

            ax2.set_title("Online day 1 ERP at Fz: FAST vs SLOW clusters")
            ax2.set_xlabel("Time (ms)")
            ax2.set_ylabel("Amplitude (µV)")
            ax2.axvline(0, linestyle="--", linewidth=1)
            ax2.legend(loc="best")
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            # ===== Clustered ERPs by condition (PRE, POST, ONLINE) =====
            # Build trials×time matrices at Pz
            X_pre = trials_time_from_tc_tr(p300_pre, pz)
            X_post = trials_time_from_tc_tr(p300_post, pz)
            X_on = trials_time_from_tc_tr(p300_online, pz)

            print("\n--- OFFLINE PRE: latency clustering at Pz ---")
            res_pre = cluster_latency_and_plot(
                time_ms, X_pre, title="Offline PRE ERP at Pz: FAST vs SLOW clusters"
            )
            print(
                f"PRE jitter SD: {res_pre['jitter_sd']:.2f} ms | centers: {np.array(res_pre['centers_ms'])}"
            )
            print("  Cluster 0 metrics:", p3_metrics(time_ms, res_pre["erp0"]))
            print("  Cluster 1 metrics:", p3_metrics(time_ms, res_pre["erp1"]))

            print("\n--- OFFLINE POST: latency clustering at Pz ---")
            res_post = cluster_latency_and_plot(
                time_ms, X_post, title="Offline POST ERP at Pz: FAST vs SLOW clusters"
            )
            print(
                f"POST jitter SD: {res_post['jitter_sd']:.2f} ms | centers: {np.array(res_post['centers_ms'])}"
            )
            print("  Cluster 0 metrics:", p3_metrics(time_ms, res_post["erp0"]))
            print("  Cluster 1 metrics:", p3_metrics(time_ms, res_post["erp1"]))

            print("\n--- ONLINE: latency clustering at Pz ---")
            res_on = cluster_latency_and_plot(
                time_ms, X_on, title="Online ERP at Fz: FAST vs SLOW clusters"
            )
            print(
                f"ONLINE jitter SD: {res_on['jitter_sd']:.2f} ms | centers: {np.array(res_on['centers_ms'])}"
            )
            print("  Cluster 0 metrics:", p3_metrics(time_ms, res_on["erp0"]))
            print("  Cluster 1 metrics:", p3_metrics(time_ms, res_on["erp1"]))

            # Time-frequency
            print("\n=== Time-Frequency Analysis PRE===")
            tf_fig, _, _ = time_frequency_analysis(p300_pre, fs=512, channel_idx=pz)
            plt.show()

            print("\n=== Time-Frequency Analysis POST===")
            tf_fig, _, _ = time_frequency_analysis(p300_post, fs=512, channel_idx=pz)
            plt.show()

            print("\n=== Time-Frequency Analysis ONLINE===")
            tf_fig, _, _ = time_frequency_analysis(p300_online, fs=512, channel_idx=pz)
            plt.show()

            # Difference waves
            print("\n=== Difference Wave Analysis  -- Pre vs Post ===")
            diff_fig = difference_wave_analysis(
                p300_pre,
                p300_post,
                time_ms,
                channels_to_plot=[25, 6, cz],
                labels=labels,
            )
            plt.show()  # <-- Change this
            plt.pause(0.1)

            print("\n=== Difference Wave Analysis -- Pre vs Online ===")
            diff_fig = difference_wave_analysis(
                p300_pre,
                p300_online,
                time_ms,
                channels_to_plot=[25, 6, cz],
                labels=labels,
            )
            plt.show()  # <-- Change this
            plt.pause(0.1)

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

            # Latency distribution
            print("\n=== Latency Distribution Analysis Online ===")
            lat_fig, lat_stats = latency_distribution_analysis(
                p300_online, time_ms, 25, window=(250, 600)
            )
            plt.show()
            plt.pause(0.1)

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

        # ===== 1. N200-P300 RELATIONSHIP =====
        print("\n=== 1. N200-P300 Temporal Relationship ===")
        try:
            # Analyze for Pre condition
            n2p3_pre_fig, n2p3_pre_stats = n200_p300_temporal_relationship(
                p300_pre, time_ms, pz, n200_window=(150, 250), p300_window=(300, 600)
            )

            save_path = os.path.join(targeted_dir, "n200_p300_relationship_pre.png")
            n2p3_pre_fig.savefig(save_path, dpi=150, bbox_inches="tight")

            print(f"✅ Pre N200-P300 analysis:")
            print(
                f"   N200 amp → P300 amp: r={n2p3_pre_stats['corr_amp']:.3f}, p={n2p3_pre_stats['p_amp']:.4f}"
            )
            print(
                f"   N200 lat → P300 lat: r={n2p3_pre_stats['corr_lat']:.3f}, p={n2p3_pre_stats['p_lat']:.4f}"
            )
            print(
                f"   Mean N200-P300 interval: {np.mean(n2p3_pre_stats['interval']):.1f} ms"
            )

            plt.close(n2p3_pre_fig)

            # Analyze for Post condition
            n2p3_post_fig, n2p3_post_stats = n200_p300_temporal_relationship(
                p300_post, time_ms, pz, n200_window=(150, 250), p300_window=(300, 600)
            )

            save_path = os.path.join(targeted_dir, "n200_p300_relationship_post.png")
            n2p3_post_fig.savefig(save_path, dpi=150, bbox_inches="tight")

            print(f"✅ Post N200-P300 analysis:")
            print(
                f"   N200 amp → P300 amp: r={n2p3_post_stats['corr_amp']:.3f}, p={n2p3_post_stats['p_amp']:.4f}"
            )
            print(
                f"   N200 lat → P300 lat: r={n2p3_post_stats['corr_lat']:.3f}, p={n2p3_post_stats['p_lat']:.4f}"
            )
            print(
                f"   Mean N200-P300 interval: {np.mean(n2p3_post_stats['interval']):.1f} ms"
            )

            plt.close(n2p3_post_fig)

            # If online data available
            if comparison == 1:
                n2p3_online_fig, n2p3_online_stats = n200_p300_temporal_relationship(
                    p300_online,
                    time_ms,
                    pz,
                    n200_window=(150, 250),
                    p300_window=(300, 600),
                )

                save_path = os.path.join(
                    targeted_dir, "n200_p300_relationship_online.png"
                )
                n2p3_online_fig.savefig(save_path, dpi=150, bbox_inches="tight")

                print(f"✅ Online N200-P300 analysis:")
                print(
                    f"   N200 amp → P300 amp: r={n2p3_online_stats['corr_amp']:.3f}, p={n2p3_online_stats['p_amp']:.4f}"
                )

                plt.close(n2p3_online_fig)

        except Exception as e:
            print(f"❌ Error in N200-P300 analysis: {e}")
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
                left_channels = ["P3", "CP1", "C3"]
                right_channels = ["P4", "CP2", "C4"]

            # Pre condition
            try:
                lat_pre_fig, lat_pre_stats = hemispheric_lateralization_analysis(
                    p300_pre,
                    time_ms,
                    labels,
                    left_channels=left_channels,
                    right_channels=right_channels,
                )

                save_path = os.path.join(targeted_dir, "lateralization_pre.png")
                lat_pre_fig.savefig(save_path, dpi=150, bbox_inches="tight")

                print(f"✅ Pre lateralization:")
                print(f"   Interpretation: {lat_pre_stats['interpretation']}")
                print(f"   Mean LI: {lat_pre_stats['mean_li']:.3f}")
                print(
                    f"   Left vs Right: t={lat_pre_stats['t_stat']:.2f}, p={lat_pre_stats['p_value']:.4f}"
                )

                plt.close(lat_pre_fig)
            except Exception as e:
                print(f"⚠️ Pre lateralization failed: {e}")

            # Post condition
            try:
                lat_post_fig, lat_post_stats = hemispheric_lateralization_analysis(
                    p300_post,
                    time_ms,
                    labels,
                    left_channels=left_channels,
                    right_channels=right_channels,
                )

                save_path = os.path.join(targeted_dir, "lateralization_post.png")
                lat_post_fig.savefig(save_path, dpi=150, bbox_inches="tight")

                print(f"✅ Post lateralization:")
                print(f"   Interpretation: {lat_post_stats['interpretation']}")
                print(f"   Mean LI: {lat_post_stats['mean_li']:.3f}")
                print(
                    f"   Left vs Right: t={lat_post_stats['t_stat']:.2f}, p={lat_post_stats['p_value']:.4f}"
                )

                plt.close(lat_post_fig)
            except Exception as e:
                print(f"⚠️ Post lateralization failed: {e}")

            # Online condition (if available)
            if comparison == 1:
                try:
                    lat_online_fig, lat_online_stats = (
                        hemispheric_lateralization_analysis(
                            p300_online,
                            time_ms,
                            labels,
                            left_channels=left_channels,
                            right_channels=right_channels,
                        )
                    )

                    save_path = os.path.join(targeted_dir, "lateralization_online.png")
                    lat_online_fig.savefig(save_path, dpi=150, bbox_inches="tight")

                    print(f"✅ Online lateralization:")
                    print(f"   Interpretation: {lat_online_stats['interpretation']}")
                    print(f"   Mean LI: {lat_online_stats['mean_li']:.3f}")

                    plt.close(lat_online_fig)
                except Exception as e:
                    print(f"⚠️ Online lateralization failed: {e}")

        except Exception as e:
            print(f"❌ Error in lateralization analysis: {e}")
            import traceback

            traceback.print_exc()

        # ===== SUMMARY =====
        print("\n" + "=" * 70)
        print("TARGETED ANALYSIS SUMMARY")
        print("=" * 70)

        print("\n1. N200-P300 Relationship:")
        if "n2p3_pre_stats" in locals() and "n2p3_post_stats" in locals():
            pre_coupled = n2p3_pre_stats["p_amp"] < 0.05
            post_coupled = n2p3_post_stats["p_amp"] < 0.05

            print(
                f"   Pre:  {'✓ Coupled' if pre_coupled else '✗ Not coupled'} (p={n2p3_pre_stats['p_amp']:.4f})"
            )
            print(
                f"   Post: {'✓ Coupled' if post_coupled else '✗ Not coupled'} (p={n2p3_post_stats['p_amp']:.4f})"
            )

            if pre_coupled or post_coupled:
                print(f"   → N200 amplitude predicts P300 amplitude")

        print("\n2. Hemispheric Lateralization:")
        if "lat_pre_stats" in locals():
            print(f"   Pre:  {lat_pre_stats['interpretation']}")
        if "lat_post_stats" in locals():
            print(f"   Post: {lat_post_stats['interpretation']}")

            if "lat_pre_stats" in locals() and "lat_post_stats" in locals():
                # Check if lateralization changed
                pre_left = lat_pre_stats["mean_li"] > 0.1
                post_left = lat_post_stats["mean_li"] > 0.1
                pre_right = lat_pre_stats["mean_li"] < -0.1
                post_right = lat_post_stats["mean_li"] < -0.1

                if pre_left != post_left or pre_right != post_right:
                    print(f"   → Lateralization pattern CHANGED from Pre to Post")
                else:
                    print(f"   → Lateralization pattern STABLE from Pre to Post")

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

        # Check available data
        available_data = {}
        if p300_pre is not None:
            available_data["Pre"] = p300_pre
        if p300_post is not None:
            available_data["Post"] = p300_post
        if comparison == 1 and p300_online is not None:
            available_data["Online"] = p300_online

        if len(available_data) == 0:
            print("❌ No P300 data available")
            return

        print(f"\nAvailable conditions: {list(available_data.keys())}")

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
            p300_data_dict=available_data,
            labels=labels,
            time_ms=time_ms,
            output_dir=source_dir,
            sfreq=512,
            ch_pos_file=ch_pos_file,
        )

    elif c == 8:
        # ===== NEW: N200-P300 CORRELATION ANALYSIS =====
        print("\n" + "=" * 70)
        print("RUNNING N200-P300 CORRELATION ANALYSIS")
        print("=" * 70)

        # Analyze each condition
        results = {}

        # Pre condition
        if p300_pre is not None:
            time_ms = np.arange(p300_pre.shape[0]) * 1000 / fs

            results["Pre"] = analyze_n200_p300_correlation(
                erp_data=p300_pre,  # Shape: (trials, channels, timepoints)
                channel_labels=labels,  # List of channel names
                time_ms=time_ms,  # Time vector
                condition_name="Pre",
            )

            components = identify_early_components(
                erp_data=p300_pre,  # Shape: (512, 32, 54) = (time, ch, trials)
                channel_labels=labels,
                time_ms=time_ms,
                condition_name="Online",
                data_format="time_ch_trials",  # ← IMPORTANT: Specify this!
            )

            if results["Pre"] is not None:
                plt.savefig(
                    f"{output_dir}/n200_p300_correlation_pre.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.show()
                plt.close()

        # Post condition
        if p300_post is not None:
            time_ms = np.arange(p300_post.shape[0]) * 1000 / fs

            results["Post"] = analyze_n200_p300_correlation(
                erp_data=p300_post,
                channel_labels=labels,
                time_ms=time_ms,
                condition_name="Post",
            )

            components = identify_early_components(
                erp_data=p300_post,  # Shape: (512, 32, 54) = (time, ch, trials)
                channel_labels=labels,
                time_ms=time_ms,
                condition_name="Online",
                data_format="time_ch_trials",  # ← IMPORTANT: Specify this!
            )
            if results["Post"] is not None:
                plt.savefig(
                    f"{output_dir}/n200_p300_correlation_post.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.show()
                plt.close()

        # Online condition
        if p300_online is not None:
            time_ms = np.arange(p300_online.shape[0]) * 1000 / fs

            results["Online"] = analyze_n200_p300_correlation(
                erp_data=p300_online,
                channel_labels=labels,
                time_ms=time_ms,
                condition_name="Online",
            )

            components = identify_early_components(
                erp_data=p300_online,  # Shape: (512, 32, 54) = (time, ch, trials)
                channel_labels=labels,
                time_ms=time_ms,
                condition_name="Online",
                data_format="time_ch_trials",  # ← IMPORTANT: Specify this!
            )

            if results["Online"] is not None:
                plt.savefig(
                    f"{output_dir}/n200_p300_correlation_online.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.show()
                plt.close()

        # ===== CREATE COMPARISON SUMMARY =====
        create_n200_p300_comparison_summary(results, output_dir)
