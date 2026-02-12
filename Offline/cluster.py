"""
© 2026 Alexandra Mikhael. All Rights Reserved.
"""
import numpy as np
import matplotlib as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from collections import Counter
import os


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
