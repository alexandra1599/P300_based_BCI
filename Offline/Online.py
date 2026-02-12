"""
© 2026 Alexandra Mikhael. All Rights Reserved.
"""
import os, re, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.lines import Line2D

# -------- Config --------
SUBJECT_ID = "403"
BASE = (
    Path("/home/alexandra-admin/Documents/Online") / f"sub-{SUBJECT_ID}" / "predictions"
)
N_PERM = 1000
RNG_SEED = 42

# -------- Parsers --------
LINE_RE = re.compile(
    r"true label\s+([01]).*?"
    r"\[\[\s*([0-9eE\.\-\+]+)\s*,\s*([0-9eE\.\-\+]+)\s*,\s*([0-9eE\.\-\+]+)\s*\]\]"
    r".*?predicted label\s+([01])",
    flags=re.IGNORECASE,
)
RUNNUM_FROM_NAME = re.compile(r"_(\d+)\.[tT][xX][tT]$")  # ..._8.txt


def parse_run_file(path: Path):
    y_true, y_pred = [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                y_true.append(int(m.group(1)))
                y_pred.append(int(m.group(5)))
    if not y_true:
        raise ValueError(f"No trials parsed from {path}.")
    return np.array(y_true), np.array(y_pred)


def acc_tpr_tnr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    return acc, tpr, tnr


def permutation_null(y_true_runs, y_pred_runs, n_perm=1000, rng=RNG_SEED):
    rng = np.random.default_rng(rng)
    acc_null = np.empty(n_perm)
    tpr_null = np.empty(n_perm)
    tnr_null = np.empty(n_perm)
    for b in range(n_perm):
        accs_b, tprs_b, tnrs_b = [], [], []
        for y_true, y_pred in zip(y_true_runs, y_pred_runs):
            y_perm = rng.permutation(y_true)  # shuffle labels within run
            a, r, s = acc_tpr_tnr(y_perm, y_pred)
            accs_b.append(a)
            tprs_b.append(r)
            tnrs_b.append(s)
        acc_null[b] = np.mean(accs_b)
        tpr_null[b] = np.mean(tprs_b)
        tnr_null[b] = np.mean(tnrs_b)
    return acc_null, tpr_null, tnr_null


# -------- Discover runs --------
if not BASE.is_dir():
    raise FileNotFoundError(f"Folder not found: {BASE}")

# run folders named 1,2,3,... under predictions/
run_dirs = sorted(
    [p for p in BASE.iterdir() if p.is_dir() and p.name.isdigit()],
    key=lambda p: int(p.name),
)
if not run_dirs:
    raise FileNotFoundError(f"No run subfolders found under {BASE}")

files = []
run_nums = []
for d in run_dirs:
    run = int(d.name)
    # Preferred file name: predictions_sub-<ID>_<run>.txt
    preferred = d / f"predictions_sub-{SUBJECT_ID}_{run}.txt"
    if preferred.is_file():
        files.append(preferred)
        run_nums.append(run)
        continue
    # Fallback: any single .txt in this folder
    cand = sorted(d.glob("*.txt"))
    if len(cand) == 1:
        files.append(cand[0])
        run_nums.append(run)
    elif len(cand) > 1:
        # Try to pick the one that ends with _<run>.txt
        matched = [
            c
            for c in cand
            if RUNNUM_FROM_NAME.search(c.name)
            and int(RUNNUM_FROM_NAME.search(c.name).group(1)) == run
        ]
        if matched:
            files.append(sorted(matched)[0])
            run_nums.append(run)
        else:
            raise FileNotFoundError(
                f"Multiple .txt files in {d}, and none match _{run}.txt"
            )
    else:
        raise FileNotFoundError(f"No .txt found in {d}")

# -------- Load predictions per run --------
y_true_runs, y_pred_runs = [], []
for f in files:
    yt, yp = parse_run_file(f)
    y_true_runs.append(yt)
    y_pred_runs.append(yp)

# -------- Metrics per run --------
TPR = []
TNR = []
ACC = []
for yt, yp in zip(y_true_runs, y_pred_runs):
    a, r, s = acc_tpr_tnr(yt, yp)
    ACC.append(100 * a)
    TPR.append(100 * r)
    TNR.append(100 * s)
TPR = np.array(TPR)
TNR = np.array(TNR)
ACC = np.array(ACC)
avg_acc = float(np.mean(ACC))

# -------- Permutation-based chance (across runs) --------
acc_null, tpr_null, tnr_null = permutation_null(
    y_true_runs, y_pred_runs, n_perm=N_PERM, rng=RNG_SEED
)
thr_acc = 100 * np.percentile(acc_null, 95)
thr_tpr = 100 * np.percentile(tpr_null, 95)
thr_tnr = 100 * np.percentile(tnr_null, 95)

# -------- Plot (bars TPR/TNR + ACC line + perm-95% lines) --------
runs = np.array(run_nums)
order = np.argsort(runs)
runs, TPR, TNR, ACC = runs[order], TPR[order], TNR[order], ACC[order]

x = np.arange(len(runs))
bar_width = 0.35

fig, ax1 = plt.subplots(figsize=(9, 5.5))

# Bars
b_tpr = ax1.bar(x - bar_width / 2, TPR, bar_width, label="TPR (Recall)")
b_tnr = ax1.bar(x + bar_width / 2, TNR, bar_width, label="TNR (Specificity)")

# Accuracy line (secondary axis)
ax2 = ax1.twinx()
(acc_line,) = ax2.plot(
    x, ACC, marker="o", linewidth=2, color="gray", label="Accuracy (%)"
)

# --- Distinct colors for chance lines ---
chance_tpr_c = "tab:blue"
chance_tnr_c = "tab:orange"
chance_acc_c = "tab:gray"

# Draw lines (no labels here — we’ll add a single legend entry via proxies)
ax1.axhline(thr_tpr, linestyle=":", linewidth=1.8, color=chance_tpr_c)
ax1.axhline(thr_tnr, linestyle=":", linewidth=1.8, color=chance_tnr_c)
ax2.axhline(thr_acc, linestyle=":", linewidth=1.8, color=chance_acc_c)

# Axes & ticks
ax1.set_xlabel("Run")
ax1.set_ylabel("TPR / TNR (%)")
ax2.set_ylabel("Accuracy (%)", color="gray")
ax1.set_xticks(x)
ax1.set_xticklabels([f"Run {r}" for r in runs])
ax1.set_ylim(0, 100)
ax2.set_ylim(0, 100)
ax1.grid(axis="y", alpha=0.25)

# --- Legend with unique entries (use proxy artists for chance lines) ---
bars, labels = ax1.get_legend_handles_labels()
lines, line_labels = ax2.get_legend_handles_labels()

chance_proxies = [
    Line2D(
        [0],
        [0],
        linestyle=":",
        color=chance_tpr_c,
        linewidth=1.8,
        label="Chance TPR (perm 95%)",
    ),
    Line2D(
        [0],
        [0],
        linestyle=":",
        color=chance_tnr_c,
        linewidth=1.8,
        label="Chance TNR (perm 95%)",
    ),
    Line2D(
        [0],
        [0],
        linestyle=":",
        color=chance_acc_c,
        linewidth=1.8,
        label="Chance Acc (perm 95%)",
    ),
]

ax1.legend(
    [b_tpr, b_tnr, acc_line] + chance_proxies,
    [
        "TPR (Recall)",
        "TNR (Specificity)",
        "Accuracy (%)",
        "Chance TPR (perm 95%)",
        "Chance TNR (perm 95%)",
        "Chance Acc (perm 95%)",
    ],
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.20),
    handlelength=3,
)

plt.title(
    f"Decoder Performance Across Runs — Avg Acc = {avg_acc:.1f}% (Subject {SUBJECT_ID})"
)
plt.tight_layout()
plt.show()

# ---- Optional: quick per-run printout
print("\nPer-run metrics")
for r, a, tpr, tnr in zip(runs, ACC, TPR, TNR):
    print(f"Run {r:>2}: ACC={a:5.1f}%  TPR={tpr:5.1f}%  TNR={tnr:5.1f}%")
print(
    f"\nPermutation 95th-percentile thresholds: ACC={thr_acc:.1f}%  TPR={thr_tpr:.1f}%  TNR={thr_tnr:.1f}%"
)


# --- Observed means across runs (needed for histograms) ---
def observed_means(y_true_runs, y_pred_runs):
    accs, tprs, tnrs = [], [], []
    for yt, yp in zip(y_true_runs, y_pred_runs):
        a, r, s = acc_tpr_tnr(yt, yp)
        accs.append(a)
        tprs.append(r)
        tnrs.append(s)
    return np.mean(accs), np.mean(tprs), np.mean(tnrs)


obs_acc, obs_tpr, obs_tnr = observed_means(y_true_runs, y_pred_runs)


# --- One-sided permutation p-values: P(null >= observed) ---
def p_value_greater(obs, null):
    null = np.asarray(null)
    return (np.sum(null >= obs) + 1) / (len(null) + 1)


p_acc = p_value_greater(obs_acc, acc_null)
p_tpr = p_value_greater(obs_tpr, tpr_null)
p_tnr = p_value_greater(obs_tnr, tnr_null)

# --- 95th percentile thresholds (already computed) ---
# thr_acc = 100 * np.percentile(acc_null, 95)
# thr_tpr = 100 * np.percentile(tpr_null, 95)
# thr_tnr = 100 * np.percentile(tnr_null, 95)


def plot_perm_hist(null_vals, observed, thr95, xlabel, pval, subject_id):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(
        100 * np.array(null_vals), bins=30, density=True, alpha=0.6, edgecolor="none"
    )
    ax.axvline(100 * observed, color="black", linewidth=2.0, label="Observed", zorder=3)
    ax.axvline(thr95, color="tab:red", linestyle=":", linewidth=2.0, label="Perm 95%")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(
        f"{xlabel} — Subject {SUBJECT_ID}\nObserved vs Permutation Null (p = {pval:.4f})"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


# Call for each metric (means across runs)
plot_perm_hist(acc_null, obs_acc, thr_acc, "Accuracy (%)", p_acc, SUBJECT_ID)
plot_perm_hist(tpr_null, obs_tpr, thr_tpr, "TPR (Recall) (%)", p_tpr, SUBJECT_ID)
plot_perm_hist(tnr_null, obs_tnr, thr_tnr, "TNR (Specificity) (%)", p_tnr, SUBJECT_ID)
