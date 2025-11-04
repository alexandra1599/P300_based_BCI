import matplotlib.pyplot as plt
import numpy as np

# --- Data from your 7 summaries ---
runs = np.arange(1, 9)
TPR = np.array([36.7, 38.3, 47.7, 47, 67.0, 64.0, 63.0, 73.3])
TNR = np.array([54.6, 63.2, 60.6, 60.0, 71.0, 55.9, 60, 65.0])
ACC = np.array([46.0, 52, 62, 56.0, 60.0, 66.0, 64.0, 65])

# --- Average accuracy ---
avg_acc = float(np.mean(ACC))
print(f"Average Accuracy across runs: {avg_acc:.2f}%")

# --- Plot configuration ---
bar_width = 0.35
x = np.arange(len(runs))

fig, ax1 = plt.subplots(figsize=(9, 5.5))

# Bars (TPR/TNR)
ax1.bar(x - bar_width / 2, TPR, bar_width, label="TPR (Recall)")
ax1.bar(x + bar_width / 2, TNR, bar_width, label="TNR (Specificity)")

# Accuracy line on secondary y-axis
ax2 = ax1.twinx()
(acc_line,) = ax2.plot(
    x, ACC, marker="o", linewidth=2, color="gray", label="Accuracy (%)"
)

# Avg accuracy line
avg_line = ax2.axhline(avg_acc, linestyle="--", linewidth=1.5, color="gray", alpha=0.8)
ax2.text(
    x[-1] + 0.1,
    avg_acc,
    f"Avg Acc = {avg_acc:.1f}%",
    va="center",
    color="gray",
    fontsize=9,
)

# Axes & ticks
ax1.set_xlabel("Run")
ax1.set_ylabel("TPR / TNR (%)")
ax2.set_ylabel("Accuracy (%)", color="gray")
ax1.set_xticks(x)
ax1.set_xticklabels([f"Run {i}" for i in runs])
ax1.set_ylim(0, 100)
ax2.set_ylim(0, 100)
ax1.grid(axis="y", alpha=0.25)

# Combined legend
bars, labels = ax1.get_legend_handles_labels()
lines, line_labels = ax2.get_legend_handles_labels()
ax1.legend(
    bars + [acc_line, avg_line],
    labels + ["Accuracy (%)", "Avg Accuracy"],
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.12),
)

plt.title("Decoder Performance Across Runs")
plt.tight_layout()
plt.show()
