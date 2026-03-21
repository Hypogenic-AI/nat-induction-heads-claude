"""Generate all plots for the naturalistic induction heads paper."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# Load results
with open(RESULTS_DIR / "experiment_results.json") as f:
    exp_results = json.load(f)

scores = np.load(RESULTS_DIR / "scores.npz")
deep = np.load(RESULTS_DIR / "deep_scores.npz")

random_mean = scores["random_mean"]  # [12, 12]
natural_mean = scores["natural_mean"]
nat_exact = deep["nat_exact"]
nat_fuzzy = deep["nat_fuzzy"]
rand_exact = deep["rand_exact"]
nat_copy = deep["nat_copy"]
rand_copy = deep["rand_copy"]
nat_entropy = deep["nat_entropy"]
rand_entropy = deep["rand_entropy"]

n_layers, n_heads = random_mean.shape


# ── Plot 1: Heatmaps of induction scores ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im0 = axes[0].imshow(random_mean, cmap='hot', aspect='auto', vmin=0, vmax=1)
axes[0].set_title("Induction Scores: Random Repeated Sequences", fontsize=12)
axes[0].set_xlabel("Head")
axes[0].set_ylabel("Layer")
axes[0].set_xticks(range(n_heads))
axes[0].set_yticks(range(n_layers))
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(natural_mean, cmap='hot', aspect='auto', vmin=0, vmax=0.1)
axes[1].set_title("Induction Scores: Naturalistic Text (OpenWebText)", fontsize=12)
axes[1].set_xlabel("Head")
axes[1].set_ylabel("Layer")
axes[1].set_xticks(range(n_heads))
axes[1].set_yticks(range(n_layers))
plt.colorbar(im1, ax=axes[1], shrink=0.8, label="(Note: different scale)")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "induction_heatmaps.png", dpi=150, bbox_inches='tight')
plt.close()


# ── Plot 2: Scatter plot of random vs natural scores ──
fig, ax = plt.subplots(figsize=(8, 8))

rand_flat = random_mean.flatten()
nat_flat = natural_mean.flatten()

# Color by layer
colors = []
for l in range(n_layers):
    for h in range(n_heads):
        colors.append(l)

sc = ax.scatter(rand_flat, nat_flat, c=colors, cmap='viridis', alpha=0.7, s=60, edgecolors='k', linewidths=0.3)
plt.colorbar(sc, ax=ax, label="Layer")

# Label top heads
for idx in np.argsort(rand_flat)[-5:]:
    l, h = idx // n_heads, idx % n_heads
    ax.annotate(f"L{l}.H{h}", (rand_flat[idx], nat_flat[idx]),
                fontsize=8, ha='left', va='bottom')

for idx in np.argsort(nat_flat)[-3:]:
    l, h = idx // n_heads, idx % n_heads
    ax.annotate(f"L{l}.H{h}", (rand_flat[idx], nat_flat[idx]),
                fontsize=8, ha='right', va='top', color='red')

ax.set_xlabel("Induction Score (Random Sequences)", fontsize=12)
ax.set_ylabel("Induction Score (Natural Text)", fontsize=12)
ax.set_title(f"Random vs Naturalistic Induction Scores\n(Spearman r={exp_results['correlations']['spearman_r']:.3f})", fontsize=13)

# Add diagonal reference line
max_val = max(rand_flat.max(), nat_flat.max())
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label="y=x")
ax.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "random_vs_natural_scatter.png", dpi=150, bbox_inches='tight')
plt.close()


# ── Plot 3: Score ratio (natural/random) for top random-detected heads ──
fig, ax = plt.subplots(figsize=(12, 5))

# Top 20 heads by random score
top_rand_idx = np.argsort(rand_flat)[::-1][:20]
labels = [f"L{i//n_heads}.H{i%n_heads}" for i in top_rand_idx]
rand_vals = [rand_flat[i] for i in top_rand_idx]
nat_vals = [nat_flat[i] for i in top_rand_idx]

x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x - width/2, rand_vals, width, label='Random', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, nat_vals, width, label='Natural', color='coral', alpha=0.8)

ax.set_xlabel("Attention Head")
ax.set_ylabel("Induction Score")
ax.set_title("Top 20 Random-Detected Induction Heads:\nRandom vs Natural Scores")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.legend()
ax.set_yscale('log')
ax.set_ylim(0.001, 1.5)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "top_heads_comparison.png", dpi=150, bbox_inches='tight')
plt.close()


# ── Plot 4: Copying scores — natural vs random ──
fig, ax = plt.subplots(figsize=(8, 8))

nat_copy_flat = nat_copy.flatten()
rand_copy_flat = rand_copy.flatten()

sc = ax.scatter(rand_copy_flat, nat_copy_flat, c=colors, cmap='viridis', alpha=0.7, s=60, edgecolors='k', linewidths=0.3)
plt.colorbar(sc, ax=ax, label="Layer")

# Annotate notable heads
for idx in np.argsort(nat_copy_flat - rand_copy_flat)[-5:]:
    l, h = idx // n_heads, idx % n_heads
    ax.annotate(f"L{l}.H{h}", (rand_copy_flat[idx], nat_copy_flat[idx]),
                fontsize=8, ha='left', color='red')

ax.set_xlabel("OV Copying Score (Random)", fontsize=12)
ax.set_ylabel("OV Copying Score (Natural)", fontsize=12)
ax.set_title("OV Copying Scores: Random vs Natural Text", fontsize=13)

max_val = max(nat_copy_flat.max(), rand_copy_flat.max())
min_val = min(nat_copy_flat.min(), rand_copy_flat.min())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label="y=x")
ax.legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "copying_scores_scatter.png", dpi=150, bbox_inches='tight')
plt.close()


# ── Plot 5: Ablation results ──
fig, ax = plt.subplots(figsize=(8, 5))

ablation = exp_results["ablation"]
categories = []
loss_increases = []

if ablation.get("random_only"):
    categories.append("Random-only\nheads")
    loss_increases.append(ablation["random_only"]["loss_increase_pct"])

if ablation.get("universal"):
    categories.append("Universal\nheads")
    loss_increases.append(ablation["universal"]["loss_increase_pct"])

if ablation.get("naturalistic_only"):
    categories.append("Naturalistic-only\nheads")
    loss_increases.append(ablation["naturalistic_only"]["loss_increase_pct"])

categories.append("Top-5 naturalistic\nheads")
loss_increases.append(ablation["top5_naturalistic"]["loss_increase_pct"])

colors_bar = ['steelblue', 'green', 'coral', 'orange'][:len(categories)]
ax.bar(categories, loss_increases, color=colors_bar, alpha=0.8, edgecolor='k')
ax.set_ylabel("Loss Increase (%)", fontsize=12)
ax.set_title("Effect of Ablating Head Categories\non Natural Text Loss", fontsize=13)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

for i, v in enumerate(loss_increases):
    ax.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "ablation_results.png", dpi=150, bbox_inches='tight')
plt.close()


# ── Plot 6: Layer-wise analysis ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Mean induction score per layer
layer_rand = random_mean.mean(axis=1)
layer_nat = natural_mean.mean(axis=1)
axes[0].plot(range(n_layers), layer_rand, 'o-', label='Random', color='steelblue')
axes[0].plot(range(n_layers), layer_nat, 's-', label='Natural', color='coral')
axes[0].set_xlabel("Layer")
axes[0].set_ylabel("Mean Induction Score")
axes[0].set_title("Induction Score by Layer")
axes[0].legend()

# Mean copying score per layer
layer_nat_copy = nat_copy.mean(axis=1)
layer_rand_copy = rand_copy.mean(axis=1)
axes[1].plot(range(n_layers), layer_rand_copy, 'o-', label='Random', color='steelblue')
axes[1].plot(range(n_layers), layer_nat_copy, 's-', label='Natural', color='coral')
axes[1].set_xlabel("Layer")
axes[1].set_ylabel("Mean Copying Score")
axes[1].set_title("OV Copying Score by Layer")
axes[1].legend()

# Entropy comparison
layer_nat_ent = nat_entropy.mean(axis=1)
layer_rand_ent = rand_entropy.mean(axis=1)
axes[2].plot(range(n_layers), layer_rand_ent, 'o-', label='Random', color='steelblue')
axes[2].plot(range(n_layers), layer_nat_ent, 's-', label='Natural', color='coral')
axes[2].set_xlabel("Layer")
axes[2].set_ylabel("Mean Attention Entropy (bits)")
axes[2].set_title("Attention Entropy by Layer")
axes[2].legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "layer_analysis.png", dpi=150, bbox_inches='tight')
plt.close()


# ── Plot 7: Naturalistic-enriched heads ──
fig, ax = plt.subplots(figsize=(10, 6))

# Heads sorted by nat_copy - rand_copy
copy_diff = nat_copy_flat - rand_copy_flat
sorted_idx = np.argsort(copy_diff)[::-1][:30]

labels = [f"L{i//n_heads}.H{i%n_heads}" for i in sorted_idx]
diffs = [copy_diff[i] for i in sorted_idx]
bar_colors = ['coral' if d > 0 else 'steelblue' for d in diffs]

ax.barh(range(len(labels)), diffs, color=bar_colors, alpha=0.8, edgecolor='k', linewidth=0.3)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Copying Score Difference (Natural - Random)")
ax.set_title("Heads with Strongest Naturalistic Copying Enrichment")
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "naturalistic_enrichment.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"All plots saved to {PLOTS_DIR}/")
print("Files:")
for f in sorted(PLOTS_DIR.glob("*.png")):
    print(f"  {f.name}")
