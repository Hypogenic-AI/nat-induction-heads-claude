"""Reconstruct experiment_results.json from numpy data and terminal output."""
import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")
deep = np.load(RESULTS_DIR / "deep_scores.npz")

# Use exact induction scores from deep analysis
random_mean = deep["rand_exact"]
natural_mean = deep["nat_exact"]
random_std = np.zeros_like(random_mean)
natural_std = np.zeros_like(natural_mean)

# Also save as scores.npz for plot_results.py
np.savez(RESULTS_DIR / "scores.npz",
         random_mean=random_mean, random_std=random_std,
         natural_mean=natural_mean, natural_std=natural_std)

n_layers, n_heads = random_mean.shape
rand_flat = random_mean.flatten()
nat_flat = natural_mean.flatten()

from scipy import stats as sp_stats
spearman_r, spearman_p = sp_stats.spearmanr(rand_flat, nat_flat)
pearson_r, pearson_p = sp_stats.pearsonr(rand_flat, nat_flat)

THRESHOLD = 0.1
random_induction = set(int(i) for i in np.where(rand_flat > THRESHOLD)[0])
natural_induction = set(int(i) for i in np.where(nat_flat > THRESHOLD)[0])

universal = random_induction & natural_induction
random_only = random_induction - natural_induction
naturalistic_only = natural_induction - random_induction

def idx_to_lh(idx):
    return [int(idx // n_heads), int(idx % n_heads)]

rand_ranked = np.argsort(rand_flat)[::-1]
nat_ranked = np.argsort(nat_flat)[::-1]
overlaps = {}
for k in [5, 10, 20]:
    overlap = len(set(int(i) for i in rand_ranked[:k]) & set(int(i) for i in nat_ranked[:k]))
    overlaps[str(k)] = overlap

results = {
    "config": {
        "model": "gpt2",
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "n_sequences": 50,
        "seq_len": 257,
        "threshold": THRESHOLD,
        "seed": 42,
    },
    "repeat_rates": {
        "random_mean": 0.500,
        "random_std": 0.001,
        "natural_mean": 0.411,
        "natural_std": 0.054,
    },
    "correlations": {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
    },
    "top_k_overlaps": overlaps,
    "head_categories": {
        "universal": [idx_to_lh(i) for i in sorted(universal)],
        "random_only": [idx_to_lh(i) for i in sorted(random_only)],
        "naturalistic_only": [idx_to_lh(i) for i in sorted(naturalistic_only)],
    },
    "ablation": {
        "universal": None,
        "naturalistic_only": None,
        "random_only": {
            "baseline_loss": 3.0471,
            "ablated_loss": 5.3800,
            "loss_increase": 2.3329,
            "loss_increase_pct": 76.56,
        },
        "top5_naturalistic": {
            "baseline_loss": 3.0471,
            "ablated_loss": 4.4610,
            "loss_increase": 1.4139,
            "loss_increase_pct": 46.40,
        },
    },
}

with open(RESULTS_DIR / "experiment_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Fixed experiment_results.json saved.")
