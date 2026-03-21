# Naturalistic Induction Heads

**Research question**: Do transformer induction heads, discovered via random token sequences (Olsson et al. 2022), behave differently on naturalistic text? Are there induction heads that only activate on training-distribution-like data?

## Key Findings

- **Same heads, different magnitudes**: The heads identified as induction heads on random data are the same ones that show the strongest induction signals on natural text (Spearman ρ = 0.84, top-5 overlap = 80%). No "hidden" naturalistic-only induction heads were found.
- **10–40x weaker induction on natural text**: The strongest induction head scores 0.94 on random data but only 0.08 on natural text — not because of fewer token repeats, but because natural text provides many useful attention targets beyond the induction position.
- **Enhanced copying on natural text**: Late-layer heads (L9–L11) show 2–28x higher OV copying scores on natural text than random, suggesting distribution-sensitive output circuits.
- **Causally important**: Ablating random-detected induction heads increases natural text loss by 77%, confirming they matter for real language processing.

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add torch transformer-lens datasets numpy matplotlib scipy scikit-learn tqdm einops jaxtyping
uv add "transformers<4.51"

# Run experiments
python src/induction_detection.py   # Main comparison (50 sequences, ~2 min)
python src/deep_analysis.py         # Fuzzy induction + copying scores (~8 min)
python src/plot_results.py          # Generate all figures
```

Requires a GPU (tested on NVIDIA RTX A6000). GPT-2 weights are downloaded automatically.

## File Structure

```
├── REPORT.md              # Full research report with all results
├── planning.md            # Research plan and motivation
├── src/
│   ├── induction_detection.py   # Main experiment: random vs natural induction scores + ablation
│   ├── deep_analysis.py         # Fuzzy induction, OV copying scores, attention pattern analysis
│   └── plot_results.py          # Generate all figures
├── results/
│   ├── experiment_results.json  # Quantitative results
│   ├── deep_analysis_results.json
│   ├── scores.npz / deep_scores.npz  # Raw score arrays
│   └── plots/                   # All figures
├── literature_review.md   # Background literature synthesis
├── resources.md           # Dataset and code catalog
├── papers/                # Downloaded reference papers
├── datasets/              # Data files (gitignored)
└── code/                  # Cloned reference repos
```

See [REPORT.md](REPORT.md) for the full analysis.
