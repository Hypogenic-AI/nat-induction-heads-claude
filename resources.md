# Resources Catalog

## Summary
This document catalogs all resources gathered for the "Naturalistic Induction Heads" research project. The hypothesis is that induction heads identified on random data may differ from mechanisms operating on naturalistic (training-distribution-like) data.

## Papers
Total papers downloaded: 13

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| In-context Learning and Induction Heads | Olsson et al. | 2022 | papers/olsson2022_induction_heads.pdf | Seminal paper, 780+ citations. Defines induction heads on random data |
| A Mathematical Framework for Transformer Circuits | Elhage et al. | 2021 | papers/elhage2021_transformer_circuits.pdf | Foundation for QK/OV circuit analysis |
| Identifying Semantic Induction Heads | Ren et al. | 2024 | papers/semantic_induction_heads_2024.pdf | Most directly relevant - semantic induction heads on naturalistic text |
| Evolution of Statistical Induction Heads | Edelman et al. | 2024 | papers/edelman2024_statistical_induction_heads.pdf | Statistical (probabilistic) induction heads on Markov chains |
| What needs to go right for an induction head? | Singh et al. | 2024 | papers/singh2024_what_needs_induction.pdf | Subcircuit analysis of induction head formation |
| Birth of a Transformer | Bietti et al. | 2023 | papers/bietti2023_birth_transformer.pdf | Data distribution effects on induction head emergence |
| From Shortcut to Induction Head | Kawata et al. | 2025 | papers/shortcut_to_induction_2025.pdf | Data diversity determines mechanism selection |
| Selective Induction Heads | (2025) | 2025 | papers/selective_induction_heads_2025.pdf | Causal structure selection in context |
| Rethinking Scale for ICL | Bansal et al. | 2022 | papers/bansal2022_rethinking_scale_icl.pdf | ICL interpretability at 66B scale |
| Rethinking Associative Memory in Induction Head | (2024) | 2024 | papers/rethinking_assoc_memory_2024.pdf | Alternative theoretical framework |
| Towards Universality | (2024) | 2024 | papers/towards_universality_2024.pdf | Cross-architecture mechanistic similarity |
| Dual-Route Model of Induction | (2025) | 2025 | papers/dual_route_induction_2025.pdf | Dual-route induction framework |
| Unveiling Induction Heads | (2024) | 2024 | papers/unveiling_induction_heads_2024.pdf | Provable training dynamics |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets available: 4

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| OpenWebText | HuggingFace (Skylion007/openwebtext) | ~8M docs, ~40GB | Naturalistic web text | Streaming | Primary dataset, approximates GPT-2 training data |
| WikiText-103 | HuggingFace (wikitext-103-raw-v1) | ~100M tokens | Clean Wikipedia text | Streaming | Controlled naturalistic text |
| Tiny Shakespeare | karpathy/char-rnn | ~1.1MB | Character-level text | datasets/tiny_shakespeare.txt | Quick iteration, debugging |
| Random Sequences | Generated | Arbitrary | Baseline comparison | Generated in code | Standard Olsson et al. methodology |

See datasets/README.md for download instructions.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | Primary mech interp library | code/TransformerLens/ | Head detection, activation hooks, patching |
| ICL Dynamics | github.com/aadityasingh/icl-dynamics | Induction head formation dynamics | code/icl-dynamics/ | JAX/Equinox, Omniglot-based |
| Transformer Birth | github.com/abietti/transformer-birth | Induction head emergence study | code/transformer-birth/ | PyTorch, bigram-based |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder service (Semantic Scholar backend) with "diligent" mode
- Searched for: "induction heads transformer mechanistic interpretability"
- Found 180 papers, filtered to 48 with relevance >= 2
- Deep-read the 4 most relevant papers; skimmed abstracts of others
- GitHub search for implementations and analysis tools

### Selection Criteria
- Prioritized papers that study induction heads on non-random data
- Focused on work that analyzes data distribution effects on induction head behavior
- Selected code repos that enable practical experimentation with induction heads
- Chose datasets that represent different points on the random-to-naturalistic spectrum

### Challenges Encountered
- Olsson et al. 2022 has no dedicated code repo (methodology in TransformerLens)
- Some arXiv IDs from search results were incorrect (wrong papers at those IDs)
- The Pile dataset requires zstd decompression; OpenWebText is a suitable substitute
- Most theoretical papers (Edelman, Kawata) study synthetic settings only

### Gaps and Workarounds
- No existing code specifically compares induction heads on random vs naturalistic data — this is the novel contribution
- Semantic induction head methodology (Ren et al.) has no released code — will need to reimplement
- No single paper addresses our exact hypothesis — the experiment must synthesize approaches

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **OpenWebText** for naturalistic data (streaming, matches GPT-2 distribution)
- **Random token sequences** for baseline comparison
- **WikiText-103** for controlled experiments

### 2. Baseline Methods
- Standard induction head detection on random sequences (Olsson et al. method)
- Compare head rankings between random and naturalistic detection
- Use TransformerLens `head_detector.py` as starting point

### 3. Evaluation Metrics
- Prefix matching score (random vs naturalistic)
- Copying score (random vs naturalistic)
- Per-head comparison across data types (correlation, overlap)
- Semantic relation index (Ren et al. method)

### 4. Code to Adapt/Reuse
- **TransformerLens**: Use `HookedTransformer` for model access and `head_detector` for baseline detection. Extend detection to work with naturalistic text patterns.
- **transformer-birth**: Reference for understanding induction head emergence with different data distributions.

### 5. Recommended Experimental Pipeline
1. Load GPT-2 (small) via TransformerLens
2. Detect induction heads using random repeated sequences (baseline)
3. Run same model on OpenWebText sequences, measure attention patterns
4. Identify heads that show induction-like behavior only on naturalistic data
5. Ablate these heads and measure effect on ICL score
6. Repeat with GPT-2-medium and Pythia models for generality
