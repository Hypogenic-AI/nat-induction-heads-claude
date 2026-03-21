# Research Plan: Naturalistic Induction Heads

## Motivation & Novelty Assessment

### Why This Research Matters
Induction heads are considered a fundamental mechanism for in-context learning in transformers. However, they were discovered and are routinely detected using random token sequences — an artificial setup far from the natural language these models were trained on. If some induction-like mechanisms only activate on naturalistic data, we're missing a significant part of the picture of how transformers do in-context learning.

### Gap in Existing Work
Olsson et al. (2022) detect induction heads on random data, then correlate them with ICL on natural text, but never directly compare head behavior across data types. Ren et al. (2024) found "semantic induction heads" but only in one model (InternLM2-1.8B) and didn't systematically compare against random-sequence detection. No study has asked: **are the heads identified by random-sequence detection the same ones that do induction on natural text?**

### Our Novel Contribution
We directly compare induction head detection on random vs. naturalistic data in GPT-2, identifying:
1. Heads that are induction heads on both random and natural data ("universal induction heads")
2. Heads that only score as induction heads on random data ("random-only")
3. Heads that only show induction behavior on natural text ("naturalistic induction heads")
4. Whether naturalistic-only heads contribute to in-context learning

### Experiment Justification
- **Experiment 1 (Random baseline)**: Reproduce standard Olsson et al. induction head detection as baseline
- **Experiment 2 (Naturalistic detection)**: Apply same detection methodology to natural text to identify which heads show induction behavior on realistic data
- **Experiment 3 (Comparison)**: Quantify overlap/divergence between random and naturalistic head rankings
- **Experiment 4 (Ablation)**: Ablate naturalistic-only heads to test their causal role in ICL

## Research Question
Do transformer models contain attention heads that exhibit induction-like behavior (pattern completion based on prior context) specifically on naturalistic text but not on random token sequences?

## Hypothesis Decomposition
1. **H1**: Induction head scores will differ between random and naturalistic data for many heads
2. **H2**: Some heads will score high on naturalistic text but low on random sequences ("naturalistic induction heads")
3. **H3**: Naturalistic induction heads, if they exist, will contribute to in-context learning performance

## Proposed Methodology

### Approach
Use TransformerLens to load GPT-2-small and measure induction head scores on both random repeated sequences (Olsson method) and naturalistic text from OpenWebText. Compare per-head scores across conditions.

For naturalistic text, the "induction pattern" is: when a token appears that previously appeared in context, does the head attend to the token *after* the previous occurrence? This is exactly the standard detection pattern, just applied to natural text where repeated tokens arise organically.

### Experimental Steps
1. Load GPT-2-small via TransformerLens
2. Generate random repeated sequences (Olsson method) and compute per-head induction scores
3. Sample OpenWebText passages, compute per-head induction scores on natural text
4. Compare head rankings: correlation, overlap of top-K heads
5. Identify heads with large score differences between conditions
6. Ablate candidate heads and measure effect on next-token prediction loss

### Baselines
- Random-sequence induction head detection (Olsson et al.)
- Uniform/null baseline (random attention patterns)

### Evaluation Metrics
- Per-head induction score (prefix matching + copying)
- Rank correlation (Spearman) between random and naturalistic scores
- Overlap of top-K heads between conditions
- Loss difference under ablation

### Statistical Analysis Plan
- Spearman rank correlation for head rankings
- Paired t-tests for score differences
- Bootstrap confidence intervals for ablation effects
- Multiple comparison correction (Bonferroni) where needed

## Expected Outcomes
- **Supporting H1-H2**: Low correlation between random and naturalistic induction scores; some heads score high only on natural text
- **Refuting H1-H2**: High correlation; same heads detected in both conditions
- **Supporting H3**: Ablating naturalistic-only heads increases loss on natural text

## Timeline
- Phase 1-2: Setup and data prep (done)
- Phase 3: Implementation (~60 min)
- Phase 4: Run experiments (~60 min)
- Phase 5: Analysis (~30 min)
- Phase 6: Documentation (~20 min)

## Potential Challenges
- Natural text may have fewer repeated tokens than random sequences, making detection noisier → use many samples and average
- GPT-2's vocabulary is large, so exact token repeats may be sparse → also consider sub-word pattern matching
- Computational cost of running many samples → batch efficiently on GPU
