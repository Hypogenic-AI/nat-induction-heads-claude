# Literature Review: Naturalistic Induction Heads

## Research Area Overview

Induction heads are attention head circuits in transformers that implement pattern completion: given a sequence `[A][B]...[A]`, they predict `[B]`. First identified by Elhage et al. (2021) and extensively studied by Olsson et al. (2022), induction heads are believed to be a key mechanism underlying in-context learning (ICL). However, a critical gap exists: **induction heads have been primarily defined and detected using random token sequences**, while their role in processing naturalistic (training-distribution-like) data remains less understood.

Our research hypothesis is that induction heads identified on random data may differ from mechanisms that operate on naturalistic data resembling the model's training distribution.

## Key Papers

### Paper 1: In-context Learning and Induction Heads (Olsson et al., 2022)
- **Key Contribution**: Established that induction heads form during a "phase change" early in training, coinciding with a dramatic improvement in in-context learning ability.
- **Methodology**: Trained 34 transformer models (1-layer to 40-layer, up to 13B parameters). Detected induction heads using **repeated random token sequences** with prefix matching and copying scores.
- **Datasets Used**: Filtered Common Crawl + internet books for training; random token sequences for induction head detection; natural text for ICL measurement.
- **Key Results**:
  - Induction heads form during a sharp phase transition (~1-2% through training)
  - The phase change coincides with a loss bump and ICL improvement
  - In small models, ablating induction heads removes nearly all ICL capability
  - Same heads that copy random sequences also perform "fuzzy" matching (translation, abstract pattern completion) in large models
- **Code Available**: No dedicated repo. Methodology absorbed into TransformerLens.
- **Critical Gap**: Induction heads are **defined on random data** but claimed to drive ICL on natural text. The bridge is correlational, not causal, for large models.

### Paper 2: Identifying Semantic Induction Heads (Ren et al., 2024)
- **Key Contribution**: Discovered "semantic induction heads" that encode syntactic dependencies and knowledge graph relations, going far beyond token-level copying.
- **Methodology**: Used QK/OV circuit decomposition on InternLM2-1.8B. Measured "relation index" — whether a head's OV output boosts logits for semantically related tokens (not just copied tokens). Tracked formation during training of a ~1B model.
- **Datasets Used**: AGENDA dataset (knowledge graphs + text), spaCy-parsed syntactic dependencies, SlimPajama (627B tokens for training).
- **Key Results**:
  - Specific attention heads encode semantic relationships (Used-for, Part-of, Hyponym-of, etc.)
  - Three levels of ICL emerge progressively: loss reduction → format compliance → pattern discovery
  - Semantic induction heads correlate with pattern discovery emergence
  - Some copying heads' scores DROP when pattern discovery emerges, suggesting pure copying can be counterproductive
- **Code Available**: No code released. Uses open InternLM2-1.8B model.
- **Relevance**: **Most directly relevant** to our hypothesis. Shows naturalistic data activates richer mechanisms than random-sequence copying.

### Paper 3: The Evolution of Statistical Induction Heads (Edelman et al., 2024)
- **Key Contribution**: Showed induction heads can compute calibrated conditional probabilities (Bayesian-optimal predictions), not just deterministic copies.
- **Methodology**: Trained small 2-layer attention-only transformers on Markov chain sequences. Proved a single-head 2-layer transformer can represent bigram statistics. Analyzed gradient dynamics.
- **Datasets Used**: Synthetic Markov chains (k=2,3,8 states, Dirichlet-distributed transitions).
- **Key Results**:
  - Three-phase learning: uniform → unigram → bigram (statistical induction head)
  - Simpler unigram solution acts as a **distractor**, delaying bigram learning
  - Layer alignment is crucial: second layer learns first, then first layer aligns
  - Single-layer transformers cannot solve the task
- **Code Available**: Based on minGPT; no dedicated repo.
- **Relevance**: Demonstrates induction heads are more general than simple copying. Explicitly calls for extension to natural language.

### Paper 4: What Needs to Go Right for an Induction Head? (Singh et al., 2024)
- **Key Contribution**: Identified three subcircuits (previous token attention, QK matching, copying) whose multiplicative interaction produces the phase change.
- **Methodology**: "Artificial optogenetics" — clamping subcircuit activations during training to isolate causal roles. Used Omniglot for few-shot learning tasks.
- **Key Results**:
  - Multiple induction heads form with emergent redundancy
  - Previous token heads wire to induction heads many-to-many
  - Data properties (classes, labels) differentially affect subcircuit formation timing
  - Copy subcircuit is harder than previously assumed — especially relevant for naturalistic data with large vocabularies
- **Code Available**: https://github.com/aadityasingh/icl-dynamics (JAX/Equinox)

### Paper 5: Birth of a Transformer (Bietti et al., 2023)
- **Key Contribution**: Studied induction head emergence through an associative memory lens, using quasi-naturalistic bigram statistics.
- **Methodology**: Trained simplified 2-layer transformers on synthetic bigram sequences parameterized by character-level statistics from tiny Shakespeare.
- **Key Results**:
  - Global bigrams learned faster than induction head mechanism
  - Top-down learning order: output associations first, then attention focusing
  - Data distribution properties significantly affect formation speed
  - Training on uniform output tokens generalizes better than bigram-distributed outputs
- **Code Available**: https://github.com/abietti/transformer-birth

### Paper 6: From Shortcut to Induction Head (Kawata et al., 2025)
- **Key Contribution**: Proved that data diversity determines whether transformers learn induction heads or positional shortcuts.
- **Methodology**: Rigorous analysis of gradient-based training on a trigger-output copying task. Proved phase transition governed by "max-sum ratio" of trigger distances.
- **Key Results**:
  - Sufficient data diversity → induction head (generalizable)
  - Low diversity → positional shortcut (fails OOD)
  - Trade-off between pretraining context length and OOD generalization
- **Relevance**: Directly shows data distribution controls mechanism selection. Natural language (high diversity) should favor induction heads over shortcuts.

## Common Methodologies

1. **QK/OV Circuit Decomposition** (Elhage 2021, Ren 2024): Decompose attention heads into query-key matching and output-value circuits to understand what each head attends to and what it outputs.

2. **Prefix Matching / Copying Scores** (Olsson 2022): Quantify induction head behavior on repeated random sequences. Standard detection method but limited to token-identity matching.

3. **Training Dynamics Analysis** (Edelman 2024, Singh 2024, Bietti 2023): Track circuit formation throughout training to understand emergence order and phase transitions.

4. **Activation Patching / Ablation** (Olsson 2022, Singh 2024): Knock out or clamp specific heads/circuits to measure causal contribution to ICL.

## Standard Baselines

- **Random sequence induction head detection** (Olsson et al.): The standard baseline for identifying induction heads. Our work should compare naturalistic detection against this.
- **Prefix matching score**: Fraction of attention on previous-token-matched positions.
- **Copying score**: Whether head output increases logits of attended tokens.

## Evaluation Metrics

- **Prefix matching score**: How much each head attends to induction-relevant positions
- **Copying score**: Whether OV circuit copies attended tokens to output
- **Relation index** (Ren 2024): Whether OV circuit boosts semantically related (not just identical) tokens
- **In-context learning score**: Loss at token 500 minus loss at token 50 (Olsson 2022)
- **KL divergence to Bayes-optimal**: For statistical induction heads (Edelman 2024)

## Datasets in the Literature

- **Random token sequences**: Used in Olsson 2022 for detection (our baseline)
- **OpenWebText / Common Crawl**: Training data for GPT-2 and studied models
- **AGENDA dataset**: Knowledge graphs + text (Ren 2024)
- **SlimPajama**: 627B token corpus (Ren 2024 training)
- **Tiny Shakespeare**: Character-level text (Bietti 2023)
- **Markov chains**: Synthetic sequences (Edelman 2024)
- **Omniglot**: Few-shot image classification (Singh 2024)

## Gaps and Opportunities

1. **No systematic comparison of induction head behavior on random vs naturalistic data**: Olsson et al. detect on random data and measure ICL on natural text, but never directly compare head behavior across data types.

2. **Semantic induction heads only studied in one paper**: Ren et al. 2024 is the only work examining semantic (vs copying) induction behavior. Their methodology could be applied to more models and data types.

3. **No study of whether "random-data induction heads" and "naturalistic induction heads" are the same heads**: It's unknown whether the heads that score highest on random-sequence prefix matching are the same ones that perform semantic pattern completion on natural text.

4. **Statistical induction heads only studied on synthetic Markov chains**: Edelman et al. show induction heads can compute probabilities, but only on simple Markov data. Extending to natural language bigram/trigram statistics is an open question.

5. **Data diversity effects only studied theoretically**: Kawata et al. prove diversity matters but only on synthetic trigger-output tasks. Empirical validation on real language models is needed.

## Recommendations for Our Experiment

### Recommended Approach
1. **Compare induction head detection on random vs naturalistic data**: Use TransformerLens to detect induction heads on GPT-2 (or similar) using both repeated random sequences (Olsson method) and naturalistic text (OpenWebText). Compare which heads are identified and how scores differ.

2. **Look for heads that only activate on naturalistic data**: Identify attention heads that show induction-like behavior on natural text but NOT on random sequences — these would be "naturalistic induction heads."

3. **Apply semantic induction head methodology**: Use Ren et al.'s relation index to measure whether induction heads encode semantic relationships on naturalistic data.

### Recommended Models
- **GPT-2 (small, medium, large)**: Well-studied, supported by TransformerLens, trained on WebText
- **Pythia suite** (70M to 12B): Training checkpoints available, enabling training dynamics analysis
- **InternLM2-1.8B**: Used in Ren et al., enables comparison

### Recommended Datasets
- **OpenWebText**: Primary naturalistic data (approximates GPT-2 training distribution)
- **WikiText-103**: Clean Wikipedia text for controlled experiments
- **Random token sequences**: Baseline comparison (Olsson methodology)

### Recommended Metrics
- Prefix matching score (random vs naturalistic)
- Copying score (random vs naturalistic)
- Relation index (Ren et al. method, naturalistic only)
- Head-by-head comparison across data types
- In-context learning score on natural text with/without ablating naturalistic-only heads

### Methodological Considerations
- Use the same models to compare random vs naturalistic detection (controlled comparison)
- Consider both token-level and semantic-level induction patterns
- Track how induction head scores change across training (if using Pythia checkpoints)
- Account for sequence length effects (naturalistic text has different length distributions)
