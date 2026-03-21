# Downloaded Papers

## Core Papers (Must Read)

1. **In-context Learning and Induction Heads** (olsson2022_induction_heads.pdf)
   - Authors: Olsson, Elhage, Nanda, Joseph, DasSarma, et al.
   - Year: 2022
   - arXiv: 2209.11895
   - Citations: 780+
   - Why relevant: THE seminal paper defining induction heads. Identifies them using random token sequences. Our research directly addresses the gap: do induction heads behave differently on naturalistic data?

2. **A Mathematical Framework for Transformer Circuits** (elhage2021_transformer_circuits.pdf)
   - Authors: Elhage, Nanda, Olsson, et al.
   - Year: 2021
   - arXiv: 2112.00791
   - Why relevant: Foundation for understanding transformer circuits. Introduces QK/OV circuit decomposition used to analyze induction heads.

3. **Identifying Semantic Induction Heads to Understand In-Context Learning** (semantic_induction_heads_2024.pdf)
   - Authors: Ren et al.
   - Year: 2024
   - arXiv: 2402.13055
   - Citations: 53
   - Why relevant: MOST DIRECTLY RELEVANT. Shows that attention heads encode semantic relationships (not just token copying) on naturalistic text. Defines "semantic induction heads" that go beyond the random-sequence definition.

4. **The Evolution of Statistical Induction Heads: In-Context Learning Markov Chains** (edelman2024_statistical_induction_heads.pdf)
   - Authors: Edelman, Edelman, Goel, Kakade, Zhang
   - Year: 2024
   - arXiv: 2402.11004
   - Citations: 111
   - Why relevant: Shows induction heads can compute calibrated conditional probabilities (not just copy). Demonstrates hierarchical learning phases. Explicitly calls for extension to natural language.

## Supporting Papers

5. **What needs to go right for an induction head?** (singh2024_what_needs_induction.pdf)
   - Authors: Singh et al.
   - Year: 2024
   - arXiv: 2404.07129
   - Why relevant: Identifies three subcircuits that must form for induction heads. Shows data properties affect formation timing.

6. **Birth of a Transformer: A Memory Viewpoint** (bietti2023_birth_transformer.pdf)
   - Authors: Bietti, Cabannes, Bouchacourt, Jégou, Bottou
   - Year: 2023
   - arXiv: 2306.00802
   - Why relevant: Studies how data distribution properties affect induction head formation speed. Uses quasi-naturalistic bigram statistics.

7. **From Shortcut to Induction Head** (shortcut_to_induction_2025.pdf)
   - Authors: Kawata, Song, Bietti, Nishikawa, Suzuki, Vaiter, Wu
   - Year: 2025
   - arXiv: 2512.18634
   - Why relevant: Proves data diversity determines whether transformers learn induction heads vs positional shortcuts.

8. **Selective Induction Heads** (selective_induction_heads_2025.pdf)
   - Authors: (2025)
   - arXiv: 2501.11443
   - Why relevant: Studies how induction heads select causal structures in context.

9. **Rethinking the Role of Scale for ICL** (bansal2022_rethinking_scale_icl.pdf)
   - Authors: Bansal et al.
   - Year: 2022
   - arXiv: 2212.09095
   - Why relevant: Interpretability-based study of ICL at 66B scale.

10. **Rethinking Associative Memory Mechanism in Induction Head** (rethinking_assoc_memory_2024.pdf)
    - Authors: (2024)
    - arXiv: 2410.07263
    - Why relevant: Alternative theoretical framework for understanding induction heads.

11. **Towards Universality: Mechanistic Similarity Across Architectures** (towards_universality_2024.pdf)
    - Authors: (2024)
    - arXiv: 2410.13131
    - Why relevant: Studies whether induction heads and similar mechanisms appear across different architectures.

12. **The Dual-Route Model of Induction** (dual_route_induction_2025.pdf)
    - Authors: (2025)
    - arXiv: 2503.03750
    - Why relevant: Proposes dual-route framework for understanding induction mechanisms.

13. **Unveiling Induction Heads: Provable Training Dynamics** (unveiling_induction_heads_2024.pdf)
    - Authors: (2024)
    - arXiv: 2406.15190
    - Why relevant: Theoretical analysis of induction head training dynamics.
