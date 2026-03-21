# Cloned Repositories

## Repo 1: TransformerLens
- **URL**: https://github.com/TransformerLensOrg/TransformerLens
- **Purpose**: Primary mechanistic interpretability library for analyzing attention heads in GPT-style models
- **Location**: `code/TransformerLens/`
- **Key files**:
  - `transformer_lens/head_detector.py` - Induction head detection (previous token heads, duplicate token heads, induction heads)
  - `transformer_lens/HookedTransformer.py` - Main model class with hooks at every activation
  - `transformer_lens/ActivationCache.py` - Activation caching for analysis
  - `transformer_lens/patching.py` - Activation patching utilities
  - `demos/Head_Detector_Demo.ipynb` - Demo for head detection
  - `demos/Exploratory_Analysis_Demo.ipynb` - Exploratory analysis
- **Notes**: This is the canonical library for mechanistic interpretability. Supports GPT-2, GPT-Neo, Pythia, and many other models. The `detect_head` function can identify induction heads using attention pattern matching. For our research, we need to extend this to work with naturalistic text (not just repeated random sequences).

## Repo 2: ICL Dynamics
- **URL**: https://github.com/aadityasingh/icl-dynamics
- **Purpose**: Code from Singh et al. 2024 studying induction head formation dynamics
- **Location**: `code/icl-dynamics/`
- **Key files**:
  - `main.py` - Training script
  - `models.py` - Model definitions
  - `ih_paper_plots.ipynb` - Induction head analysis plots
  - `ih_paper_runs.sh` - Experiment reproduction scripts
  - `samplers.py` - Data sampling strategies
- **Notes**: JAX/Equinox based. Uses Omniglot for few-shot learning tasks. Demonstrates "artificial optogenetics" for causal manipulation during training.

## Repo 3: Transformer Birth
- **URL**: https://github.com/abietti/transformer-birth
- **Purpose**: Code from Bietti et al. 2023 studying induction head emergence with associative memory framework
- **Location**: `code/transformer-birth/`
- **Key files**:
  - `ihead_basic_main.py` - Training script for minimal model
  - `ihead_basic_model.py` - Simplified induction head model
  - `ihead_full_main.py` - Full transformer training
  - `ihead_data.py` - Data generation (bigram distributions from tiny Shakespeare)
- **Notes**: PyTorch based. Clean minimal implementation ideal for understanding induction head mechanics. Uses character-level bigram statistics from tiny Shakespeare.
