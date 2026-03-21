"""
Naturalistic Induction Head Detection

Compare induction head behavior on random vs naturalistic data in GPT-2.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def load_model():
    """Load GPT-2 small via TransformerLens."""
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads/layer")
    return model


def compute_induction_score_from_cache(tokens, cache, n_layers, n_heads):
    """
    Compute induction head scores from attention patterns.

    For each head, measures what fraction of attention goes to induction-relevant
    positions: when token at position i matches token at position j (j < i),
    does position i attend to position j+1?

    This is the standard Olsson et al. methodology.
    """
    tokens_np = tokens[0].cpu().numpy()  # [seq_len]
    seq_len = len(tokens_np)

    scores = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        attn = cache["pattern", layer, "attn"]  # [n_heads, q_pos, k_pos]
        if attn.dim() == 4:
            attn = attn[0]  # remove batch dim
        attn_np = attn.cpu().float().numpy()

        for head in range(n_heads):
            head_attn = attn_np[head]  # [q_pos, k_pos]

            # For each query position, find where the same token appeared before
            # and measure attention to the position AFTER that previous occurrence
            total_weight = 0.0
            total_positions = 0

            for q_pos in range(2, seq_len):
                current_token = tokens_np[q_pos]
                # Find previous occurrences of this token
                for k_pos in range(0, q_pos - 1):
                    if tokens_np[k_pos] == current_token:
                        # Induction target: position k_pos + 1
                        target_pos = k_pos + 1
                        if target_pos < q_pos:
                            total_weight += head_attn[q_pos, target_pos]
                            total_positions += 1

            if total_positions > 0:
                scores[layer, head] = total_weight / total_positions

    return scores


def compute_induction_scores_batch(model, tokens_list):
    """Compute induction scores averaged over a batch of token sequences."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_scores = []

    for tokens in tqdm(tokens_list, desc="Computing induction scores"):
        tokens = tokens.to(DEVICE)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, remove_batch_dim=False)

        scores = compute_induction_score_from_cache(tokens, cache, n_layers, n_heads)
        all_scores.append(scores)

        # Free memory
        del cache
        torch.cuda.empty_cache()

    return np.mean(all_scores, axis=0), np.std(all_scores, axis=0)


# ─── Random sequence generation (Olsson et al. method) ───

def generate_random_repeated_sequences(model, n_sequences=50, half_len=128):
    """Generate repeated random token sequences for standard induction head detection."""
    vocab_size = model.cfg.d_vocab
    sequences = []
    for _ in range(n_sequences):
        # BOS + random_half + random_half (repeated)
        random_half = torch.randint(1, vocab_size, (half_len,))  # avoid token 0
        seq = torch.cat([
            torch.tensor([model.tokenizer.bos_token_id]),
            random_half,
            random_half
        ]).unsqueeze(0)
        sequences.append(seq)
    return sequences


# ─── Naturalistic text loading ───

def load_naturalistic_sequences(model, n_sequences=50, seq_len=256):
    """Load and tokenize OpenWebText samples for naturalistic induction detection."""
    from datasets import load_dataset

    print("Streaming OpenWebText from HuggingFace...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    sequences = []
    for item in ds:
        text = item["text"]
        if len(text) < 500:  # skip very short docs
            continue
        tokens = model.to_tokens(text, prepend_bos=True)
        if tokens.shape[1] >= seq_len:
            sequences.append(tokens[:, :seq_len])
        if len(sequences) >= n_sequences:
            break

    print(f"Loaded {len(sequences)} naturalistic sequences of length {seq_len}")
    return sequences


# ─── Prefix matching score (simpler, faster metric) ───

def compute_prefix_matching_score(tokens, cache, n_layers, n_heads):
    """
    Prefix matching score: for each query position where the token appeared
    before, what fraction of attention goes to the previous occurrence?
    (Complement to induction score - this is duplicate token detection.)
    """
    tokens_np = tokens[0].cpu().numpy()
    seq_len = len(tokens_np)

    scores = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        attn = cache["pattern", layer, "attn"]
        if attn.dim() == 4:
            attn = attn[0]
        attn_np = attn.cpu().float().numpy()

        for head in range(n_heads):
            head_attn = attn_np[head]
            total_weight = 0.0
            count = 0

            for q_pos in range(1, seq_len):
                current_token = tokens_np[q_pos]
                for k_pos in range(0, q_pos):
                    if tokens_np[k_pos] == current_token:
                        total_weight += head_attn[q_pos, k_pos]
                        count += 1

            if count > 0:
                scores[layer, head] = total_weight / count

    return scores


def compute_prefix_matching_batch(model, tokens_list):
    """Compute prefix matching scores averaged over sequences."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_scores = []

    for tokens in tqdm(tokens_list, desc="Computing prefix matching"):
        tokens = tokens.to(DEVICE)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, remove_batch_dim=False)

        scores = compute_prefix_matching_score(tokens, cache, n_layers, n_heads)
        all_scores.append(scores)

        del cache
        torch.cuda.empty_cache()

    return np.mean(all_scores, axis=0), np.std(all_scores, axis=0)


# ─── Ablation study ───

def ablation_study(model, sequences, heads_to_ablate, description=""):
    """
    Measure loss with and without ablating specific heads.
    Ablation = zero out the head's attention pattern contribution.
    """
    n_seqs = len(sequences)

    # Baseline loss
    baseline_losses = []
    for tokens in tqdm(sequences[:20], desc=f"Baseline loss ({description})"):
        tokens = tokens.to(DEVICE)
        with torch.no_grad():
            logits = model(tokens)
            # Cross-entropy loss
            loss = F.cross_entropy(
                logits[0, :-1].float(),
                tokens[0, 1:],
                reduction='mean'
            ).item()
            baseline_losses.append(loss)

    baseline_loss = np.mean(baseline_losses)

    # Ablation: hook to zero out specific heads
    def make_ablation_hook(head_idx):
        def hook_fn(pattern, hook):
            pattern[:, head_idx, :, :] = 0.0
            return pattern
        return hook_fn

    ablated_losses = []
    for tokens in tqdm(sequences[:20], desc=f"Ablated loss ({description})"):
        tokens = tokens.to(DEVICE)
        hooks = []
        for layer, head in heads_to_ablate:
            hook_name = f"blocks.{layer}.attn.hook_pattern"
            hooks.append((hook_name, make_ablation_hook(head)))

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
            loss = F.cross_entropy(
                logits[0, :-1].float(),
                tokens[0, 1:],
                reduction='mean'
            ).item()
            ablated_losses.append(loss)

    ablated_loss = np.mean(ablated_losses)

    return {
        "baseline_loss": float(baseline_loss),
        "ablated_loss": float(ablated_loss),
        "loss_increase": float(ablated_loss - baseline_loss),
        "loss_increase_pct": float((ablated_loss - baseline_loss) / baseline_loss * 100),
    }


# ─── Count repeated tokens ───

def count_repeated_tokens(tokens_list):
    """Count how many token repetitions exist in each sequence."""
    stats = []
    for tokens in tokens_list:
        t = tokens[0].cpu().numpy()
        seq_len = len(t)
        repeats = 0
        total = 0
        for i in range(1, seq_len):
            if t[i] in t[:i]:
                repeats += 1
            total += 1
        stats.append(repeats / total if total > 0 else 0)
    return np.mean(stats), np.std(stats)


# ─── Main experiment ───

def main():
    print("=" * 60)
    print("NATURALISTIC INDUCTION HEAD DETECTION")
    print("=" * 60)

    model = load_model()
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # ─── Step 1: Generate data ───
    print("\n--- Step 1: Generating/loading data ---")

    N_SEQUENCES = 50
    SEQ_LEN = 257  # 1 BOS + 128 + 128 for random; variable for natural

    random_seqs = generate_random_repeated_sequences(model, n_sequences=N_SEQUENCES, half_len=128)
    natural_seqs = load_naturalistic_sequences(model, n_sequences=N_SEQUENCES, seq_len=SEQ_LEN)

    # Count repeated tokens
    rand_repeat_rate = count_repeated_tokens(random_seqs)
    nat_repeat_rate = count_repeated_tokens(natural_seqs)
    print(f"Random seq repeat rate: {rand_repeat_rate[0]:.3f} ± {rand_repeat_rate[1]:.3f}")
    print(f"Natural seq repeat rate: {nat_repeat_rate[0]:.3f} ± {nat_repeat_rate[1]:.3f}")

    # ─── Step 2: Compute induction scores ───
    print("\n--- Step 2: Computing induction scores ---")

    t0 = time.time()
    random_ind_mean, random_ind_std = compute_induction_scores_batch(model, random_seqs)
    print(f"Random induction scores computed in {time.time()-t0:.1f}s")

    t0 = time.time()
    natural_ind_mean, natural_ind_std = compute_induction_scores_batch(model, natural_seqs)
    print(f"Naturalistic induction scores computed in {time.time()-t0:.1f}s")

    # ─── Step 3: Compare ───
    print("\n--- Step 3: Comparing random vs naturalistic ---")

    from scipy import stats as sp_stats

    # Flatten scores for correlation
    rand_flat = random_ind_mean.flatten()
    nat_flat = natural_ind_mean.flatten()

    spearman_r, spearman_p = sp_stats.spearmanr(rand_flat, nat_flat)
    pearson_r, pearson_p = sp_stats.pearsonr(rand_flat, nat_flat)

    print(f"Spearman correlation: r={spearman_r:.4f}, p={spearman_p:.2e}")
    print(f"Pearson correlation:  r={pearson_r:.4f}, p={pearson_p:.2e}")

    # Top-K overlap
    K_VALUES = [5, 10, 20]
    rand_ranked = np.argsort(rand_flat)[::-1]
    nat_ranked = np.argsort(nat_flat)[::-1]

    overlaps = {}
    for k in K_VALUES:
        top_rand = set(rand_ranked[:k])
        top_nat = set(nat_ranked[:k])
        overlap = len(top_rand & top_nat)
        overlaps[k] = overlap
        print(f"Top-{k} overlap: {overlap}/{k} heads ({overlap/k*100:.0f}%)")

    # Identify head categories
    THRESHOLD = 0.1  # Score threshold for "induction head"

    random_induction = set(np.where(rand_flat > THRESHOLD)[0])
    natural_induction = set(np.where(nat_flat > THRESHOLD)[0])

    universal = random_induction & natural_induction
    random_only = random_induction - natural_induction
    naturalistic_only = natural_induction - random_induction

    def idx_to_layer_head(idx):
        return (idx // n_heads, idx % n_heads)

    print(f"\nUsing threshold={THRESHOLD}:")
    print(f"  Universal induction heads: {len(universal)}")
    print(f"  Random-only heads: {len(random_only)}")
    print(f"  Naturalistic-only heads: {len(naturalistic_only)}")

    print("\nUniversal induction heads (L.H):")
    for idx in sorted(universal):
        l, h = idx_to_layer_head(idx)
        print(f"  L{l}.H{h}: random={rand_flat[idx]:.4f}, natural={nat_flat[idx]:.4f}")

    print("\nRandom-only heads (L.H):")
    for idx in sorted(random_only):
        l, h = idx_to_layer_head(idx)
        print(f"  L{l}.H{h}: random={rand_flat[idx]:.4f}, natural={nat_flat[idx]:.4f}")

    print("\nNaturalistic-only heads (L.H):")
    for idx in sorted(naturalistic_only):
        l, h = idx_to_layer_head(idx)
        print(f"  L{l}.H{h}: random={rand_flat[idx]:.4f}, natural={nat_flat[idx]:.4f}")

    # Heads with biggest naturalistic > random difference
    diff = nat_flat - rand_flat
    top_nat_diff = np.argsort(diff)[::-1][:10]
    print("\nTop 10 heads by naturalistic - random score difference:")
    for idx in top_nat_diff:
        l, h = idx_to_layer_head(idx)
        print(f"  L{l}.H{h}: random={rand_flat[idx]:.4f}, natural={nat_flat[idx]:.4f}, diff={diff[idx]:.4f}")

    # ─── Step 4: Ablation study ───
    print("\n--- Step 4: Ablation study ---")

    # Ablate universal heads
    if universal:
        universal_heads = [idx_to_layer_head(idx) for idx in sorted(universal)]
        universal_result = ablation_study(model, natural_seqs, universal_heads, "universal")
        print(f"Universal heads ablation: loss {universal_result['baseline_loss']:.4f} → {universal_result['ablated_loss']:.4f} (+{universal_result['loss_increase_pct']:.2f}%)")
    else:
        universal_result = None

    # Ablate naturalistic-only heads
    if naturalistic_only:
        nat_only_heads = [idx_to_layer_head(idx) for idx in sorted(naturalistic_only)]
        nat_only_result = ablation_study(model, natural_seqs, nat_only_heads, "naturalistic-only")
        print(f"Naturalistic-only heads ablation: loss {nat_only_result['baseline_loss']:.4f} → {nat_only_result['ablated_loss']:.4f} (+{nat_only_result['loss_increase_pct']:.2f}%)")
    else:
        nat_only_result = None

    # Ablate random-only heads on natural text
    if random_only:
        rand_only_heads = [idx_to_layer_head(idx) for idx in sorted(random_only)]
        rand_only_result = ablation_study(model, natural_seqs, rand_only_heads, "random-only")
        print(f"Random-only heads ablation: loss {rand_only_result['baseline_loss']:.4f} → {rand_only_result['ablated_loss']:.4f} (+{rand_only_result['loss_increase_pct']:.2f}%)")
    else:
        rand_only_result = None

    # Also ablate top naturalistic heads (by score) regardless of threshold
    top_nat_heads = [idx_to_layer_head(nat_ranked[i]) for i in range(min(5, len(nat_ranked)))]
    top_nat_result = ablation_study(model, natural_seqs, top_nat_heads, "top-5 naturalistic")
    print(f"Top-5 naturalistic heads ablation: loss {top_nat_result['baseline_loss']:.4f} → {top_nat_result['ablated_loss']:.4f} (+{top_nat_result['loss_increase_pct']:.2f}%)")

    # ─── Save results ───
    print("\n--- Saving results ---")

    results = {
        "config": {
            "model": "gpt2",
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_sequences": N_SEQUENCES,
            "seq_len": SEQ_LEN,
            "threshold": THRESHOLD,
            "seed": SEED,
        },
        "repeat_rates": {
            "random_mean": float(rand_repeat_rate[0]),
            "random_std": float(rand_repeat_rate[1]),
            "natural_mean": float(nat_repeat_rate[0]),
            "natural_std": float(nat_repeat_rate[1]),
        },
        "correlations": {
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
        },
        "top_k_overlaps": {str(k): v for k, v in overlaps.items()},
        "head_categories": {
            "universal": [list(idx_to_layer_head(i)) for i in sorted(universal)],
            "random_only": [list(idx_to_layer_head(i)) for i in sorted(random_only)],
            "naturalistic_only": [list(idx_to_layer_head(i)) for i in sorted(naturalistic_only)],
        },
        "ablation": {
            "universal": universal_result,
            "naturalistic_only": nat_only_result,
            "random_only": rand_only_result,
            "top5_naturalistic": top_nat_result,
        },
        "scores": {
            "random_induction_mean": random_ind_mean.tolist(),
            "random_induction_std": random_ind_std.tolist(),
            "natural_induction_mean": natural_ind_mean.tolist(),
            "natural_induction_std": natural_ind_std.tolist(),
        },
    }

    with open(RESULTS_DIR / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save score arrays for plotting
    np.savez(RESULTS_DIR / "scores.npz",
             random_mean=random_ind_mean, random_std=random_ind_std,
             natural_mean=natural_ind_mean, natural_std=natural_ind_std)

    print("Results saved to results/experiment_results.json")
    print("Score arrays saved to results/scores.npz")

    return results


if __name__ == "__main__":
    results = main()
