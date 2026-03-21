"""
Deep analysis: investigate WHY naturalistic induction scores are so much lower,
and look for "fuzzy" / semantic induction patterns.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda:0"
RESULTS_DIR = Path("results")


def load_model():
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    return model


def load_naturalistic_sequences(model, n_sequences=50, seq_len=257):
    from datasets import load_dataset
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    sequences = []
    for item in ds:
        text = item["text"]
        if len(text) < 500:
            continue
        tokens = model.to_tokens(text, prepend_bos=True)
        if tokens.shape[1] >= seq_len:
            sequences.append(tokens[:, :seq_len])
        if len(sequences) >= n_sequences:
            break
    return sequences


def generate_random_repeated(model, n=50, half_len=128):
    vocab_size = model.cfg.d_vocab
    seqs = []
    for _ in range(n):
        h = torch.randint(1, vocab_size, (half_len,))
        seq = torch.cat([torch.tensor([model.tokenizer.bos_token_id]), h, h]).unsqueeze(0)
        seqs.append(seq)
    return seqs


def compute_fuzzy_induction_score(model, tokens, use_embeddings=True):
    """
    Fuzzy induction: instead of requiring exact token match at previous position,
    measure attention to positions where the previous context is SIMILAR
    (using embedding cosine similarity).

    This captures "semantic induction" where the model attends to positions
    after semantically similar (not just identical) prior contexts.
    """
    tokens = tokens.to(DEVICE)
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, remove_batch_dim=False)

    tokens_np = tokens[0].cpu().numpy()
    seq_len = len(tokens_np)

    # Get token embeddings for similarity computation
    if use_embeddings:
        embed = model.W_E[tokens[0]].float()  # [seq_len, d_model]
        # Cosine similarity matrix
        norms = embed.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        embed_norm = embed / norms
        sim_matrix = embed_norm @ embed_norm.T  # [seq_len, seq_len]
        sim_np = sim_matrix.detach().cpu().numpy()
    else:
        sim_np = None

    # Standard exact-match induction score
    exact_scores = np.zeros((n_layers, n_heads))
    # Fuzzy induction score (using embedding similarity)
    fuzzy_scores = np.zeros((n_layers, n_heads))
    # "Semantic pattern completion" score
    semantic_scores = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        attn = cache["pattern", layer, "attn"]
        if attn.dim() == 4:
            attn = attn[0]
        attn_np = attn.cpu().float().numpy()

        for head in range(n_heads):
            head_attn = attn_np[head]  # [q_pos, k_pos]

            exact_total = 0.0
            exact_count = 0
            fuzzy_total = 0.0
            fuzzy_count = 0

            for q_pos in range(2, seq_len):
                current_token = tokens_np[q_pos]

                # Exact match induction
                for k_pos in range(0, q_pos - 1):
                    if tokens_np[k_pos] == current_token:
                        target = k_pos + 1
                        if target < q_pos:
                            exact_total += head_attn[q_pos, target]
                            exact_count += 1

                # Fuzzy match: find positions with similar embeddings
                if use_embeddings:
                    # Top similar positions (similarity > 0.8, excluding self)
                    sims = sim_np[q_pos, :q_pos]  # similarities to all prior positions
                    high_sim_positions = np.where(sims > 0.8)[0]

                    for k_pos in high_sim_positions:
                        target = k_pos + 1
                        if target < q_pos and target != q_pos:
                            fuzzy_total += head_attn[q_pos, target]
                            fuzzy_count += 1

            if exact_count > 0:
                exact_scores[layer, head] = exact_total / exact_count
            if fuzzy_count > 0:
                fuzzy_scores[layer, head] = fuzzy_total / fuzzy_count

    del cache
    torch.cuda.empty_cache()

    return exact_scores, fuzzy_scores


def compute_copying_score(model, tokens):
    """
    OV copying score: does the head's output boost the logit of the attended token?
    Measures whether the OV circuit copies information from attended positions.
    """
    tokens = tokens.to(DEVICE)
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, remove_batch_dim=False)

    scores = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        attn = cache["pattern", layer, "attn"]
        if attn.dim() == 4:
            attn = attn[0]

        # Get the value vectors and OV matrix output
        v = cache[f"blocks.{layer}.attn.hook_v"]
        if v.dim() == 4:
            v = v[0]  # [seq_len, n_heads, d_head]

        # The result of attention: weighted sum of values
        # result[head, q_pos] = sum_k attn[head, q, k] * v[k, head]
        # We want to check: does this result, when projected through W_O,
        # increase the logit of the most-attended token?

        # hook_z shape: [batch, seq_len, n_heads, d_head]
        z = cache[f"blocks.{layer}.attn.hook_z"]
        if z.dim() == 4:
            z = z[0]  # [seq_len, n_heads, d_head]

        # W_O for this layer: [n_heads, d_head, d_model]
        W_O = model.W_O[layer]

        for head in range(n_heads):
            head_attn = attn[head]  # [q_pos, k_pos]
            seq_len = head_attn.shape[0]

            copy_score = 0.0
            count = 0

            for q_pos in range(1, min(seq_len, 128)):  # limit for speed
                # Most attended position (excluding self and BOS)
                attn_weights = head_attn[q_pos, 1:q_pos].clone()
                if len(attn_weights) == 0:
                    continue
                max_k = attn_weights.argmax().item() + 1  # +1 because we skipped BOS
                attended_token = tokens[0, max_k].item()

                # Head output projected through W_O
                head_z = z[q_pos, head]  # [d_head]
                head_out = head_z.float() @ W_O[head].float()  # [d_model]

                # Project to vocab space via unembedding
                logit_contribution = head_out @ model.W_U.float()  # [vocab_size]

                # Does this boost the attended token's logit?
                attended_logit = logit_contribution[attended_token].item()
                mean_logit = logit_contribution.mean().item()
                copy_score += (attended_logit - mean_logit)
                count += 1

            if count > 0:
                scores[layer, head] = copy_score / count

    del cache
    torch.cuda.empty_cache()

    return scores


def analyze_attention_patterns(model, tokens):
    """
    For each head, characterize its attention pattern on natural text:
    - Entropy of attention distribution
    - Fraction attending to BOS
    - Fraction attending to previous token
    - Average attention distance
    """
    tokens = tokens.to(DEVICE)
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, remove_batch_dim=False)

    entropy = np.zeros((n_layers, n_heads))
    bos_frac = np.zeros((n_layers, n_heads))
    prev_token_frac = np.zeros((n_layers, n_heads))
    mean_distance = np.zeros((n_layers, n_heads))

    for layer in range(n_layers):
        attn = cache["pattern", layer, "attn"]
        if attn.dim() == 4:
            attn = attn[0]
        attn_np = attn.cpu().float().numpy()

        for head in range(n_heads):
            head_attn = attn_np[head]
            seq_len = head_attn.shape[0]

            # Average entropy across positions
            ent = 0.0
            bos = 0.0
            prev = 0.0
            dist = 0.0
            count = 0

            for q in range(1, seq_len):
                probs = head_attn[q, :q + 1]
                probs = probs.clip(1e-10, 1.0)
                ent += -np.sum(probs * np.log2(probs))
                bos += head_attn[q, 0]
                prev += head_attn[q, q - 1]
                positions = np.arange(q + 1)
                dist += np.sum(probs * (q - positions))
                count += 1

            if count > 0:
                entropy[layer, head] = ent / count
                bos_frac[layer, head] = bos / count
                prev_token_frac[layer, head] = prev / count
                mean_distance[layer, head] = dist / count

    del cache
    torch.cuda.empty_cache()

    return entropy, bos_frac, prev_token_frac, mean_distance


def main():
    print("=" * 60)
    print("DEEP ANALYSIS: FUZZY & SEMANTIC INDUCTION")
    print("=" * 60)

    model = load_model()
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Load data
    print("\nLoading data...")
    natural_seqs = load_naturalistic_sequences(model, n_sequences=30, seq_len=257)
    random_seqs = generate_random_repeated(model, n=30, half_len=128)

    # ─── Fuzzy induction scores ───
    print("\n--- Computing fuzzy induction scores on natural text ---")
    all_exact = []
    all_fuzzy = []
    for seq in tqdm(natural_seqs[:20], desc="Fuzzy scores (natural)"):
        exact, fuzzy = compute_fuzzy_induction_score(model, seq, use_embeddings=True)
        all_exact.append(exact)
        all_fuzzy.append(fuzzy)

    nat_exact_mean = np.mean(all_exact, axis=0)
    nat_fuzzy_mean = np.mean(all_fuzzy, axis=0)

    print("\n--- Computing fuzzy induction scores on random text ---")
    all_exact_rand = []
    all_fuzzy_rand = []
    for seq in tqdm(random_seqs[:20], desc="Fuzzy scores (random)"):
        exact, fuzzy = compute_fuzzy_induction_score(model, seq, use_embeddings=True)
        all_exact_rand.append(exact)
        all_fuzzy_rand.append(fuzzy)

    rand_exact_mean = np.mean(all_exact_rand, axis=0)
    rand_fuzzy_mean = np.mean(all_fuzzy_rand, axis=0)

    # ─── Copying scores ───
    print("\n--- Computing OV copying scores ---")
    nat_copy_scores = []
    for seq in tqdm(natural_seqs[:15], desc="Copying (natural)"):
        scores = compute_copying_score(model, seq)
        nat_copy_scores.append(scores)

    rand_copy_scores = []
    for seq in tqdm(random_seqs[:15], desc="Copying (random)"):
        scores = compute_copying_score(model, seq)
        rand_copy_scores.append(scores)

    nat_copy_mean = np.mean(nat_copy_scores, axis=0)
    rand_copy_mean = np.mean(rand_copy_scores, axis=0)

    # ─── Attention pattern analysis ───
    print("\n--- Analyzing attention patterns ---")
    nat_patterns = [analyze_attention_patterns(model, seq) for seq in tqdm(natural_seqs[:10], desc="Patterns (natural)")]
    rand_patterns = [analyze_attention_patterns(model, seq) for seq in tqdm(random_seqs[:10], desc="Patterns (random)")]

    nat_entropy = np.mean([p[0] for p in nat_patterns], axis=0)
    nat_bos = np.mean([p[1] for p in nat_patterns], axis=0)
    nat_prev = np.mean([p[2] for p in nat_patterns], axis=0)
    nat_dist = np.mean([p[3] for p in nat_patterns], axis=0)

    rand_entropy = np.mean([p[0] for p in rand_patterns], axis=0)
    rand_bos = np.mean([p[1] for p in rand_patterns], axis=0)
    rand_prev = np.mean([p[2] for p in rand_patterns], axis=0)
    rand_dist = np.mean([p[3] for p in rand_patterns], axis=0)

    # ─── Results summary ───
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    from scipy import stats as sp_stats

    # Compare exact vs fuzzy scores on natural text
    print("\n--- Exact vs Fuzzy induction on natural text ---")
    nat_exact_flat = nat_exact_mean.flatten()
    nat_fuzzy_flat = nat_fuzzy_mean.flatten()
    r, p = sp_stats.spearmanr(nat_exact_flat, nat_fuzzy_flat)
    print(f"Spearman(exact, fuzzy): r={r:.4f}, p={p:.2e}")
    print(f"Max exact score (natural): {nat_exact_flat.max():.4f}")
    print(f"Max fuzzy score (natural): {nat_fuzzy_flat.max():.4f}")
    print(f"Max exact score (random): {rand_exact_mean.flatten().max():.4f}")
    print(f"Max fuzzy score (random): {rand_fuzzy_mean.flatten().max():.4f}")

    # Identify heads with high fuzzy but low exact scores (semantic induction)
    fuzzy_minus_exact = nat_fuzzy_flat - nat_exact_flat
    top_semantic = np.argsort(fuzzy_minus_exact)[::-1][:10]
    print("\nTop 10 heads by (fuzzy - exact) on natural text:")
    for idx in top_semantic:
        l, h = idx // n_heads, idx % n_heads
        print(f"  L{l}.H{h}: exact={nat_exact_flat[idx]:.4f}, fuzzy={nat_fuzzy_flat[idx]:.4f}, diff={fuzzy_minus_exact[idx]:.4f}")

    # Copying score comparison
    print("\n--- Copying score analysis ---")
    nat_copy_flat = nat_copy_mean.flatten()
    rand_copy_flat = rand_copy_mean.flatten()
    r, p = sp_stats.spearmanr(nat_copy_flat, rand_copy_flat)
    print(f"Spearman(nat_copy, rand_copy): r={r:.4f}, p={p:.2e}")

    # Top copying heads on natural text
    top_nat_copy = np.argsort(nat_copy_flat)[::-1][:10]
    print("\nTop 10 copying heads on natural text:")
    for idx in top_nat_copy:
        l, h = idx // n_heads, idx % n_heads
        print(f"  L{l}.H{h}: nat_copy={nat_copy_flat[idx]:.4f}, rand_copy={rand_copy_flat[idx]:.4f}")

    # Heads that copy MORE on natural than random (naturalistic copiers)
    copy_diff = nat_copy_flat - rand_copy_flat
    top_nat_copiers = np.argsort(copy_diff)[::-1][:10]
    print("\nTop 10 heads by (nat_copy - rand_copy) score:")
    for idx in top_nat_copiers:
        l, h = idx // n_heads, idx % n_heads
        print(f"  L{l}.H{h}: nat_copy={nat_copy_flat[idx]:.4f}, rand_copy={rand_copy_flat[idx]:.4f}, diff={copy_diff[idx]:.4f}")

    # Attention pattern analysis
    print("\n--- Attention pattern differences ---")
    # Find heads with very different attention patterns between random and natural
    entropy_diff = nat_entropy.flatten() - rand_entropy.flatten()
    print(f"Mean attention entropy: natural={nat_entropy.mean():.3f}, random={rand_entropy.mean():.3f}")
    print(f"Mean BOS fraction: natural={nat_bos.mean():.3f}, random={rand_bos.mean():.3f}")
    print(f"Mean prev-token fraction: natural={nat_prev.mean():.3f}, random={rand_prev.mean():.3f}")
    print(f"Mean attention distance: natural={nat_dist.mean():.1f}, random={rand_dist.mean():.1f}")

    # Top-K analysis with adaptive thresholds
    print("\n--- Adaptive threshold analysis ---")
    # Use percentile-based thresholds
    for percentile in [90, 95, 99]:
        nat_thresh = np.percentile(nat_exact_flat, percentile)
        rand_thresh = np.percentile(rand_exact_mean.flatten(), percentile)
        nat_high = set(np.where(nat_exact_flat > nat_thresh)[0])
        rand_high = set(np.where(rand_exact_mean.flatten() > rand_thresh)[0])
        overlap = nat_high & rand_high
        nat_only = nat_high - rand_high
        rand_only = rand_high - nat_high
        print(f"\nTop {100-percentile}% threshold (nat>={nat_thresh:.4f}, rand>={rand_thresh:.4f}):")
        print(f"  Both: {len(overlap)}, Nat-only: {len(nat_only)}, Rand-only: {len(rand_only)}")
        if nat_only:
            print(f"  Naturalistic-only heads:")
            for idx in sorted(nat_only):
                l, h = idx // n_heads, idx % n_heads
                print(f"    L{l}.H{h}: exact_nat={nat_exact_flat[idx]:.4f}, exact_rand={rand_exact_mean.flatten()[idx]:.4f}, fuzzy_nat={nat_fuzzy_flat[idx]:.4f}")

    # Save all results
    deep_results = {
        "scores": {
            "nat_exact_mean": nat_exact_mean.tolist(),
            "nat_fuzzy_mean": nat_fuzzy_mean.tolist(),
            "rand_exact_mean": rand_exact_mean.tolist(),
            "rand_fuzzy_mean": rand_fuzzy_mean.tolist(),
            "nat_copy_mean": nat_copy_mean.tolist(),
            "rand_copy_mean": rand_copy_mean.tolist(),
        },
        "attention_patterns": {
            "nat_entropy": nat_entropy.tolist(),
            "nat_bos": nat_bos.tolist(),
            "nat_prev": nat_prev.tolist(),
            "nat_dist": nat_dist.tolist(),
            "rand_entropy": rand_entropy.tolist(),
            "rand_bos": rand_bos.tolist(),
            "rand_prev": rand_prev.tolist(),
            "rand_dist": rand_dist.tolist(),
        },
    }

    with open(RESULTS_DIR / "deep_analysis_results.json", "w") as f:
        json.dump(deep_results, f, indent=2)

    np.savez(RESULTS_DIR / "deep_scores.npz",
             nat_exact=nat_exact_mean, nat_fuzzy=nat_fuzzy_mean,
             rand_exact=rand_exact_mean, rand_fuzzy=rand_fuzzy_mean,
             nat_copy=nat_copy_mean, rand_copy=rand_copy_mean,
             nat_entropy=nat_entropy, rand_entropy=rand_entropy,
             nat_bos=nat_bos, rand_bos=rand_bos,
             nat_prev=nat_prev, rand_prev=rand_prev)

    print("\nDeep analysis results saved.")


if __name__ == "__main__":
    main()
