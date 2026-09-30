"""
NextLat from Scratch: Next-Latent Prediction in PyTorch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end NextLat experiment (Teoh et al., arXiv:2511.05963) on a grid world.

Story: train the SAME tiny transformer twice on goal-directed random walks -
once with plain next-token prediction (GPT), once with NextLat's auxiliary
next-latent objective - then score both the way the paper does: next-token
legality, effective latent rank, sequence compression and detour robustness.
Finally, use NextLat's latent dynamics model as a free draft model for
variable-length self-speculative decoding and measure accepted tokens and
speedup. Greedy verification makes the speculative output identical to plain
greedy decoding, which the script checks.
"""
import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)

    G, T = 4, 14
    n_heads = 4
    train_ds = make_dataset(n=1024, G=G, T=T, seed=0)
    eval_ds = make_dataset(n=256, G=G, T=T, seed=1)
    print(f"grid {G}x{G}, T={T}, vocab={4 + G * G + 1}, train sequences={train_ds['tokens'].shape[0]}")

    base = dict(d_model=32, n_layers=2, n_heads=n_heads, hidden=64, steps=250, batch_size=32, lr=3e-3, beta=1.0)
    cfg_gpt = dict(base, d_steps=0, lam_h=0.0, lam_kl=0.0)
    cfg_nextlat = dict(base, d_steps=2, lam_h=1.0, lam_kl=0.5)

    # ---- 1. Train both objectives from the same initialization ----
    p_gpt, _dyn_unused, hist_gpt = train_model(train_ds, cfg_gpt, seed=0)
    print(f"GPT      next-token loss: {hist_gpt[0]['next_token']:.3f} -> {hist_gpt[-1]['next_token']:.3f}")
    p_nl, dyn_nl, hist_nl = train_model(train_ds, cfg_nextlat, seed=0)
    print(f"NextLat  next-token loss: {hist_nl[0]['next_token']:.3f} -> {hist_nl[-1]['next_token']:.3f}"
          f" | next-hidden {hist_nl[0]['next_h']:.4f} -> {hist_nl[-1]['next_h']:.4f}"
          f" | KL {hist_nl[0]['kl']:.4f} -> {hist_nl[-1]['kl']:.4f}")

    # ---- 2. World-model metrics (the paper's Table 1) ----
    kw = dict(n_heads=n_heads, n_rows=128, n_tokens=4, max_pairs=60, n_trials=40, seed=2)
    rep_gpt = world_model_report(eval_ds, p_gpt, **kw)
    rep_nl = world_model_report(eval_ds, p_nl, **kw)
    print("\nmetric                   GPT      NextLat")
    for key, arrow in [("valid_move_rate", "up"), ("effective_rank", "down"),
                       ("sequence_compression", "up"), ("detour_robustness", "up")]:
        print(f"{key:22s} {rep_gpt[key]:8.4f} {rep_nl[key]:8.4f}   (better = {arrow})")
    print("(lower effective rank = more compact latent state; the true world has only "
          f"{G * G} positions x {G * G} goals)")

    # ---- 3. Variable-length self-speculative decoding with NextLat's dynamics ----
    n_tokens, max_draft = 6, 4
    accepted, speedups, lossless = [], [], True
    for i in range(8):
        prefix = eval_ds["tokens"][i, :2].tolist()  # [start_cell, goal_cell]
        res = self_speculative_generate(p_nl, dyn_nl, n_heads, prefix, n_tokens, max_draft)
        stats = speculative_stats(res, n_tokens)
        accepted.append(stats["mean_accepted"])
        speedups.append(stats["speedup"])
        lossless = lossless and (res["tokens"] == greedy_decode(p_nl, n_heads, prefix, n_tokens))
    print(f"\nself-speculative decoding ({n_tokens} tokens, draft up to {max_draft}):")
    print(f"  mean accepted drafts per cycle: {float(np.mean(accepted)):.2f}")
    print(f"  speedup in transformer passes:  {float(np.mean(speedups)):.2f}x")
    print(f"  identical to greedy decoding:   {lossless}")
    print("\nnote: at this toy scale two effects are robust across seeds - NextLat's lower effective "
          "rank (a more compact latent state) and its usable drafts (GPT's untrained dynamics accept "
          "almost none). The other metrics are noisy on 60 pairs / 40 episodes; rerun with more "
          "training steps, sequences and trials before reading anything into small gaps.")


if __name__ == "__main__":
    main()

