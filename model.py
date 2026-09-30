"""
NextLat from Scratch: Next-Latent Prediction in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - grid_step
def grid_step(pos: tuple[int, int], action: int, G: int) -> tuple[tuple[int, int], bool]:
    """Apply one action (0: Up, 1: Down, 2: Left, 3: Right) to a (row, col) position."""
    # (row_offset, col_offset)
    deltas = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
    }

    if action not in deltas:
        raise ValueError(f"Invalid action: {action}. Expected 0, 1, 2, or 3.")

    dr, dc = deltas[action]
    n1, n2 = pos[0] + dr, pos[1] + dc

    if 0 <= n1 < G and 0 <= n2 < G:
        return (n1, n2), True

    return pos, False

# Step 2 - legal_actions
def legal_actions(pos: tuple, G: int) -> list:
    # TODO: Return the sorted list of legal action ids from pos.
    act = []
    for action in [0, 1, 2, 3]:
        _ , legal = grid_step(pos, action, G)
        if legal:
            act.append(action)
    return act

# Step 3 - random_walk_to_goal
def random_walk_to_goal(start: tuple, goal: tuple, G: int, max_len: int, rng) -> list:
    # TODO: Random legal moves from start until goal is reached or max_len moves.
    count = 0
    if start == goal:
        return []
    steps = []
    pos = start 
    while count < max_len:
        actions = legal_actions(pos, G)
        a = int(rng.choice(actions))
        steps.append(a)
        pos, _ = grid_step(pos, a, G)
        if pos == goal:
            break
        count += 1
    return steps

# Step 4 - encode_sequence
import torch


def encode_sequence(
    start: tuple[int, int],
    goal: tuple[int, int],
    moves: list[int],
    G: int,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a walk into a fixed-length token tensor and validity mask.

    Vocabulary:
      - Actions: 0..3
      - Cell tokens: 4 + row * G + col
      - EOS / Pad: 4 + G * G

    Sequence layout:
      [start_cell, goal_cell, move_1, ..., move_k, EOS] padded with EOS up to length T.
      If moves exceed the budget, extra moves are dropped so that EOS fits.
    """
    eos_token = 4 + G * G

    # Encode cell positions
    start_cell = 4 + start[0] * G + start[1]
    goal_cell = 4 + goal[0] * G + goal[1]

    # Prefix always contains start and goal cells (assuming T >= 2)
    prefix = [start_cell, goal_cell]

    # Reserve 1 slot for EOS; any remaining slots in T - 2 go to moves
    max_moves = max(0, T - len(prefix) - 1)
    truncated_moves = moves[:max_moves]

    # Assemble real sequence up to the first EOS
    real_tokens = prefix + truncated_moves + [eos_token]
    # Truncate in edge cases where T < len(real_tokens) (e.g. T < 3)
    real_tokens = real_tokens[:T]

    real_len = len(real_tokens)

    # Initialize padded tensors of length T
    tokens = torch.full((T,), eos_token, dtype=torch.long)
    mask = torch.zeros((T,), dtype=torch.bool)

    # Fill valid token values and mark them True in the mask
    tokens[:real_len] = torch.tensor(real_tokens, dtype=torch.long)
    mask[:real_len] = True

    return tokens, mask

# Step 5 - make_dataset
import numpy as np
import torch


def make_dataset(n: int, G: int, T: int, seed: int = 0) -> dict:
    """Generate n encoded goal-directed walks and track true cell states after each token.

    Args:
        n: Number of samples to generate.
        G: Grid dimension (G x G).
        T: Maximum sequence length.
        seed: Random seed.

    Returns:
        A dictionary containing:
          - 'tokens': (n, T) torch.long tensor of encoded sequences.
          - 'mask':   (n, T) torch.bool tensor indicating non-pad tokens.
          - 'states': (n, T) torch.long tensor of cell indices (row * G + col)
                      after each token is consumed.
          - 'G':      int, the grid size.
    """
    rng = np.random.default_rng(seed)

    all_tokens = torch.empty((n, T), dtype=torch.long)
    all_masks = torch.empty((n, T), dtype=torch.bool)
    all_states = torch.empty((n, T), dtype=torch.long)

    max_len = T - 3

    for i in range(n):
        # Draw start and goal positions uniformly at random
        start = tuple(rng.integers(0, G, size=2).tolist())
        goal = tuple(rng.integers(0, G, size=2).tolist())

        # Generate walk moves
        moves = random_walk_to_goal(
            start=start, goal=goal, G=G, max_len=max_len, rng=rng
        )

        # Encode tokens and mask
        tokens, mask = encode_sequence(start, goal, moves, G, T)
        all_tokens[i] = tokens
        all_masks[i] = mask

        # Track the true cell state after consuming each token
        states = torch.empty((T,), dtype=torch.long)
        curr_pos = start

        for t in range(T):
            tok = tokens[t].item()
            if tok in (0, 1, 2, 3):
                # Valid move actions update position
                curr_pos, _ = grid_step(curr_pos, tok, G)
            # Token 0 is start_cell, token 1 is goal_cell, and EOS/pad tokens
            # leave the position unchanged.
            states[t] = curr_pos[0] * G + curr_pos[1]

        all_states[i] = states

    return {
        "tokens": all_tokens,
        "mask": all_masks,
        "states": all_states,
        "G": G,
    }

# Step 6 - get_batch (not yet solved)
# TODO: implement

# Step 7 - causal_mask (not yet solved)
# TODO: implement

# Step 8 - init_gpt_params (not yet solved)
# TODO: implement

# Step 9 - attention_block (not yet solved)
# TODO: implement

# Step 10 - mlp_block (not yet solved)
# TODO: implement

# Step 11 - gpt_hidden_states (not yet solved)
# TODO: implement

# Step 12 - output_head (not yet solved)
# TODO: implement

# Step 13 - next_token_loss (not yet solved)
# TODO: implement

# Step 14 - init_dynamics_params (not yet solved)
# TODO: implement

# Step 15 - latent_transition (not yet solved)
# TODO: implement

# Step 16 - rollout_latents (not yet solved)
# TODO: implement

# Step 17 - next_hidden_loss (not yet solved)
# TODO: implement

# Step 18 - kl_alignment_loss (not yet solved)
# TODO: implement

# Step 19 - nextlat_loss (not yet solved)
# TODO: implement

# Step 20 - train_step (not yet solved)
# TODO: implement

# Step 21 - train_model (not yet solved)
# TODO: implement

# Step 22 - greedy_decode (not yet solved)
# TODO: implement

# Step 23 - effective_rank (not yet solved)
# TODO: implement

# Step 24 - eval_hidden_states (not yet solved)
# TODO: implement

# Step 25 - valid_move_rate (not yet solved)
# TODO: implement

# Step 26 - sequence_compression (not yet solved)
# TODO: implement

# Step 27 - detour_robustness (not yet solved)
# TODO: implement

# Step 28 - world_model_report (not yet solved)
# TODO: implement

# Step 29 - draft_from_latent (not yet solved)
# TODO: implement

# Step 30 - verify_draft (not yet solved)
# TODO: implement

# Step 31 - self_speculative_generate (not yet solved)
# TODO: implement

# Step 32 - speculative_stats (not yet solved)
# TODO: implement

