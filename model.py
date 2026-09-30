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

# Step 5 - make_dataset (not yet solved)
# TODO: implement

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

