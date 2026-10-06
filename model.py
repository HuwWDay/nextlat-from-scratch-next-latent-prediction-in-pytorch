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

# Step 6 - get_batch
import torch


def get_batch(dataset: dict, batch_size: int, step: int) -> dict:
    """Slice a deterministic batch from the dataset cyclically and prepare causal LM targets.

    Args:
        dataset: Dict containing 'tokens', 'mask', 'states', and 'G'.
        batch_size: Number of samples per batch.
        step: Step index determining the cyclic batch slice.

    Returns:
        A dict with:
          - 'x':      (B, T - 1) tokens without the last position.
          - 'y':      (B, T - 1) tokens without the first position (targets).
          - 'mask':   (B, T - 1) mask aligned to 'y' (without the first position).
          - 'states': (B, T - 1) states aligned to 'x' (without the last position).
    """
    n = dataset["tokens"].size(0)

    # Compute cyclic row indices for this batch step
    indices = (step * batch_size + torch.arange(batch_size)) % n

    # Slice rows
    batch_tokens = dataset["tokens"][indices]
    batch_mask = dataset["mask"][indices]
    batch_states = dataset["states"][indices]

    return {
        "x": batch_tokens[:, :-1],
        "y": batch_tokens[:, 1:],
        "mask": batch_mask[:, 1:],
        "states": batch_states[:, :-1],
    }

# Step 7 - causal_mask
def causal_mask(T: int):
    # TODO: Return a (T, T) bool tensor, True where key index <= query index.
    return torch.tril(torch.ones(T, T, dtype=torch.bool))

# Step 8 - init_gpt_params
import torch


def init_gpt_params(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    max_len: int,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Initialize parameter dictionary for a small pre-LN GPT.

    Order:
      1. 'wte' (vocab_size, d_model), 'wpe' (max_len, d_model)
      2. For each layer l in 0..n_layers-1:
         - ln1_w{l}, ln1_b{l}
         - qkv_w{l} (d_model, 3*d_model), qkv_b{l}
         - proj_w{l} (d_model, d_model), proj_b{l}
         - ln2_w{l}, ln2_b{l}
         - fc_w{l} (d_model, 4*d_model), fc_b{l}
         - fc2_w{l} (4*d_model, d_model), fc2_b{l}
      3. 'lnf_w', 'lnf_b', 'head_w' (d_model, vocab_size), 'head_b'
    """
    torch.manual_seed(seed)
    params = {}

    def _rand_mat(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, dtype=torch.float32) * 0.02).requires_grad_(
            True
        )

    def _zeros(*shape: int) -> torch.Tensor:
        return torch.zeros(*shape, dtype=torch.float32, requires_grad=True)

    def _ones(*shape: int) -> torch.Tensor:
        return torch.ones(*shape, dtype=torch.float32, requires_grad=True)

    # 1. Embeddings
    params["wte"] = _rand_mat(vocab_size, d_model)
    params["wpe"] = _rand_mat(max_len, d_model)

    # 2. Transformer layers
    for l in range(n_layers):
        params[f"ln1_w{l}"] = _ones(d_model)
        params[f"ln1_b{l}"] = _zeros(d_model)

        params[f"qkv_w{l}"] = _rand_mat(d_model, 3 * d_model)
        params[f"qkv_b{l}"] = _zeros(3 * d_model)

        params[f"proj_w{l}"] = _rand_mat(d_model, d_model)
        params[f"proj_b{l}"] = _zeros(d_model)

        params[f"ln2_w{l}"] = _ones(d_model)
        params[f"ln2_b{l}"] = _zeros(d_model)

        params[f"fc_w{l}"] = _rand_mat(d_model, 4 * d_model)
        params[f"fc_b{l}"] = _zeros(4 * d_model)

        params[f"fc2_w{l}"] = _rand_mat(4 * d_model, d_model)
        params[f"fc2_b{l}"] = _zeros(d_model)

    # 3. Final LayerNorm and output head
    params["lnf_w"] = _ones(d_model)
    params["lnf_b"] = _zeros(d_model)

    params["head_w"] = _rand_mat(d_model, vocab_size)
    params["head_b"] = _zeros(vocab_size)

    return params

# Step 9 - attention_block
import math
import torch
import torch.nn.functional as F


def attention_block(
    x: torch.Tensor, params: dict[str, torch.Tensor], layer: int, n_heads: int
) -> torch.Tensor:
    """Apply one pre-LayerNorm causal multi-head self-attention block with residual connection."""
    B, T, d = x.shape
    head_dim = d // n_heads

    # Pre-LayerNorm
    ln1_w = params[f"ln1_w{layer}"]
    ln1_b = params[f"ln1_b{layer}"]
    z = F.layer_norm(x, (d,), weight=ln1_w, bias=ln1_b, eps=1e-5)

    # Q, K, V projection
    qkv_w = params[f"qkv_w{layer}"]
    qkv_b = params[f"qkv_b{layer}"]
    qkv = z @ qkv_w + qkv_b
    q, k, v = qkv.split(d, dim=-1)

    # Reshape to (B, n_heads, T, head_dim)
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)

    # Scaled dot-product scores: (B, n_heads, T, T)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)

    # Apply causal mask: mask positions that are NOT allowed (~mask) with -inf
    mask = causal_mask(T)
    if mask.dtype == torch.bool:
        scores = scores.masked_fill(~mask, float("-inf"))
    else:
        # If causal_mask returns 0.0 for keep and -inf for mask out
        scores = scores + mask

    # Softmax & weighted values
    attn_weights = F.softmax(scores, dim=-1)
    out = attn_weights @ v  # (B, n_heads, T, head_dim)

    # Merge heads back: (B, T, d)
    out = out.transpose(1, 2).contiguous().view(B, T, d)

    # Output projection
    proj_w = params[f"proj_w{layer}"]
    proj_b = params[f"proj_b{layer}"]
    out = out @ proj_w + proj_b

    return x + out

# Step 10 - mlp_block
import torch
import torch.nn.functional as F


def mlp_block(x: torch.Tensor, params: dict[str, torch.Tensor], layer: int) -> torch.Tensor:
    """Apply one pre-LayerNorm feed-forward block with a residual connection.

    Args:
        x: Input tensor of shape (B, T, d).
        params: Dictionary containing layer parameters.
        layer: Layer index.

    Returns:
        Tensor of shape (B, T, d) after MLP transformation and residual connection.
    """
    d = x.shape[-1]

    # Pre-LayerNorm
    ln2_w = params[f"ln2_w{layer}"]
    ln2_b = params[f"ln2_b{layer}"]
    z = F.layer_norm(x, (d,), weight=ln2_w, bias=ln2_b, eps=1e-5)

    # First linear projection + GELU (tanh approximation)
    fc_w = params[f"fc_w{layer}"]
    fc_b = params[f"fc_b{layer}"]
    h = F.gelu(z @ fc_w + fc_b, approximate="tanh")

    # Second linear projection
    fc2_w = params[f"fc2_w{layer}"]
    fc2_b = params[f"fc2_b{layer}"]
    out = h @ fc2_w + fc2_b

    # Residual connection
    return x + out

# Step 11 - gpt_hidden_states
import torch
import torch.nn.functional as F


def gpt_hidden_states(
    tokens: torch.Tensor, params: dict[str, torch.Tensor], n_heads: int
) -> torch.Tensor:
    """Run the transformer backbone and return final hidden states before logits.

    Args:
        tokens: Long tensor of token IDs with shape (B, T).
        params: Parameter dictionary initialized by init_gpt_params.
        n_heads: Number of attention heads.

    Returns:
        Tensor of shape (B, T, d) containing final layer-normed hidden states.
    """
    B, T = tokens.shape

    # Infer the number of transformer layers
    n_layers = sum(1 for k in params if k.startswith("ln1_w"))

    # Token embeddings + learned positional embeddings
    x = params["wte"][tokens] + params["wpe"][:T]

    # Pre-LN Transformer blocks: Attention followed by MLP
    for layer in range(n_layers):
        x = attention_block(x, params, layer=layer, n_heads=n_heads)
        x = mlp_block(x, params, layer=layer)

    # Final LayerNorm
    d = x.shape[-1]
    lnf_w = params["lnf_w"]
    lnf_b = params["lnf_b"]
    h = F.layer_norm(x, (d,), weight=lnf_w, bias=lnf_b, eps=1e-5)

    return h

# Step 12 - output_head
def output_head(h, params: dict):
    # TODO: Linear map from hidden states to logits over the vocabulary.
    return h @ params["head_w"] + params["head_b"]

# Step 13 - next_token_loss
import torch
import torch.nn.functional as F


def next_token_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked mean cross-entropy loss from logits; returns 0.0 tensor if mask is empty.

    Args:
        logits: (B, T, V) unnormalized prediction scores.
        targets: (B, T) target token indices.
        mask: (B, T) boolean or numeric mask (True/1 for valid tokens).

    Returns:
        Scalar tensor with the mean cross-entropy loss over masked tokens.
    """
    total_valid = mask.sum()
    if total_valid == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    V = logits.shape[-1]
    # Compute per-token loss without reduction: shape (B * T,)
    loss_flat = F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), reduction="none"
    )

    # Flatten mask to align with loss_flat
    mask_flat = mask.reshape(-1)

    return (loss_flat * mask_flat).sum() / total_valid

# Step 14 - init_dynamics_params
import torch


def init_dynamics_params(
    d_model: int, hidden: int, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Allocate parameters W1, b1, W2, b2, W3, b3 for a 3-layer MLP dynamics model.

    Inputs are concatenated (h_t, a_t) of dimension 2 * d_model, mapped to d_model.
    Weights are initialized with std 0.02 and biases with zeros, all with requires_grad=True.
    """
    torch.manual_seed(seed)

    def _rand_mat(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, dtype=torch.float32) * 0.02).requires_grad_(
            True
        )

    def _zeros(*shape: int) -> torch.Tensor:
        return torch.zeros(*shape, dtype=torch.float32, requires_grad=True)

    return {
        "W1": _rand_mat(2 * d_model, hidden),
        "b1": _zeros(hidden),
        "W2": _rand_mat(hidden, hidden),
        "b2": _zeros(hidden),
        "W3": _rand_mat(hidden, d_model),
        "b3": _zeros(d_model),
    }

# Step 15 - latent_transition
import torch
import torch.nn.functional as F


def latent_transition(
    h: torch.Tensor, x_emb: torch.Tensor, dyn: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Predict the next latent belief state using a residual 3-layer MLP dynamics model.

    Computes:
      z = LayerNorm(concat([h, x_emb], dim=-1))
      a1 = GELU(z @ W1 + b1)
      a2 = GELU(a1 @ W2 + b2)
      delta = a2 @ W3 + b3
      return h + delta
    """
    # 1. Concatenate current latent state and action/token embedding
    z = torch.cat([h, x_emb], dim=-1)

    # 2. Input LayerNorm
    z = F.layer_norm(z, (z.shape[-1],), eps=1e-5)

    # 3. 3-layer MLP with GELU (tanh approximation)
    a1 = F.gelu(z @ dyn["W1"] + dyn["b1"], approximate="tanh")
    a2 = F.gelu(a1 @ dyn["W2"] + dyn["b2"], approximate="tanh")
    delta = a2 @ dyn["W3"] + dyn["b3"]

    # 4. Residual update
    return h + delta

# Step 16 - rollout_latents
import torch


def rollout_latents(
    h: torch.Tensor,
    x: torch.Tensor,
    params: dict[str, torch.Tensor],
    dyn: dict[str, torch.Tensor],
    d_steps: int,
) -> list[torch.Tensor]:
    """Perform recursive d_steps rollout using the dynamics model.

    At step i (0 <= i < d_steps):
      - Step 0 predicts h_{t+1} from h_t and x_{t+1}: slice x[:, 1 : T - d_steps + 1]
      - Step i predicts h_{t+1+i} from h_hat and x_{t+1+i}: slice x[:, 1 + i : T - d_steps + 1 + i]
    """
    B, T, d_model = h.shape

    # Base hidden states h_t for t in [0, T - d_steps)
    h_hat = h[:, : T - d_steps]
    out = []

    for i in range(d_steps):
        # Action/token that transitions state t+i to t+1+i
        token_slice = x[:, 1 + i : T - d_steps + 1 + i]
        emb = params["wte"][token_slice]

        # Transition forward
        h_hat = latent_transition(h_hat, emb, dyn)
        out.append(h_hat)

    return out

# Step 17 - next_hidden_loss
import torch
import torch.nn.functional as F


def next_hidden_loss(
    h: torch.Tensor,
    h_hats: list[torch.Tensor],
    mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Compute the stop-gradient Smooth L1 loss between rolled-out latents and true hidden states.

    Args:
        h: (B, T, d) ground-truth hidden states from the backbone.
        h_hats: List of rolled-out latents from rollout_latents, length d_steps.
        mask: (B, T) bool tensor indicating valid positions.
        beta: Threshold for Smooth L1 loss.

    Returns:
        Scalar 0-dim tensor containing the average loss across all steps.
    """
    d_steps = len(h_hats)
    if d_steps == 0:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype)

    B, T, d = h.shape
    step_losses = []

    for idx, h_hat in enumerate(h_hats):
        # 1-based step index i: 1, 2, ..., d_steps
        i = idx + 1

        # Target hidden state slice with stop-gradient
        target = h[:, i : T - d_steps + i].detach()
        # Corresponding mask slice
        m = mask[:, i : T - d_steps + i]

        # Elementwise smooth L1 loss: (B, T - d_steps, d)
        loss = F.smooth_l1_loss(h_hat, target, beta=beta, reduction="none")

        # Average over the feature dimension d: (B, T - d_steps)
        loss = loss.mean(dim=-1)

        # Average over valid (masked) positions
        valid_count = m.sum()
        if valid_count > 0:
            step_loss = (loss * m).sum() / valid_count
        else:
            step_loss = torch.tensor(0.0, device=h.device, dtype=h.dtype)

        step_losses.append(step_loss)

    # Average across all rollout steps
    return torch.stack(step_losses).mean()

# Step 18 - kl_alignment_loss
import torch
import torch.nn.functional as F


def kl_alignment_loss(
    h: torch.Tensor,
    h_hats: list[torch.Tensor],
    mask: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute forward KL(true || predicted) in token space using a detached output head.

    Args:
        h: (B, T, d) true hidden states.
        h_hats: List of d_steps rolled-out latent tensors from rollout_latents.
        mask: (B, T) boolean or numeric mask of valid token positions.
        params: Model parameter dictionary containing 'head_w' and 'head_b'.

    Returns:
        Scalar 0-dim tensor with the average KL loss across rollout steps.
    """
    d_steps = len(h_hats)
    if d_steps == 0:
        return torch.tensor(0.0, device=h.device, dtype=h.dtype)

    T = h.shape[1]
    head_w = params["head_w"].detach()
    head_b = params["head_b"].detach()

    def _apply_head(x: torch.Tensor) -> torch.Tensor:
        # If an output_head helper exists in your scope, it can be called directly:
        # return output_head(x, {"head_w": head_w, "head_b": head_b})
        return x @ head_w + head_b

    step_losses = []

    for idx, h_hat in enumerate(h_hats):
        # 1-based step index i: 1, 2, ..., d_steps
        i = idx + 1

        # Slice target hidden states and mask
        h_true = h[:, i : T - d_steps + i].detach()
        m = mask[:, i : T - d_steps + i]

        # Logits under frozen output head
        logits_true = _apply_head(h_true)
        logits_pred = _apply_head(h_hat)

        # KL(P_true || Q_pred) = sum(P_true * (log P_true - log Q_pred))
        log_p_true = F.log_softmax(logits_true, dim=-1)
        log_q_pred = F.log_softmax(logits_pred, dim=-1)

        # kl_div with log_target=True computes exp(log_target) * (log_target - input)
        kl = F.kl_div(log_q_pred, log_p_true, log_target=True, reduction="none").sum(dim=-1)

        # Masked average over positions
        valid_count = m.sum()
        if valid_count > 0:
            step_loss = (kl * m).sum() / valid_count
        else:
            step_loss = torch.tensor(0.0, device=h.device, dtype=h.dtype)

        step_losses.append(step_loss)

    return torch.stack(step_losses).mean()

# Step 19 - nextlat_loss
import torch


def nextlat_loss(
    batch: dict,
    params: dict,
    dyn: dict,
    n_heads: int,
    d_steps: int,
    lam_h: float,
    lam_kl: float,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute the full NextLat training objective on a batch.

    Total loss:
        total = next_token + lam_h * next_h + lam_kl * kl

    Args:
        batch: Dict containing 'x', 'y', and 'mask'.
        params: GPT parameters dict.
        dyn: Dynamics model parameters dict.
        n_heads: Number of attention heads.
        d_steps: Rollout horizon for the latent dynamics model.
        lam_h: Weight for the next-hidden Smooth L1 loss.
        lam_kl: Weight for the token-space KL alignment loss.
        beta: Smooth L1 threshold parameter.

    Returns:
        Dict of 0-dim scalar tensors: 'total', 'next_token', 'next_h', 'kl'.
    """
    x = batch["x"]
    y = batch["y"]
    mask = batch["mask"]

    # 1. Run GPT backbone to obtain hidden states
    h = gpt_hidden_states(x, params, n_heads)

    # 2. Compute logits and next-token cross-entropy loss
    logits = output_head(h, params)
    loss_nt = next_token_loss(logits, y, mask)

    # 3. Latent dynamics rollouts and auxiliary losses
    if d_steps > 0:
        # EOS token ID corresponds to the last index in the vocabulary
        eos = params["head_b"].shape[0] - 1
        mask_x = x != eos

        # Single shared latent rollout
        h_hats = rollout_latents(h, x, params, dyn, d_steps)

        loss_h = next_hidden_loss(h, h_hats, mask_x, beta=beta)
        loss_kl = kl_alignment_loss(h, h_hats, mask_x, params)
    else:
        loss_h = torch.tensor(0.0, device=h.device, dtype=h.dtype)
        loss_kl = torch.tensor(0.0, device=h.device, dtype=h.dtype)

    # 4. Total weighted objective
    total_loss = loss_nt + lam_h * loss_h + lam_kl * loss_kl

    return {
        "total": total_loss,
        "next_token": loss_nt,
        "next_h": loss_h,
        "kl": loss_kl,
    }

# Step 20 - train_step
def train_step(batch: dict, params: dict, dyn: dict, opt, n_heads: int, d_steps: int,
               lam_h: float, lam_kl: float, beta: float = 1.0) -> dict:
    # TODO: zero_grad -> nextlat_loss -> backward on 'total' -> step; return the four losses as floats.
    opt.zero_grad()
    out = nextlat_loss(batch, params, dyn, n_heads, d_steps, lam_h, lam_kl, beta)
    out["total"].backward()
    opt.step()
    return {k: v.item() for k, v in out.items()}

# Step 21 - train_model
import torch


def train_model(dataset: dict, cfg: dict, seed: int = 0) -> tuple[dict, dict, list[dict]]:
    """Initialize model & dynamics, optimize with Adam over cyclic batches, and record history.

    Args:
        dataset: Dict containing 'tokens', 'mask', 'states', and 'G'.
        cfg: Configuration dictionary with hyperparameters:
             - 'd_model', 'n_layers', 'n_heads'
             - 'hidden' (for dynamics MLP)
             - 'lr', 'steps', 'batch_size'
             - 'd_steps', 'lam_h', 'lam_kl', and optionally 'beta'
        seed: Random seed for parameter initialization.

    Returns:
        tuple of (params, dyn, history), where history is a list of loss dicts
        (with float values) recorded at each step.
    """
    G = dataset["G"]
    # Vocabulary: 4 actions (0..3) + G*G cells + 1 EOS = 4 + G*G + 1
    vocab_size = 4 + G * G + 1
    max_len = dataset["tokens"].shape[1]

    # Initialize GPT and dynamics parameters
    params = init_gpt_params(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        max_len=max_len,
        seed=seed,
    )
    dyn = init_dynamics_params(
        d_model=cfg["d_model"],
        hidden=cfg["hidden"],
        seed=seed,
    )

    # Combine all trainable parameters into a single Adam optimizer
    all_params = list(params.values()) + list(dyn.values())
    optimizer = torch.optim.Adam(all_params, lr=cfg["lr"])

    history = []
    beta = cfg.get("beta", 1.0)

    for step in range(cfg["steps"]):
        optimizer.zero_grad()

        # Deterministic cyclic batch slice
        batch = get_batch(dataset, batch_size=cfg["batch_size"], step=step)

        # Compute full NextLat objective
        losses = nextlat_loss(
            batch=batch,
            params=params,
            dyn=dyn,
            n_heads=cfg["n_heads"],
            d_steps=cfg["d_steps"],
            lam_h=cfg["lam_h"],
            lam_kl=cfg["lam_kl"],
            beta=beta,
        )

        losses["total"].backward()
        optimizer.step()

        # Record scalar metric history
        history.append({k: v.detach().item() for k, v in losses.items()})

    return params, dyn, history

# Step 22 - greedy_decode
import torch


def greedy_decode(
    params: dict, n_heads: int, prefix: list[int], n_tokens: int
) -> list[int]:
    """Repeatedly run the transformer on the accumulated sequence and append the argmax token.

    Args:
        params: Model parameter dictionary.
        n_heads: Number of attention heads.
        prefix: Initial list of token IDs.
        n_tokens: Number of tokens to generate.

    Returns:
        List of generated tokens (length `n_tokens`).
    """
    seq = list(prefix)
    generated = []

    with torch.no_grad():
        for _ in range(n_tokens):
            x = torch.tensor([seq], dtype=torch.long)
            h = gpt_hidden_states(x, params, n_heads)
            # Obtain logits for the final sequence position: (1, vocab_size)
            logits = output_head(h[:, -1:], params)
            tok = int(torch.argmax(logits[0, -1]).item())

            seq.append(tok)
            generated.append(tok)

    return generated

# Step 23 - effective_rank
def effective_rank(H, tol: float = 1e-12) -> float:
    # TODO: exp of the Shannon entropy of the normalized singular values above tol.
    s = torch.linalg.svdvals(H)
    s = s[s>tol]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    H = -(p*p.log()).sum()
    return float(torch.exp(H))

# Step 24 - eval_hidden_states
import torch


def eval_hidden_states(
    dataset: dict, params: dict, n_heads: int, n_rows: int
) -> torch.Tensor:
    """Collect transformer hidden states at every real input position of the first n_rows sequences.

    Args:
        dataset: Dict containing 'tokens' and 'mask'.
        params: Model parameter dictionary.
        n_heads: Number of attention heads.
        n_rows: Number of sequences to evaluate.

    Returns:
        A detached (N, d) float tensor containing hidden states for valid positions.
    """
    x = dataset["tokens"][:n_rows, :-1]
    mask_x = dataset["mask"][:n_rows, :-1]

    with torch.no_grad():
        h = gpt_hidden_states(x, params, n_heads)
        return h[mask_x].detach()

# Step 25 - valid_move_rate
import torch


def valid_move_rate(
    dataset: dict, params: dict, n_heads: int, n_rows: int
) -> float:
    """Evaluate the fraction of top-1 predictions that are legal under the true world model.

    Evaluates across the first `n_rows` sequences at positions t >= 1 where `y_mask` is True.
    A prediction is legal if:
      - The walker has reached the goal and predicts EOS (= 4 + G*G), OR
      - The walker has not reached the goal and predicts a valid action in legal_actions(pos, G).

    Returns:
        Fraction of legal predictions as a float (0.0 if no positions scored).
    """
    G = dataset["G"]
    eos_token = 4 + G * G

    # Slice sequences and alignments for the first n_rows
    x = dataset["tokens"][:n_rows, :-1]
    y_mask = dataset["mask"][:n_rows, 1:]
    true_states = dataset["states"][:n_rows, :-1]

    # Goal cell index: tokens[:, 1] stores 4 + row * G + col
    goal_cells = dataset["tokens"][:n_rows, 1] - 4

    with torch.no_grad():
        h = gpt_hidden_states(x, params, n_heads)
        logits = output_head(h, params)
        preds = torch.argmax(logits, dim=-1)

    total_scored = 0
    legal_count = 0

    B, T_minus_1 = x.shape

    for b in range(B):
        goal = goal_cells[b].item()
        for t in range(1, T_minus_1):
            if not y_mask[b, t].item():
                continue

            pred = preds[b, t].item()
            curr_state = true_states[b, t].item()
            pos = (curr_state // G, curr_state % G)

            total_scored += 1
            if curr_state == goal:
                if pred == eos_token:
                    legal_count += 1
            else:
                if pred in legal_actions(pos, G):
                    legal_count += 1

    if total_scored == 0:
        return 0.0

    return float(legal_count / total_scored)

# Step 26 - sequence_compression
def sequence_compression(
    dataset: dict,
    params: dict,
    n_heads: int,
    n_tokens: int,
    max_pairs: int,
) -> float:
    """Compute the sequence compression metric.

    Evaluates the fraction of distinct prefix pairs that share the same true current
    state and goal cell for which greedy autoregressive decoding yields identical
    continuations.
    """
    tokens = dataset["tokens"]
    mask = dataset["mask"]
    states = dataset["states"]
    n_rows, T = tokens.shape

    # Collect prefixes mapped by key: (state_at_t, goal_cell_token)
    # Using a dict to preserve the first-seen order of keys.
    prefix_groups = {}

    for i in range(n_rows):
        goal_token = tokens[i, 1].item()
        for t in range(2, T):
            if not mask[i, t].item():
                continue
            if tokens[i, t].item() >= 4:
                continue
            if t + 1 + n_tokens > T:
                continue

            key = (states[i, t].item(), goal_token)
            prefix = tokens[i, : t + 1].tolist()

            if key not in prefix_groups:
                prefix_groups[key] = [prefix]
            else:
                existing = prefix_groups[key]
                if len(existing) < 2 and prefix != existing[0]:
                    existing.append(prefix)

    # Filter to keys that gathered at least two distinct prefixes
    valid_pairs = [prefixes for prefixes in prefix_groups.values() if len(prefixes) == 2]

    # Restrict to at most max_pairs in first-seen order
    eval_pairs = valid_pairs[:max_pairs]
    if not eval_pairs:
        return 0.0

    matching_pairs = 0
    for p1, p2 in eval_pairs:
        gen1 = greedy_decode(params, n_heads, p1, n_tokens)
        gen2 = greedy_decode(params, n_heads, p2, n_tokens)
        if gen1 == gen2:
            matching_pairs += 1

    return float(matching_pairs / len(eval_pairs))

# Step 27 - detour_robustness
import numpy as np


def detour_robustness(
    params: dict,
    n_heads: int,
    G: int,
    max_steps: int,
    n_trials: int,
    detour_prob: float = 0.75,
    seed: int = 0,
) -> float:
    """Evaluate detour robustness: fraction of episodes reaching the goal despite random perturbations.

    For each trial:
      - Samples start and goal cells.
      - At each step, if pos == goal, episode succeeds immediately.
      - With probability detour_prob, samples a random legal action (detour).
      - Otherwise, queries the model via greedy_decode for 1 token. If the model
        predicts an illegal move, the episode fails immediately.
      - Advances pos via grid_step and appends the action to seq.
      - After max_steps, succeeds only if pos == goal.

    Returns:
        Fraction of successful episodes as a float (0.0 if n_trials == 0).
    """
    if n_trials <= 0:
        return 0.0

    rng = np.random.default_rng(seed)
    successes = 0

    for _ in range(n_trials):
        start = tuple(rng.integers(0, G, size=2).tolist())
        goal = tuple(rng.integers(0, G, size=2).tolist())

        start_cell = 4 + start[0] * G + start[1]
        goal_cell = 4 + goal[0] * G + goal[1]
        seq = [start_cell, goal_cell]

        pos = start
        episode_success = False

        for _ in range(max_steps):
            if pos == goal:
                episode_success = True
                break

            allowed_actions = legal_actions(pos, G)

            if rng.random() < detour_prob:
                action = int(rng.choice(allowed_actions))
            else:
                pred_token = greedy_decode(params, n_heads, seq, 1)[0]
                if pred_token not in allowed_actions:
                    # Model attempted an illegal move
                    break
                action = pred_token

            pos, _ = grid_step(pos, action, G)
            seq.append(action)
        else:
            # Check if final step reached the goal
            if pos == goal:
                episode_success = True

        if episode_success:
            successes += 1

    return float(successes / n_trials)

# Step 28 - world_model_report
def world_model_report(
    dataset: dict,
    params: dict,
    n_heads: int,
    n_rows: int,
    n_tokens: int,
    max_pairs: int,
    n_trials: int,
    seed: int = 0,
) -> dict[str, float]:
    """Compute and report the four core world-model evaluation metrics.

    Metrics:
      - valid_move_rate: Fraction of top-1 predictions that are legal actions or EOS at goal.
      - effective_rank: Effective rank of the evaluation hidden states matrix.
      - sequence_compression: Fraction of matching greedy rollouts from equivalent (state, goal) states.
      - detour_robustness: Goal-reaching success rate under random action perturbations.

    Returns:
        Dictionary mapping metric names to floats rounded to 4 decimal places.
    """
    G = dataset["G"]
    T = dataset["tokens"].shape[1]

    # 1. Valid move rate
    vmr = valid_move_rate(dataset, params, n_heads=n_heads, n_rows=n_rows)

    # 2. Effective rank of hidden states
    h_states = eval_hidden_states(dataset, params, n_heads=n_heads, n_rows=n_rows)
    erank = effective_rank(h_states)

    # 3. Sequence compression
    seq_comp = sequence_compression(
        dataset, params, n_heads=n_heads, n_tokens=n_tokens, max_pairs=max_pairs
    )

    # 4. Detour robustness
    detour = detour_robustness(
        params,
        n_heads=n_heads,
        G=G,
        max_steps=T - 3,
        n_trials=n_trials,
        detour_prob=0.75,
        seed=seed,
    )

    return {
        "valid_move_rate": round(float(vmr), 4),
        "effective_rank": round(float(erank), 4),
        "sequence_compression": round(float(seq_comp), 4),
        "detour_robustness": round(float(detour), 4),
    }

# Step 29 - draft_from_latent
import torch


def draft_from_latent(
    h_last: torch.Tensor,
    dyn: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    max_draft: int,
) -> tuple[int, list[int]]:
    """Draft tokens by rolling latent dynamics forward from the last hidden state.

    Args:
        h_last: Hidden state tensor of shape (d,) representing the final sequence position.
        dyn: Parameter dictionary for the dynamics model.
        params: Parameter dictionary for the GPT model (wte, head_w, head_b, etc.).
        max_draft: Number of speculative/draft tokens to generate via dynamics rollout.

    Returns:
        tuple (next_token, drafts):
          - next_token: int, argmax prediction directly from h_last.
          - drafts: list of int, length max_draft, tokens generated by alternating
            output_head and latent_transition.
    """
    with torch.no_grad():
        # Ensure 2D shape (1, d) for compatibility with head and transition helpers
        h = h_last if h_last.ndim == 2 else h_last.unsqueeze(0)

        # 1. Base prediction from the backbone's final hidden state
        logits_0 = output_head(h, params)
        next_token = int(torch.argmax(logits_0, dim=-1).item())

        drafts: list[int] = []
        if max_draft <= 0:
            return next_token, drafts

        # 2. First transition consumes next_token embedding
        emb = params["wte"][next_token].unsqueeze(0)
        h = latent_transition(h, emb, dyn)

        # 3. Roll out max_draft speculative tokens
        for _ in range(max_draft):
            logits = output_head(h, params)
            tok = int(torch.argmax(logits, dim=-1).item())
            drafts.append(tok)

            emb = params["wte"][tok].unsqueeze(0)
            h = latent_transition(h, emb, dyn)

    return next_token, drafts

# Step 30 - verify_draft
import torch


def verify_draft(
    params: dict,
    n_heads: int,
    prefix: list[int],
    next_token: int,
    drafts: list[int],
) -> tuple[int, int]:
    """Verify speculative draft tokens in a single causal transformer pass.

    Constructs seq = prefix + [next_token] + drafts of length L.
    Position index layout:
      - indices 0 .. len(prefix) - 1 : prefix tokens
      - index len(prefix)            : next_token
      - index len(prefix) + 1 + j    : drafts[j] for j in 0..len(drafts)-1

    The prediction generated *at* position idx predicts the token at idx + 1:
      - preds[len(prefix)] predicts token at len(prefix) + 1 (i.e., drafts[0])
      - preds[len(prefix) + j] predicts token at len(prefix) + 1 + j (i.e., drafts[j])

    Args:
        params: Model parameter dictionary.
        n_heads: Number of attention heads.
        prefix: Initial context token list.
        next_token: The verified next token emitted from prefix.
        drafts: List of speculative draft tokens.

    Returns:
        tuple (n_accepted, correction):
          - n_accepted: Number of contiguous accepted draft tokens from the beginning.
          - correction: The transformer's true argmax prediction at the first divergence
                        (or the follow-on prediction if all drafts are accepted).
    """
    seq = prefix + [next_token] + drafts
    x = torch.tensor([seq], dtype=torch.long)

    with torch.no_grad():
        h = gpt_hidden_states(x, params, n_heads)
        logits = output_head(h[0], params)
        preds = torch.argmax(logits, dim=-1)

    n_accepted = 0
    prefix_len = len(prefix)

    for j, draft_tok in enumerate(drafts):
        # preds[prefix_len + j] is the prediction at the position right before drafts[j]
        pred_tok = int(preds[prefix_len + j].item())
        if pred_tok == draft_tok:
            n_accepted += 1
        else:
            break

    # Correction token is the model's true top-1 output immediately after the accepted prefix
    correction = int(preds[prefix_len + n_accepted].item())

    return n_accepted, correction

# Step 31 - self_speculative_generate
import torch


def self_speculative_generate(
    params: dict,
    dyn: dict,
    n_heads: int,
    prefix: list[int],
    n_tokens: int,
    max_draft: int,
) -> dict:
    """Generate n_tokens with variable-length self-speculative decoding.

    Args:
        params: GPT model parameter dictionary.
        dyn: Dynamics model parameter dictionary.
        n_heads: Number of attention heads.
        prefix: Initial list of prompt/prefix tokens.
        n_tokens: Number of tokens to generate.
        max_draft: Maximum number of speculative tokens per draft cycle.

    Returns:
        dict with:
          - 'tokens': list of generated token IDs (truncated to n_tokens).
          - 'cycles': int, number of speculative verify cycles executed.
          - 'accepted': list of int, number of accepted draft tokens per cycle.
    """
    seq = list(prefix)
    max_len = params["wpe"].shape[0]

    accepted_history = []
    cycles = 0

    while len(seq) - len(prefix) < n_tokens:
        cycles += 1

        with torch.no_grad():
            x = torch.tensor([seq], dtype=torch.long)
            h = gpt_hidden_states(x, params, n_heads)
            h_last = h[0, -1]

        # Ensure draft tokens fit within the positional embedding budget:
        # verifying seq + [next_token] + drafts has length len(seq) + 1 + k <= max_len - 1
        # (allowing space for the correction token within max_len)
        k = max(0, min(max_draft, max_len - len(seq) - 2))

        # Draft candidate tokens from the latent dynamics model
        next_token, drafts = draft_from_latent(h_last, dyn, params, k)

        # Single transformer pass verification
        n_accepted, correction = verify_draft(
            params, n_heads, seq, next_token, drafts
        )

        accepted_history.append(n_accepted)

        # Extend sequence with verified base token, accepted drafts, and correction token
        new_tokens = [next_token] + drafts[:n_accepted] + [correction]
        seq.extend(new_tokens)

    # Slice out generated tokens (excluding the initial prefix) and truncate to n_tokens
    generated = seq[len(prefix) : len(prefix) + n_tokens]

    return {
        "tokens": generated,
        "cycles": cycles,
        "accepted": accepted_history,
    }

# Step 32 - speculative_stats (not yet solved)
# TODO: implement

