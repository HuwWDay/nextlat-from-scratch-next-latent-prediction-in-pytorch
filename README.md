# NextLat from Scratch: Next-Latent Prediction in PyTorch

Build Next-Latent Prediction (NextLat, arXiv:2511.05963) end to end in functional PyTorch: a grid world whose true belief state is (position, goal), a tiny causal GPT, and a residual-delta latent dynamics model trained with a stop-gradient loss and a frozen-head KL term. Then measure the paper's world-model metrics and self-speculative decoding.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** grid_step
- [x] **2.** legal_actions
- [x] **3.** random_walk_to_goal
- [x] **4.** encode_sequence
- [x] **5.** make_dataset
- [x] **6.** get_batch
- [x] **7.** causal_mask
- [x] **8.** init_gpt_params
- [x] **9.** attention_block
- [x] **10.** mlp_block
- [x] **11.** gpt_hidden_states
- [x] **12.** output_head
- [x] **13.** next_token_loss
- [x] **14.** init_dynamics_params
- [x] **15.** latent_transition
- [x] **16.** rollout_latents
- [x] **17.** next_hidden_loss
- [x] **18.** kl_alignment_loss
- [x] **19.** nextlat_loss
- [x] **20.** train_step
- [x] **21.** train_model
- [x] **22.** greedy_decode
- [x] **23.** effective_rank
- [x] **24.** eval_hidden_states
- [x] **25.** valid_move_rate
- [x] **26.** sequence_compression
- [x] **27.** detour_robustness
- [x] **28.** world_model_report
- [x] **29.** draft_from_latent
- [x] **30.** verify_draft
- [x] **31.** self_speculative_generate
- [x] **32.** speculative_stats

---

Built on Deep-ML.
