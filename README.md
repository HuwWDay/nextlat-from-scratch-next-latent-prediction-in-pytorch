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
- [ ] **9.** attention_block
- [ ] **10.** mlp_block
- [ ] **11.** gpt_hidden_states
- [ ] **12.** output_head
- [ ] **13.** next_token_loss
- [ ] **14.** init_dynamics_params
- [ ] **15.** latent_transition
- [ ] **16.** rollout_latents
- [ ] **17.** next_hidden_loss
- [ ] **18.** kl_alignment_loss
- [ ] **19.** nextlat_loss
- [ ] **20.** train_step
- [ ] **21.** train_model
- [ ] **22.** greedy_decode
- [ ] **23.** effective_rank
- [ ] **24.** eval_hidden_states
- [ ] **25.** valid_move_rate
- [ ] **26.** sequence_compression
- [ ] **27.** detour_robustness
- [ ] **28.** world_model_report
- [ ] **29.** draft_from_latent
- [ ] **30.** verify_draft
- [ ] **31.** self_speculative_generate
- [ ] **32.** speculative_stats

---

Built on Deep-ML.
