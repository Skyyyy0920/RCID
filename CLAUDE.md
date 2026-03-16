# CLAUDE.md — RCID Project (SaGD)

This file is the authoritative guide for AI assistants working on this codebase.
Read it completely before making any changes.

---

## 1. Project Overview

**Research goal**: Knowledge distillation from large to small LLMs.  
**Active method**: SaGD (Saliency-Guided Knowledge Distillation).  
**Paper title**: *Saliency-Guided Knowledge Distillation: A Sobolev Perspective on Teaching Students Where to Look*  
**Target venue**: Academic paper submission.

### Model pairs
- Primary: Qwen3-8B (teacher) → Qwen3-0.6B (student)
- Secondary (cross-architecture): LLaMA 3.1-8B → LLaMA 3.1-1B

### Dataset & evaluation
- Training data: `databricks/databricks-dolly-15k`, train split, seed=42, max_seq_len=512
- Primary metric: ROUGE-L on Dolly test-500
- Secondary metric: Mean JSD (Saliency Loyalty) — mean teacher/student saliency divergence on val-500
- Benchmark defense: MMLU, ARC-Challenge, TruthfulQA (lm-eval-harness, appendix only)

### Environment
- Server path: `/mnt/nvme1/tianhao/RCID`
- Conda env: `py310_torch24`
- Hardware: 4× A100 80GB
- Python: 3.10, PyTorch 2.4, Hugging Face Transformers

---

## 2. Repository Structure
```
src/rcid/
  data/
    instruction_dataset.py      # InstructionDataset — returns index field
  distillation/
    __init__.py                 # exports SaliencyComputer, SaliencyAlignmentLoss
    scalable_trainer.py         # ScalableDistillationTrainer — main training loop
    saliency.py                 # SaliencyComputer + SaliencyAlignmentLoss
    baselines.py                # StandardKDLoss, ReverseKLLoss
    adaptive_loss.py            # AKL, KLR (baseline methods)
    trainer.py                  # UnifiedTrainer (toy data, kept but not used for SaGD)
    # legacy files (kept but NOT used in main paths):
    ocid_loss.py                # OCID regularizer — DEPRECATED, do not route
    logit_lens_monitor.py       # Logit lens monitor — DEPRECATED, do not route
    adaptive_lambda.py          # Adaptive λ controller — DEPRECATED, do not route
  models/
    adapter.py                  # ModelAdapter abstraction (Qwen3, LLaMA 3)
    loader.py                   # Model loading utilities
  eval/
    rouge_eval.py               # ROUGE-L evaluation

scripts/
  run_large_scale_distill.py    # main CLI for single-method runs
  run_paper_experiments.py      # named experiment configs (ablations etc.)
  run_all_experiments.sh        # full experiment sweep
  precompute_teacher_saliency.py  # one-shot teacher saliency precomputation
  diagnose_saliency.py          # saliency divergence analysis tool

configs/
  large_scale.yaml              # primary experiment config

tests/
  test_saliency.py              # unit tests for SaliencyComputer etc.
  test_large_scale.py           # existing trainer tests (keep working)

data/
  teacher_saliency.pt           # precomputed teacher saliency cache (gitignored)
```

---

## 3. Active Method: SaGD

### 3.1 Theory

Standard KD minimizes $\mathcal{L}_\text{KD} = \frac{1}{n}\sum_i D_\text{KL}(f_T(x_i) \| f_S(x_i))$,
which only constrains function values at training points (zero-order / L² matching).
By Taylor expansion, the error at a perturbed input $x + \delta$ decomposes as:

$$D_\text{KL}(f_T(x+\delta) \| f_S(x+\delta)) \leq \underbrace{D_\text{KL}(f_T(x) \| f_S(x))}_\text{zero-order} + \epsilon \cdot \underbrace{\|J_T(x) - J_S(x)\|_F}_\text{first-order: Jacobian gap} + O(\epsilon^2)$$

SaGD adds first-order (Jacobian) matching, upgrading from L² to Sobolev W^{1,2} approximation.
Because $\epsilon^1 \gg \epsilon^2 \gg \cdots$ for $\epsilon < 1$, first-order is the highest-ROI additional signal.

The full Jacobian $J \in \mathbb{R}^{V \times (L \cdot d)}$ is intractable (~7×10¹⁰ elements).
**Input saliency** compresses it to a per-position scalar: $s_i = \|\partial \log P(\text{response}) / \partial e_i\|$.
By Cauchy-Schwarz, $\|s_T - s_S\|^2 \leq \|J_T - J_S\|_F^2$, so saliency distance is a
principled lower bound of Jacobian distance — not an arbitrary approximation.

**Saliency-guided reweighting** corresponds to distributionally robust optimization (DRO):
the minimax neighborhood error upper bound is $\text{KL}_i + \epsilon \cdot \text{sal\_div}_i$,
so prioritizing samples with high saliency divergence = tightening the loosest bounds first.

### 3.2 Complete Loss

$$\mathcal{L}_\text{SaGD} = \underbrace{\sum_{i=1}^B w_i \cdot D_\text{KL}(f_T(x_i) \| f_S(x_i))}_\text{weighted KL (zero-order + DRO)} + \lambda \cdot \underbrace{\frac{1}{B}\sum_{i=1}^B (1 - \cos(s_T^i, s_S^i))}_\text{saliency alignment loss (first-order)}$$

Sample weights:
$$w_i = \frac{\exp(\text{JSD}_i / \tau_w)}{\frac{1}{B}\sum_j \exp(\text{JSD}_j / \tau_w)} \quad \text{(mean = 1)}$$

where $\text{JSD}_i = \text{JSD}(\hat{s}_T^i \| \hat{s}_S^i)$, and $\hat{s} = \text{softmax}(s/\tau_s)$.

**Why cosine for alignment loss** (not KL or MSE):
- KL requires softmax normalization → discards magnitude information
- MSE is sensitive to scale differences between teacher/student (different model sizes)
- Cosine captures directional pattern ("which positions matter") independent of scale

**Why two components are complementary**:
- Saliency loss directly shrinks the first-order gap → tightens neighborhood error bounds for all samples
- Reweighting concentrates zero-order optimization on high-risk samples → ensures KL is small where it matters most
- Neither alone achieves both effects

### 3.3 Saliency Computation (Canonical Implementation)
```python
def compute(self, model, input_ids, attention_mask, labels_mask):
    with torch.enable_grad():
        # Temporarily disable all model parameter gradients.
        # CRITICAL: without this, response_ll.backward() accumulates gradients
        # into model parameters (W_q, W_k, etc.), polluting the training gradient.
        param_grad_states = {}
        for name, p in model.named_parameters():
            param_grad_states[name] = p.requires_grad
            p.requires_grad_(False)
        try:
            embed = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
            logits = model(inputs_embeds=embed, attention_mask=attention_mask).logits

            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_resp_mask = labels_mask[:, 1:].float()   # shift to align with logit[j] → token[j+1]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            response_ll = (token_lp * shift_resp_mask).sum()

            if response_ll.abs().item() < 1e-9:
                return torch.zeros(input_ids.shape[0], input_ids.shape[1], device=input_ids.device)

            response_ll.backward()
            saliency = embed.grad.norm(dim=-1)  # (B, L)
        finally:
            for name, p in model.named_parameters():
                p.requires_grad_(param_grad_states[name])

    # Mask: keep only PROMPT positions (labels_mask=0 AND attention_mask=1)
    # CRITICAL: (1 - labels_mask) alone incorrectly retains padding positions.
    prompt_mask = (1 - labels_mask).float() * attention_mask.float()
    saliency = saliency * prompt_mask
    return saliency  # (B, L), zero at response and padding positions
```

### 3.4 Per-Sample KL (Canonical Implementation)
```python
def _compute_per_sample_kl(self, t_logits, s_logits, labels_mask):
    T = self.config.get("temperature", 2.0)
    t_log = F.log_softmax(t_logits.float() / T, dim=-1)
    s_log = F.log_softmax(s_logits.float() / T, dim=-1)
    t_probs = t_log.exp()
    per_pos = (t_probs * (t_log - s_log)).sum(dim=-1)   # (B, L)
    # Shift mask to align: logit[j] predicts token[j+1]
    per_pos_shifted = per_pos[:, :-1]                    # (B, L-1)
    mask_f = labels_mask[:, 1:].float()                  # (B, L-1)
    per_sample = (per_pos_shifted * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp(min=1)
    return per_sample * (T * T)  # (B,)
```

### 3.5 Training Loop Logic (Pseudocode)
```
Pre-training (once, ~1h):
  precompute_teacher_saliency.py → data/teacher_saliency.pt

Each training step:
  1. Load batch (input_ids, attention_mask, labels_mask, index)
  2. Teacher forward → t_logits  (under torch.no_grad)
  3. Student forward → s_logits

  if method == "standard_kd_sagd" AND global_step % N == 0:
    4. per_sample_kl = _compute_per_sample_kl(t_logits, s_logits, labels_mask)
    5. student_sal = saliency_computer.compute(student, ...)
    6. teacher_sal = _get_cached_teacher_saliency(batch["index"])
    7. sal_loss, sal_stats = saliency_loss_fn(teacher_sal, student_sal, labels_mask)
    8. jsd = saliency_computer.divergence(teacher_sal, student_sal, labels_mask)
    9. weights = softmax(jsd / τ_w) * B   # mean=1
    10. loss = (weights.detach() * per_sample_kl).mean() + λ * sal_loss
  else:
    10. loss = standard KL loss (uniform weighting)

  11. loss.backward() → optimizer.step()
```

### 3.6 Teacher Saliency Cache

- Precomputed once before training: `scripts/precompute_teacher_saliency.py`
- Stored as `List[Tensor]`, each `(L_i,)` = **full sequence length** (prompt + response positions, response positions are 0)
- Retrieved by sample index during training and padded/trimmed to batch seq_len
- Must use identical dataset instantiation parameters (same split, seed, tokenizer) as training

### 3.7 Ablation Theory Correspondence

| Config | KL (zero-order) | Sal loss (first-order) | Reweight | Theoretical space |
|--------|-----------------|------------------------|----------|-------------------|
| Standard KD | uniform | — | — | L² |
| + Sal loss only | uniform | ✓ | — | W^{1,2} |
| + Reweight only | weighted | — | ✓ | L² + DRO |
| **SaGD (full)** | weighted | ✓ | ✓ | W^{1,2} + DRO |

---

## 4. Registered Methods
```python
_VALID_METHODS = {
    "standard_kd",           # Forward KL baseline
    "reverse_kl",            # Reverse KL baseline
    "standard_kd_akl",       # AKL (Wu et al., COLING 2025)
    "standard_kd_klr",       # KL-Ratio adaptive (our prior work)
    "standard_kd_sagd",      # SaGD (current main method)
}
```

**Do not add back** `standard_kd_ocid`, `standard_kd_ocid_adaptive` — those directions
are abandoned. Their source files are kept for reference but must not appear in any
active code path.

---

## 5. Hyperparameters

| Parameter | Symbol | Default | Notes |
|-----------|--------|---------|-------|
| Saliency loss weight | λ | 0.5 | High sensitivity — sweep [0.01, 0.1, 0.5, 1.0, 2.0] |
| Reweighting temperature | τ_w | 1.0 | High sensitivity — sweep [0.1, 0.5, 1.0, 2.0, 5.0] |
| Saliency normalization temp | τ_s | 2.0 | Low sensitivity |
| Saliency update frequency | N | 5 | Medium sensitivity — sweep [1, 3, 5, 10, 20] |
| KL temperature | T | 2.0 | Standard KD value, keep fixed |
| Training epochs | — | 3 | Fixed |
| Batch size | — | 8 | Fixed |
| Gradient accumulation | — | 4 | Effective batch = 32 |
| Learning rate | — | 2e-5 | Fixed |
| Seeds | — | [42, 123, 456] | All main experiments use 3 seeds |

**Ablation configurations** (`run_paper_experiments.py`):

| Config | λ | τ_w | Effect |
|--------|---|-----|--------|
| `sagd` (full) | 0.5 | 1.0 | Both components active |
| `sagd_loss_only` | 0.5 | 100.0 | τ_w≈∞ → weights≈uniform → only saliency loss |
| `sagd_reweight_only` | 0.0 | 1.0 | No saliency loss, only reweighting |

---

## 6. Known Bugs Fixed (Do Not Reintroduce)

### Bug 1 — Saliency backward pollutes model parameter gradients
**Root cause**: `response_ll.backward()` accumulates gradients into all `requires_grad=True` leaf tensors in the computation graph, including model parameters W_q, W_k, etc.  
**Fix**: Temporarily set all model parameters to `requires_grad=False` inside `compute()`, restore in `finally` block.  
**Location**: `src/rcid/distillation/saliency.py` → `SaliencyComputer.compute()`

### Bug 2 — Saliency not masked at padding positions
**Root cause**: `(1 - labels_mask)` is 1 for both prompt tokens AND padding tokens.  
**Fix**: `prompt_mask = (1 - labels_mask).float() * attention_mask.float()`  
**Location**: `src/rcid/distillation/saliency.py` → `SaliencyComputer.compute()`

### Bug 3 — Per-sample KL mask off by one
**Root cause**: `logit[j]` predicts `token[j+1]`, but old code used `labels_mask[:, j]` (token j's label) instead of `labels_mask[:, j+1]`.  
**Fix**: Use `per_pos[:, :-1]` aligned with `labels_mask[:, 1:]`.  
**Location**: `src/rcid/distillation/scalable_trainer.py` → `_compute_per_sample_kl()`

### Bug 4 — Teacher/student saliency length misalignment
**Root cause**: Teacher saliency cached at natural length L_i; student saliency computed at padded batch length L_batch. If teacher cache stores only prompt slice (not full sequence), zero-padding produces incorrect alignment.  
**Fix**: Cache stores full-sequence-length saliency (L_i = prompt + response, response positions = 0). Pad/trim to L_batch in `_get_cached_teacher_saliency()`.  
**Location**: `scripts/precompute_teacher_saliency.py`, `scalable_trainer.py` → `_get_cached_teacher_saliency()`

---

## 7. Critical Implementation Constraints

1. **Baseline isolation**: When `method != "standard_kd_sagd"`, zero SaGD components are initialized. Baselines must run identically to before SaGD was added.

2. **Teacher is always frozen**: Teacher model stays in `eval()` mode with no gradient updates throughout training. `SaliencyComputer.compute()` must never modify teacher parameters.

3. **Student saliency is gradient-independent**: The saliency backward must not affect the main training gradient. The `param_grad_states` save/restore pattern is the canonical way to achieve this.

4. **Saliency is pre-masked**: `SaliencyComputer.compute()` always returns saliency with response and padding positions zeroed. Downstream functions (`SaliencyAlignmentLoss`, `divergence`) must NOT apply additional masking — they receive pre-masked vectors.

5. **Dataset index required**: `InstructionDataset.__getitem__` must return `"index": torch.tensor(idx, dtype=torch.long)`. The `collate_fn` must stack it with `torch.stack([b["index"] for b in batch])`. This field is silently ignored by non-SaGD methods.

6. **Precompute/train alignment**: `precompute_teacher_saliency.py` must use identical `data_source`, `dataset_seed`, `max_seq_len`, and tokenizer as training. Any mismatch silently corrupts the index→saliency mapping.

7. **fp16 compatibility**: SaGD loss components are computed inside the fp16 autocast context. Saliency computation uses `float()` casts where precision matters.

8. **Weights are detached**: Sample weights `w_i` must be `.detach()`ed before computing `weighted_kl`. Gradients must not flow through the weights.

---

## 8. Abandoned Directions (Context Only)

These directions were explored and abandoned. Their files remain in the repo for reference.
Do not route to them in any active code.

| Direction | Why abandoned |
|-----------|---------------|
| **RCID** (Residual Causal Imprint Distillation) | Underperforms standard KD by 1–3 ROUGE-L points across all benchmarks, regardless of hyperparameters. Root cause unresolved. |
| **KL-RCD** (KL-Ratio Guided Contrastive Distillation) | Key experimental flaw: FKL baseline trained on gold responses while KL-RCD trained on teacher-generated outputs, making comparison invalid. Inverted alpha mapping also gave weak gradients when strong ones were needed. |
| **OCID** (Output-level Causal Imprint Distillation) | Simplification of RCID at output layer. Superseded by SaGD before empirical validation. |
| **Logit lens health monitoring** | lm_head probe co-adapts with the final layer, causing the metric to decline artifactually. Signal is unreliable; adaptive lambda control based on it was abandoned. |
| **Circuit-level distillation** (attention head granularity) | Not feasible across architectures at large scale. |

---

## 9. Related Work Summary

| Paper | Method | Setting | Key difference from SaGD |
|-------|--------|---------|--------------------------|
| AD-KD (Wu et al., ACL 2023) | IG attribution alignment as loss | BERT classification | Encoder-only, classification, MSE on raw attribution, no reweighting, no theory |
| GKD (Wang et al., 2022) | Input gradient MSE alignment | BERT classification | Requires same d_model (full gradient vector, not compressed to saliency), no reweighting, encoder-only |
| GKD (Agarwal et al., 2024) | Generalized JSD + on-policy distillation | LLM generation | Zero-order only (various divergences on output distributions), no first-order matching |
| DA-KD (ICML 2025) | Difficulty-adaptive reweighting | LLM generation | Reweights by output KL divergence (zero-order signal), not saliency (first-order signal) |
| TSD (2026) | KL on softmax-normalized saliency | Time series classification | KL on normalized distributions (loses magnitude), no reweighting, not LLM |
| Sobolev Training (Czarnecki, NeurIPS 2017) | Full Jacobian matching in training | Small regression/Atari | Not in KD setting, full Jacobian intractable for LLMs |
| Srinivas & Fleuret (ICML 2018) | Proves Jacobian matching ≈ Gaussian input noise | CNN image classification | Complementary theoretical result; we provide saliency compression + reweighting |
| Ballout et al. (2024) | Teacher saliency extracts top-K tokens as rationales for student | T5 classification/QA | Uses saliency for data augmentation (rationale text), not as loss or reweighting signal |

**SaGD's genuine novelty**: (1) decoder-only LLM instruction distillation setting, (2) saliency-based sample reweighting (no prior work uses attribution divergence for reweighting), (3) cosine alignment (not KL/MSE), (4) loss + reweighting dual channel, (5) Sobolev/Taylor + DRO theoretical framework unifying prior work.

---

## 10. Experiment Design

### 10.1 Experiment checklist
```
Phase 0  Precompute teacher saliency         1 GPU  ~1h     MUST run before any SaGD training
Phase 1  Exp 1: Saliency divergence diagnosis 1 GPU  ~1h    motivation data for paper §4.2
Phase 2  Exp 2: Main table (5 methods × 3 seeds = 15 runs)  4 GPU ~8h wall
Phase 3  Exp 3: Ablations (~20 runs)          4 GPU  ~10h wall
Phase 4  Exp 4–5: Dynamics + analysis         1 GPU  ~3h
Phase 5  Exp 6: Cross-arch LLaMA (optional)   1 GPU  ~4h
Phase 6  Exp 7: Benchmark defense (appendix)  1 GPU  ~2h
```

### 10.2 Main table methods

| Method | CLI | Theory |
|--------|-----|--------|
| Standard KD | `--method standard_kd` | Zero-order (L²) |
| Reverse KL | `--method reverse_kl` | Zero-order (mode-seeking) |
| AKL | `--method standard_kd_akl` | Zero-order (adaptive mix) |
| KLR-Batch | `--method standard_kd_klr` | Zero-order (adaptive mix) |
| **SaGD (ours)** | `--method standard_kd_sagd` | **Zero-order + first-order (W^{1,2} + DRO)** |

### 10.3 Evaluation metrics

**ROUGE-L** (primary): Student greedy decode on Dolly test-500 → ROUGE-L F-score vs ground truth.

**Saliency Loyalty / Mean JSD** (saliency metric): For each checkpoint, compute mean teacher/student saliency JSD on val-500. Lower = student attends to same input tokens as teacher.

**Knowledge Benchmarks** (defense, appendix): MMLU, ARC-Challenge, TruthfulQA via lm-eval-harness. Confirms SaGD does not degrade general capabilities.

### 10.4 Paper structure mapping
```
§1 Introduction
§2 Background (KD, Input Saliency, Related Work)
§3 Method
  3.1 KD as Function Approximation: The L² Perspective
  3.2 Beyond Pointwise Matching: Taylor Expansion and Sobolev Norms
  3.3 Saliency as Tractable First-Order Approximation
  3.4 SaGD: Saliency Alignment Loss + Saliency-Guided Reweighting
  3.5 Complete Algorithm
§4 Experiments
  4.1 Setup                                          ← Exp 0
  4.2 Motivation: Does Standard KD Preserve Saliency? ← Exp 1
  4.3 Main Results                                    ← Exp 2
  4.4 Ablation Study                                  ← Exp 3
  4.5 Training Dynamics                               ← Exp 4
  4.6 Analysis: When Does the Student Look Wrong?     ← Exp 5
  4.7 Cross-Architecture Generalization (optional)    ← Exp 6
§5 Discussion & Limitations
Appendix: Proofs, Benchmark Defense (Exp 7), Hyperparameter Sensitivity, Visualizations
```

---

## 11. Quick Commands
```bash
# Activate environment
conda activate py310_torch24
cd /mnt/nvme1/tianhao/RCID

# Precompute teacher saliency (run once before any SaGD experiment)
python scripts/precompute_teacher_saliency.py \
    --model_name Qwen/Qwen3-8B \
    --data_source "databricks/databricks-dolly-15k" \
    --output_path data/teacher_saliency.pt \
    --batch_size 4 --max_seq_len 512 --device cuda:0

# Validate cache
python -c "
import torch; c = torch.load('data/teacher_saliency.pt', weights_only=False)
s0 = c['saliency'][0]
print(f'n={len(c[\"saliency\"])}, s0: shape={s0.shape}, max={s0.max():.4f}, nonzero={(s0>0).sum()}/{s0.shape[0]}')
"

# Smoke test: standard KD baseline
python scripts/run_large_scale_distill.py \
    --method standard_kd --epochs 1 --max_train_samples 200 \
    --device cuda:0 --skip_eval

# Smoke test: SaGD
python scripts/run_large_scale_distill.py \
    --method standard_kd_sagd \
    --teacher_saliency_path data/teacher_saliency.pt \
    --lambda_sal 0.5 --sagd_every_n_steps 5 \
    --epochs 1 --max_train_samples 200 \
    --device cuda:0 --skip_eval

# Unit tests
pytest tests/test_saliency.py -v

# Main experiment (seed 42, 4 GPUs)
OUTPUT=outputs/exp2
python scripts/run_large_scale_distill.py --method standard_kd     --seed 42 --output_dir $OUTPUT --device cuda:0 &
python scripts/run_large_scale_distill.py --method reverse_kl      --seed 42 --output_dir $OUTPUT --device cuda:1 &
python scripts/run_large_scale_distill.py --method standard_kd_akl --seed 42 --output_dir $OUTPUT --device cuda:2 &
python scripts/run_large_scale_distill.py --method standard_kd_klr --klr_granularity batch --klr_beta 0.99 \
    --seed 42 --output_dir $OUTPUT --device cuda:3 &
wait
python scripts/run_large_scale_distill.py --method standard_kd_sagd \
    --teacher_saliency_path data/teacher_saliency.pt \
    --lambda_sal 0.5 --sagd_every_n_steps 5 \
    --seed 42 --output_dir $OUTPUT --device cuda:0

# Saliency diagnosis on a trained checkpoint
python scripts/diagnose_saliency.py \
    --student_ckpt outputs/exp2/standard_kd/seed_42/student_final.pt \
    --teacher_saliency_path data/teacher_saliency.pt \
    --output_path outputs/exp2/standard_kd/seed_42/saliency_diagnosis.json \
    --device cuda:0
```

---

## 12. What NOT to Do

- **Do not** use logit lens as an adaptive training signal — it co-adapts with the final layer and is unreliable.
- **Do not** compute full Jacobians — they are $O(V \times L \times d) \approx 7\times10^{10}$ elements, intractable.
- **Do not** use KL divergence for saliency alignment loss — it requires softmax normalization which discards magnitude information.
- **Do not** use MSE for saliency alignment loss — it is sensitive to scale differences between teacher and student (different parameter counts).
- **Do not** restore the `standard_kd_ocid` or `standard_kd_ocid_adaptive` method routes.
- **Do not** let the saliency backward touch model parameter gradients — always use the `param_grad_states` save/restore pattern.
- **Do not** mask only with `(1 - labels_mask)` in saliency computation — always multiply by `attention_mask` to exclude padding.
- **Do not** align KL mask without the shift — use `labels_mask[:, 1:]` to pair with `per_pos[:, :-1]`.
- **Do not** let reweighting weights carry gradients — always `.detach()` before multiplying with per-sample KL.
- **Do not** assume teacher saliency cache and training data are aligned without verification — any mismatch in data_source, seed, max_seq_len, or tokenizer silently corrupts results.
