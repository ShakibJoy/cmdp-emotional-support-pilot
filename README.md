# cmdp-emotional-support-pilot

**Paper:** [cmdp_emotional_support_pilot.pdf](cmdp_emotional_support_pilot.pdf) manuscript, 2026; 

Lagrangian-Constrained PPO with an LLM Safety-Cost Oracle for Emotional-Support Dialogue

**Summary.** TinyLlama-1.1B-Chat → rejection fine-tuning (RFT) → unconstrained vs.
Lagrangian-constrained PPO, with a live GPT-4o-mini safety-cost oracle, evaluated on 100
held-out ESConv prompts. Result: **no significant difference between systems** (Fisher's
exact p ≥ 0.36); the constraint was largely inactive (λ ≤ 0.04) because the cost limit
(0.30) sat above the policy's violation rate. See the paper for details.

> ⚠️ **Not for deployment.** These are research artifacts. They produce short,
> low-quality and sometimes unsafe responses and must not be used with people
> seeking mental health support.


## Repository contents
- `rft.ipynb` — RFT: candidate generation (500 prompts × 3) and LoRA fine-tuning
- `ppo unconstrain.ipynb` / `ppo constrain.ipynb` — critic-free PPO (`USE_CONSTRAINT = False / True`)
- `evaluation v1.ipynb` — first evaluation run (single attempt per prompt)
- `evaluation v2.ipynb` — main evaluation (retry-on-failure, up to 5 attempts)
- `diagnostic-run.ipynb` — untuned base-model failure diagnostic (19/30)
- `figures/` — paper figures
- `results/` — per-response evaluation outputs

## Reproducing
1. Download ESConv from https://github.com/thu-coai/Emotional-Support-Conversation
   (academic research use only) and build the triples with the script.
2. Model checkpoints and Kaggle dataset available on request.
3. Set `OPENAI_API_KEY` as an environment variable or Kaggle Secret.
4. Hardware: one NVIDIA T4 (16 GB). PPO takes ~3 h per run; total API cost was ~US$0.42.

Generation is unseeded, so re-running evaluation will give somewhat different numbers.

## License
Code: MIT. ESConv and Counsel Chat remain under their original terms.
