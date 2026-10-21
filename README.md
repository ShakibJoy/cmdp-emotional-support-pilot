# cmdp-emotional-support-pilot

# Lagrangian-Constrained PPO with an LLM Safety-Cost Oracle for Emotional-Support Dialogue

Code, prompts, and evaluation outputs for the pilot study:

> Shakib Taher Joy. Lagrangian-Constrained PPO with an LLM Safety-Cost Oracle for
> Emotional-Support Dialogue: A Small-Model Pilot Study.

**Summary.** TinyLlama-1.1B-Chat → rejection fine-tuning (RFT) → unconstrained vs.
Lagrangian-constrained PPO, with a live GPT-4o-mini safety-cost oracle, evaluated on 100
held-out ESConv prompts. Result: **no significant difference between systems** (Fisher's
exact p ≥ 0.36); the constraint was largely inactive (λ ≤ 0.04) because the cost limit
(0.30) sat above the policy's violation rate. See the paper for details.

> ⚠️ **Not for deployment.** These are research artifacts. They produce short,
> low-quality and sometimes unsafe responses and must not be used with people
> seeking mental health support.

## Results (evaluation run 2, retry-on-failure)

| System | Reward | Violations | Empathy | Approp. | Hard failures |
|---|---|---|---|---|---|
| RFT | −2.680 | 11.3% (8/71) | 2.31 | 2.55 | 29/100 |
| PPO unconstrained | −2.680 | 16.3% (13/80) | 2.23 | 2.48 | 20/100 |
| PPO constrained | −2.545 | 11.0% (8/73) | 2.40 | 2.55 | 27/100 |

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
   (academic research use only) and build the triples with [script, or "see paper §3.1"].
2. Model checkpoints and Kaggle dataset available on request.
3. Set `OPENAI_API_KEY` as an environment variable or Kaggle Secret.
4. Hardware: one NVIDIA T4 (16 GB). PPO takes ~3 h per run; total API cost was ~US$0.42.

Generation is unseeded, so re-running evaluation will give somewhat different numbers
(see Appendix B of the paper).



## License
Code: MIT. ESConv and Counsel Chat remain under their original terms.
