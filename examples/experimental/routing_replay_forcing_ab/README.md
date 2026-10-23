# Routing replay forcing A/B

> **Test-only, not for merge.** This directory holds the scenario, test agent, engine smoke and plotting scripts
> behind the evidence in `radixark/miles#<RFC>`, so anyone can rerun it.

In multi-turn session RL, a weight sync makes SGLang re-prefill tokens a session already holds, and the new weights can
pick different MoE routes for them, so rollout routing replay (R3) trains those tokens on routes that did not sample
them. Routing replay forcing makes the engine reuse the routes it first computed for every held token. This A/B runs one
fully async multi-turn scenario twice, forcing **OFF** and **ON**, and compares the train/rollout log-prob gap.

| Part | PR |
|---|---|
| SGLang: hold and force routes per session | `sgl-project/sglang#<S1>` |
| Router: forward `routed_experts_session_key` | `radixark/sgl-router-for-miles#<G1>` |
| Miles: `--use-rollout-routing-replay-forcing` | `radixark/miles#<M1>` (this branch builds on it) |

Files: `run_ab.py` (one arm), `agent.py` (its agent), `engine_smoke.py` (engine checks), `plot_results.py` (figure and
summary), `plot_diff.py` (paired-difference figure).

## Requirements

- SGLang built from the SGLang PR and a router (`sglang_router`) built from the router PR. Without the router change
  every ON turn is refused, because the router drops `routed_experts_session_key`.
- HF checkpoints under `--model-dir` (default `/root/models`). For `GLM-5.3-Flash-4layer`, `run_ab.py prepare` adds
  `processor_config.json` from `zai-org/GLM-5.3-Flash`.

## Scenario

`train_async.py --fully-async` on separate trainer and rollout GPUs: one policy update and one weight sync
(`--pause-generation-mode retract`) per step, samples up to 4 weight versions stale, session server, R3, `triton` MoE
runner, radix cache off, TIS. The agent works dapo-math-17k problems for at least 4 turns. 32 prompts x 8 samples per
step, all in flight; 2,048 tokens per turn, 32,768 per session. ON adds `--use-rollout-routing-replay-forcing` with a
store that holds every in-flight session at full context (`--sglang-routed-experts-forcing-max-tokens 8388608`, about
7.6 GiB of host memory per TP rank for Qwen3.8-Flash-Next). With forcing on, the session server also requests only new
R3 rows under `retract` (radixark/miles#3307), so ON differs from OFF in that too.

## Stages

Run from the repository root; each stage gates the next.

```bash
# 0. Data, the 4-layer checkpoint's missing processor config and its torch_dist conversion, 8 GPUs.
python examples/experimental/routing_replay_forcing_ab/run_ab.py prepare --model-name GLM-5.3-Flash-4layer
# 1. Engine smoke, 4 GPUs. For Qwen3.8-Flash-Next-4layer (the A/B model's layout): prefix
#    SGLANG_DISABLE_MULTIMEM_AG=1, use --tito-model qwen4exp, and replace the --dsa-* and --kv-cache-dtype flags with
#    --linear-attn-prefill-backend flashinfer.
python -m examples.experimental.routing_replay_forcing_ab.engine_smoke \
    --model-path /root/models/GLM-5.3-Flash-4layer --tito-model glm53 \
    --engine-args="--tp-size 4 --ep-size 4 --moe-runner-backend triton --mem-fraction-static 0.7 \
        --chunked-prefill-size 8192 --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
        --kv-cache-dtype bfloat16"
# 2. The SGLang PR's GPU test, 2 GPUs, in the SGLang checkout.
python -m pytest test/registered/e2e/rl/test_routed_experts_forcing_retract.py
# 3. The Miles e2e test, 2 GPUs.
python -m pytest tests/e2e/sglang/test_session_server_routing_replay_forcing.py
# 4. 4-layer RL, 8 GPUs per arm, 10 steps; then the same with --arm on --run-id ab-glm-4layer-on. Checks plumbing
#    only (near-random text).
python examples/experimental/routing_replay_forcing_ab/run_ab.py train --arm off --run-id ab-glm-4layer-off \
    --model-name GLM-5.3-Flash-4layer --trainer-gpus 4 --rollout-gpus 4 --num-steps 10
# 5. The A/B: Qwen3.8-Flash-Next, 100 steps, each arm on its own 48 GPUs (32 trainer, 16 rollout) with Ray already
#    started on them; the same with --arm on --run-id ab-on and tee on.log.
python examples/experimental/routing_replay_forcing_ab/run_ab.py prepare
MILES_SCRIPT_EXTERNAL_RAY=1 python examples/experimental/routing_replay_forcing_ab/run_ab.py train \
    --arm off --run-id ab-off --num-steps 100 --save <checkpoint dir> --save-interval 25 2>&1 | tee off.log
```

## Pass and stop rules (set before the runs)

- **Pass.** The mean per-step ON - OFF difference in `train/train_rollout_logprob_abs_diff` has a 95% interval below
  0 (resampled in blocks of 5 steps: with samples up to 4 versions stale, nearby steps are correlated);
  `train/train_rollout_kl` and `train/tis_clipfrac` are no worse in ON; weight-sync time is within 2%.
- **Stop ON** if any turn reports `unforced_tokens > 0`, or a step refuses more than 5 turns (2% of its 256 sessions)
  for 3 steps in a row. **Stop both** on a NaN or inf loss, grad norm or gap, or a gap above 3x its first-10-step
  median for 3 steps in a row.
- **Extend** both arms to 200 steps if the ON - OFF difference over steps 75-99 lies outside the 95% interval of steps
  50-74.

## Plotting

```bash
# From W&B instead: --source wandb --wandb-project <project> --off ab-off --on ab-on
python examples/experimental/routing_replay_forcing_ab/plot_results.py --source local \
    --off off.log --on on.log --out-dir ab-results \
    --title "Routing replay forcing, OFF vs ON (Qwen3.8-Flash-Next, 100 steps)"
python examples/experimental/routing_replay_forcing_ab/plot_diff.py \
    --off off.log --on on.log --out-dir ab-results --subtitle Qwen3.8-Flash-Next
```

`plot_results.py` writes `fig-3-off-vs-on.png` (gap, KL, TIS clip fraction, retracted, forced and refused turns, dropped
groups, weight-sync time, reward) and `summary.json` (the interval overall and per 25 steps, and any stop rule that
fired). `plot_diff.py` reads saved logs only and writes `fig-4-on-minus-off.png`: per step, ON - OFF in gap, TIS clip
fraction, reward and response length, and ON/OFF in gap and KL. Every panel has a 10-step trailing mean; the ON - OFF
gap panel also has the running mean and its 95% interval.
