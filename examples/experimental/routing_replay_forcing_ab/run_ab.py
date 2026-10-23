"""Routing replay forcing A/B: one arm, forcing off or on, of a fully async multi-turn GRPO run. Test-only.

Both arms run the scenario in README.md and differ only in --arm: `on` adds --use-rollout-routing-replay-forcing and
the size of the engines' route store. Model flags come from scripts/models/<megatron model type>.py and the model's
launcher under scripts/. A multi-node run needs Ray already started on its nodes and MILES_SCRIPT_EXTERNAL_RAY=1, as
for the scripts/ launchers.

    python examples/experimental/routing_replay_forcing_ab/run_ab.py prepare --model-name GLM-5.3-Flash-4layer
    python examples/experimental/routing_replay_forcing_ab/run_ab.py train --arm on \\
        --model-name GLM-5.3-Flash-4layer --trainer-gpus 4 --rollout-gpus 4 --num-steps 10
"""

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

# One rollout batch, 32 prompts x 8 samples, is one training step and is generated all at once.
_PROMPTS_PER_STEP = 32
_SAMPLES_PER_PROMPT = 8
_STEP_SAMPLES = _PROMPTS_PER_STEP * _SAMPLES_PER_PROMPT
_TURN_TOKENS = 2048
_CONTEXT_TOKENS = 32768
# Holds every in-flight session at full context, so the store's LRU never evicts a live session (which would refuse
# its next turn in ON only). Host memory per TP rank: tokens x (MoE layers x top-k x 2 B + 8 B), about 7.6 GiB for
# Qwen3.8-Flash-Next.
_FORCING_STORE_TOKENS = _STEP_SAMPLES * _CONTEXT_TOKENS
_AGENT = "examples.experimental.routing_replay_forcing_ab.agent"
# Read by agent.py.
_AGENT_MIN_TURNS_ENV = "ROUTING_REPLAY_AB_MIN_TURNS"
_AGENT_IGNORE_EOS_ENV = "ROUTING_REPLAY_AB_IGNORE_EOS"


@dataclass(frozen=True)
class _Recipe:
    megatron_model_type: str
    tito_model: str
    tensor_parallel: int
    pipeline_parallel: int
    engine_gpus: int
    # Megatron flags beyond scripts/models/<megatron_model_type>.py, from the family's launcher.
    trainer_args: str
    # SGLang flags from the same launcher.
    sglang_args: str
    # The same launcher's --check-weight-update-skip-list.
    weight_check_skip: str
    engine_env: tuple[tuple[str, str], ...] = ()
    # Environment for `prepare`'s checkpoint conversion.
    convert_env: tuple[tuple[str, str], ...] = ()
    # HF repository to take processor_config.json from when the checkpoint lacks it (SGLang loads it for this
    # multimodal model).
    processor_config_from: str | None = None
    # A 4-layer checkpoint writes near-random text, so the agent runs every reply to the turn cap.
    four_layer: bool = False


# scripts/run_qwen3_8_next.py
_FLASH_NEXT = _Recipe(
    megatron_model_type="qwen3.8-flash-next",
    tito_model="qwen4exp",
    tensor_parallel=2,
    pipeline_parallel=8,
    engine_gpus=8,
    trainer_args=(
        "--model-name qwen4_exp --qkv-format thd --linear-attention-backend flashqla "
        "--custom-model-provider-path miles_plugins.models.qwen3_8_next.model_provider.get_qwen3_8_next_model_provider "
    ),
    sglang_args="--sglang-linear-attn-prefill-backend flashinfer --sglang-chunked-prefill-size 8192 ",
    weight_check_skip="visual. ple_embedding.",
    engine_env=(("SGLANG_DISABLE_MULTIMEM_AG", "1"),),
)
# scripts/run_glm5_3_flash.py
_GLM53_FLASH = _Recipe(
    megatron_model_type="glm5.3-flash",
    tito_model="glm53",
    tensor_parallel=8,
    pipeline_parallel=4,
    engine_gpus=8,
    trainer_args=(
        "--model-name glm5_next --qkv-format thd "
        "--decoder-first-pipeline-num-layers 11 --decoder-last-pipeline-num-layers 12 "
    ),
    sglang_args=(
        "--sglang-chunked-prefill-size 8192 --sglang-dsa-prefill-backend tilelang --sglang-dsa-decode-backend tilelang "
        "--sglang-kv-cache-dtype bfloat16 "
    ),
    weight_check_skip="visual.",
)
ModelName = Literal["Qwen3.8-Flash-Next", "Qwen3.8-Flash-Next-4layer", "GLM-5.3-Flash", "GLM-5.3-Flash-4layer"]
_RECIPES: dict[str, _Recipe] = {
    "Qwen3.8-Flash-Next": _FLASH_NEXT,
    "Qwen3.8-Flash-Next-4layer": replace(
        _FLASH_NEXT,
        megatron_model_type="qwen3.8-flash-next-4layer",
        pipeline_parallel=2,
        engine_gpus=4,
        # the converter's automatic pipeline split puts the PLE layer on a later stage, which the model rejects
        convert_env=(("CONVERT_KEEP_PP1", "1"),),
        four_layer=True,
    ),
    "GLM-5.3-Flash": _GLM53_FLASH,
    "GLM-5.3-Flash-4layer": replace(
        _GLM53_FLASH,
        megatron_model_type="glm5.3-flash-4layer",
        tensor_parallel=2,
        pipeline_parallel=2,
        engine_gpus=4,
        trainer_args="--model-name glm5_next --qkv-format thd ",
        processor_config_from="zai-org/GLM-5.3-Flash",
        four_layer=True,
    ),
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    arm: Literal["off", "on"] | None = None
    model_name: ModelName = "Qwen3.8-Flash-Next"
    run_id: str = field(default_factory=U.create_run_id)
    num_gpus_per_node: int = 8
    trainer_gpus: int = 32
    rollout_gpus: int = 16
    # TP = EP size of one engine; default per model.
    engine_gpus: int | None = None
    num_steps: int = 20
    seed: int = 1234
    # The agent nudges every episode on to at least this many turns; 0 ends it at the first reply with no code to run.
    agent_min_turns: int = 4
    model_dir: str = "/root/models"
    hf_checkpoint: str | None = None
    ref_load: str | None = None
    data_dir: str = "/root/datasets"
    data_path: str | None = None
    megatron_path: str = "/root/Megatron-LM"
    save: str | None = None
    save_interval: int = 25
    extra_args: str = ""

    def __post_init__(self):
        self.hf_checkpoint = self.hf_checkpoint or f"{self.model_dir}/{self.model_name}"
        # where `prepare` converts to, as convert_checkpoint names it
        self.ref_load = self.ref_load or f"{self.model_dir}/{self.model_name}_torch_dist"
        self.data_path = self.data_path or f"{self.data_dir}/dapo-math-17k/dapo-math-17k.jsonl"
        self.engine_gpus = self.engine_gpus or _RECIPES[self.model_name].engine_gpus


def _prepare(args: ScriptArgs):
    ref_load = Path(args.ref_load)
    # convert_checkpoint names its output <dir_dst>/<model_name>_torch_dist
    assert ref_load.name == f"{args.model_name}_torch_dist", f"prepare cannot convert to {ref_load}"
    recipe = _RECIPES[args.model_name]
    backend = args.create_backend()
    if not Path(args.data_path).exists():
        backend.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    if recipe.processor_config_from and not (Path(args.hf_checkpoint) / "processor_config.json").exists():
        backend.exec_command_cpu(
            f"hf download {recipe.processor_config_from} processor_config.json --local-dir {args.hf_checkpoint}"
        )
    os.environ.update(recipe.convert_env)
    backend.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=recipe.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=str(ref_load.parent),
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def _check_topology(args: ScriptArgs, recipe: _Recipe) -> None:
    gpus_per_node = args.num_gpus_per_node
    assert (args.trainer_gpus + args.rollout_gpus) % gpus_per_node == 0 and (
        args.trainer_gpus <= gpus_per_node or args.trainer_gpus % gpus_per_node == 0
    ), "trainer + rollout GPUs must fill whole nodes, and a multi-node trainer must own whole nodes"
    assert (
        args.trainer_gpus % (recipe.tensor_parallel * recipe.pipeline_parallel) == 0
    ), f"--trainer-gpus must be a multiple of TP x PP = {recipe.tensor_parallel * recipe.pipeline_parallel}"
    assert args.rollout_gpus % args.engine_gpus == 0, "--rollout-gpus must be a multiple of the engine size"


def _train(args: ScriptArgs):
    assert args.arm in ("off", "on"), "pass --arm off or --arm on"
    recipe = _RECIPES[args.model_name]
    _check_topology(args, recipe)

    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.ref_load} "
    if args.save:
        ckpt_args += f"--save {args.save} --save-interval {args.save_interval} "

    rollout_args = (
        "--fully-async "
        "--update-weights-interval 1 "
        "--max-weight-staleness 4 "
        "--pause-generation-mode retract "
        "--update-weight-transfer-mode broadcast "
        f"--prompt-data {args.data_path} "
        "--input-key prompt "
        "--label-key label "
        "--rollout-shuffle "
        f"--rollout-seed {args.seed} "
        f"--seed {args.seed} "
        "--balance-data "
        f"--num-rollout {args.num_steps} "
        f"--rollout-batch-size {_PROMPTS_PER_STEP} "
        f"--n-samples-per-prompt {_SAMPLES_PER_PROMPT} "
        f"--global-batch-size {_STEP_SAMPLES} "
        f"--async-max-concurrent-samples {_STEP_SAMPLES} "
        "--rollout-temperature 1.0 "
        f"--rollout-max-response-len {_TURN_TOKENS} "
        f"--rollout-max-context-len {_CONTEXT_TOKENS} "
        f"--max-seq-len {_CONTEXT_TOKENS} "
    )

    agent_args = (
        f"--use-session-server --tito-model {recipe.tito_model} "
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        # inline: the agent only awaits HTTP and its own interpreter, so no episode needs a process of its own
        f"--custom-agent-function-path {_AGENT}.run_agent --custom-agent-function-mode inline "
        "--custom-rm-path examples.retool_v2.tool_sandbox.reward_func --reward-key score "
    )

    replay_args = "--use-rollout-routing-replay "
    if args.arm == "on":
        replay_args += (
            "--use-rollout-routing-replay-forcing "
            f"--sglang-routed-experts-forcing-max-tokens {_FORCING_STORE_TOKENS} "
        )

    perf_args = (
        f"--tensor-model-parallel-size {recipe.tensor_parallel} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {recipe.pipeline_parallel} "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {args.trainer_gpus // recipe.pipeline_parallel} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {_CONTEXT_TOKENS} "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-tis "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-num-gpus {args.rollout_gpus} "
        f"--rollout-num-gpus-per-engine {args.engine_gpus} "
        f"--sglang-tp-size {args.engine_gpus} "
        f"--sglang-ep-size {args.engine_gpus} "
        # routing replay needs materialized topk ids
        "--sglang-moe-runner-backend triton "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-disable-radix-cache "
        f"{recipe.sglang_args}"
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--accumulate-allreduce-grads-in-fp32 "
        f"--update-weight-buffer-size {1 * 1024**3} "
        "--distributed-timeout-minutes 60 "
        f"--actor-num-nodes {max(1, args.trainer_gpus // args.num_gpus_per_node)} "
        f"--actor-num-gpus-per-node {min(args.num_gpus_per_node, args.trainer_gpus)} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"{recipe.trainer_args}"
        # catches a checkpoint whose weights the engines load differently, at step 0
        f"--check-weight-update-equal --check-weight-update-skip-list {recipe.weight_check_skip} "
    )

    train_args = (
        f"{ckpt_args} {rollout_args} {agent_args} {replay_args} {optimizer_args} {grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} {sglang_args} {misc_args} {args.extra_args} "
    )

    extra_env_vars = {
        "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        # inductor's subprocess compile pool deadlocks under the torch_memory_saver LD_PRELOAD
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        **dict(recipe.engine_env),
        _AGENT_MIN_TURNS_ENV: str(args.agent_min_turns),
        _AGENT_IGNORE_EOS_ENV: "1" if recipe.four_layer else "0",
    }
    # ray's runtime_env replaces the actor env; without the JIT cache dirs the kernels recompile every run
    for cache_var in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        cache_dir = os.environ.get(cache_var)
        if cache_dir:
            extra_env_vars[cache_var] = cache_dir

    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=recipe.megatron_model_type,
        train_script="train_async.py",
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download dapo-math-17k when --data-path is missing, fetch a missing processor_config.json when the recipe names a
    source, and convert the checkpoint to torch_dist."""
    _prepare(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run one arm."""
    _train(args)


if __name__ == "__main__":
    app()
