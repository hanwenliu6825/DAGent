#!/bin/bash
# Unified LoRA training script for BrowseComp-Plus with Qwen3-8B.
# Hardware: 2 x H200 (LoRA + vLLM rollout).
#
# Usage:
#   bash scripts/train_bc_qwen3_8b.sh <workflow> [seed]
#
# Workflows (Paper Table 1 training-based rows):
#   dagent_dagrpo        DAGent + DAGRPO (alpha=0.5 multiplicative credit + structural penalty)
#   dagent_grpo          DAGent + role-separated outcome-only GRPO (alpha=1.0, penalty off)
#   search_branch_grpo   Fold Agent + outcome-only GRPO
#   search_grpo          ReAct Agent + GRPO
#   search_summary_grpo  Summary Agent + GRPO
#
# Seed (optional, default 42; multi-seed values: 42 / 123 / 777):
#   Controls data.seed (dataloader shuffle order). Checkpoint dir + wandb run
#   name include _seed${SEED} suffix automatically.

WORKFLOW=${1:?Usage: bash scripts/train_bc_qwen3_8b.sh <dagent_dagrpo|dagent_grpo|search_branch_grpo|search_grpo|search_summary_grpo> [seed] [extra_hydra_args...]}
SEED=${2:-42}   # canonical: 42 / 123 / 777
# Drop $1 / $2 so remaining $@ can be forwarded as Hydra overrides
# (e.g. trainer.total_training_steps=1 from a smoke-test wrapper).
shift $(( $# >= 2 ? 2 : $# ))
EXTRA_HYDRA_ARGS=("$@")

# ===== Common hyperparameters =====
PROMPT_LENGTH=8192
RESPONSE_LENGTH=32768
MAX_LENGTH=40960
# Default: Qwen3-8B for RL training (paper Table 1 training-based rows).
MODEL_PATH=${MODEL_PATH:-/path/to/models/Qwen3-8B}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-data/bc_train.parquet}
TEST_DATA_PATH=${TEST_DATA_PATH:-data/bc_test.parquet}
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}

# ===== Workflow-specific parameters =====
case $WORKFLOW in
  dagent_dagrpo)
    # DAGRPO: alpha=0.5 (off-chain credit) + structural compliance penalty on.
    # Ablation sweep values for reference:
    #   dagent_credit_alpha:        0.0 / 0.25 / 0.5 / 0.75 / 1.0
    #   dagent_structural_penalty:  True / False
    AGENT_LOOP=dagent
    WORKFLOW_NAME=dagent
    PROJECT_NAME=dagent
    EXPERIMENT_NAME=dagent_bc_qwen3_8b
    ADV_ESTIMATOR=dagrpo
    EXTRA_ARGS="\
      +actor_rollout_ref.rollout.plugin.dagent_max_iterations=30 \
      +actor_rollout_ref.rollout.plugin.dagent_task_timeout=1800 \
      +actor_rollout_ref.rollout.plugin.dagent_credit_alpha=0.5 \
      +actor_rollout_ref.rollout.plugin.dagent_structural_penalty=True"
    ;;
  dagent_grpo)
    # GRPO baseline for DAGent (alpha=1.0 disables off-chain attenuation,
    # structural compliance penalty off). Same-budget direct baseline for DAGRPO.
    AGENT_LOOP=dagent
    WORKFLOW_NAME=dagent
    PROJECT_NAME=dagent_grpo_baseline
    EXPERIMENT_NAME=dagent_grpo_bc_qwen3_8b
    ADV_ESTIMATOR=dagrpo
    EXTRA_ARGS="\
      +actor_rollout_ref.rollout.plugin.dagent_max_iterations=30 \
      +actor_rollout_ref.rollout.plugin.dagent_task_timeout=1800 \
      +actor_rollout_ref.rollout.plugin.dagent_credit_alpha=1.0 \
      +actor_rollout_ref.rollout.plugin.dagent_structural_penalty=False"
    ;;
  search_branch_grpo)
    AGENT_LOOP=fold_agent
    WORKFLOW_NAME=search_branch
    PROJECT_NAME=fold_grpo_baseline
    EXPERIMENT_NAME=fold_grpo_bc_qwen3_8b
    ADV_ESTIMATOR=foldgrpo
    EXTRA_ARGS="\
      +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
      +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=2048 \
      +actor_rollout_ref.rollout.plugin.max_session=10 \
      +actor_rollout_ref.rollout.plugin.val_max_session=10 \
      +actor_rollout_ref.rollout.plugin.enable_summary=False \
      +actor_rollout_ref.rollout.plugin.branch_len=${RESPONSE_LENGTH}"
    ;;
  search_grpo)
    AGENT_LOOP=fold_agent
    WORKFLOW_NAME=search
    PROJECT_NAME=react_baseline
    EXPERIMENT_NAME=react_bc_qwen3_8b
    ADV_ESTIMATOR=foldgrpo
    EXTRA_ARGS=""
    ;;
  search_summary_grpo)
    AGENT_LOOP=fold_agent
    WORKFLOW_NAME=search
    PROJECT_NAME=summary_baseline
    EXPERIMENT_NAME=summary_bc_qwen3_8b
    ADV_ESTIMATOR=foldgrpo
    EXTRA_ARGS="\
      +actor_rollout_ref.rollout.plugin.enable_summary=True \
      +actor_rollout_ref.rollout.plugin.max_session=10 \
      +actor_rollout_ref.rollout.plugin.val_max_session=10 \
      +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=2048"
    ;;
  *)
    echo "Unknown workflow: $WORKFLOW"
    echo "Valid options: dagent_dagrpo, dagent_grpo, search_branch_grpo, search_grpo, search_summary_grpo"
    exit 1
    ;;
esac

# Per-seed isolation: append _seed${SEED} so checkpoint dir + wandb run name
# differ across seeds. Old single-seed checkpoints (without _seedN suffix) are
# unaffected; new runs always carry the suffix.
EXPERIMENT_NAME="${EXPERIMENT_NAME}_seed${SEED}"

echo "Training with workflow=${WORKFLOW_NAME}, agent_loop=${AGENT_LOOP}, project=${PROJECT_NAME}, experiment=${EXPERIMENT_NAME}, seed=${SEED}"

python -m scripts.train \
  ray_kwargs.ray_init.num_cpus=32 \
  algorithm.adv_estimator=${ADV_ESTIMATOR} \
  actor_rollout_ref.rollout.agent.default_agent_loop=${AGENT_LOOP} \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=${REPO_ROOT}/configs/agent_loop/agent_loops.yaml \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
  actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_LENGTH} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_LENGTH} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.agent.num_workers=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  data.train_files=${TRAIN_DATA_PATH} \
  data.val_files=${TEST_DATA_PATH} \
  data.seed=${SEED} \
  data.train_batch_size=32 \
  data.max_prompt_length=${PROMPT_LENGTH} \
  data.max_response_length=${RESPONSE_LENGTH} \
  data.return_raw_chat=True \
  data.filter_overlong_prompts=True \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_LENGTH} \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=${MAX_LENGTH} \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.actor.use_torch_compile=False \
  +actor_rollout_ref.rollout.plugin.workflow=${WORKFLOW_NAME} \
  +actor_rollout_ref.rollout.plugin.max_turn=200 \
  +actor_rollout_ref.rollout.plugin.val_max_turn=200 \
  +actor_rollout_ref.rollout.plugin.session_timeout=5400 \
  +actor_rollout_ref.rollout.plugin.max_traj=11 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=True \
  +actor_rollout_ref.rollout.plugin.val_response_length=${RESPONSE_LENGTH} \
  ${EXTRA_ARGS} \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_training_steps=21 \
  trainer.test_freq=7 \
  trainer.save_freq=7 \
  trainer.project_name=${PROJECT_NAME} \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  "${EXTRA_HYDRA_ARGS[@]}"
