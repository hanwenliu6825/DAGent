<h1 align="center">
  DAGent: Evaluate-then-Grow Planning <br>
  for Deep Research Agents
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.39154">
    <img src="https://img.shields.io/badge/Paper-arXiv%3A2609.39154-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper (arXiv)">
  </a>
  <a href="https://openreview.net/forum?id=gNnU6UhVDm">
    <img src="https://img.shields.io/badge/NeurIPS%202026-Poster-1f4e79?style=for-the-badge" alt="NeurIPS 2026">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-2c7a39?style=for-the-badge" alt="License">
  </a>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-news">News</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-repository-layout">Repository Layout</a> •
  <a href="#-training">Training</a> •
  <a href="#-evaluation">Evaluation</a> •
  <a href="#-citation">Citation</a> •
  <a href="#-acknowledgments">Acknowledgments</a> •
  <a href="#-license">License</a>
</p>

<p align="center">
  <img src="figs/figure2.png" alt="DAGent overview" width="100%">
</p>

## 📖 Overview

Deep research tasks require an agent to search many sources, combine the evidence, and revise its plan as findings come in. Existing DAG-based agents commit to a full plan before execution and patch it after failures. This Plan-then-Patch strategy commits most when the agent knows least, and the patches spend computation on branches that should never have been planned.

DAGent replaces this with **Evaluate-then-Grow** incremental planning: an Orchestrator grows the task DAG one batch at a time, conditioning each expansion on the confidence and uncertainty reported by completed nodes. A hierarchical context layer passes compact **QueryDocs** between nodes by default and exposes a node's full **InteractionTranscript** through a `recall` tool only on demand. The append-only graph also records a topology that defines structural RL signals; **DAGRPO** uses it for topology-conditioned credit on Executor rollouts and a structural compliance regularization on Orchestrator plans.

On BrowseComp-Plus, GAIA, and xbench-DeepSearch, DAGent beats the strongest open-source baseline by 5.3 / 5.8 / 2.0 points at the Qwen3-235B-A22B scale, holds the lead across four open-source backbones and GPT-5 at 327K context, and costs less per task than its Plan-then-Patch counterpart. DAGRPO adds 3.0 points on average over same-budget GRPO at the Qwen3-8B scale.

## 🔥 News

- **[2026-09]** 📄 The paper is on arXiv: [arXiv:2609.39154](https://arxiv.org/abs/2609.39154).
- **[2026-09]** 🎉 DAGent is accepted at **NeurIPS 2026** as a poster.
- **[2026-09]** 🧩 Initial code release: DAGent, DAGRPO, and the training and evaluation scripts. More code is coming soon. ⭐

## ⚙️ Installation

The paper used Python 3.12, PyTorch 2.8.0 (CUDA 12.8), vLLM 0.11.0, and FlashAttention 2.8.3. Training also needs our verl fork (coming soon); `verl/trainer/ppo/core_algos.py` replaces the same file there.

```bash
conda create -n dagent python=3.12 -y
conda activate dagent

pip install vllm==0.11.0
pip install transformers datasets accelerate safetensors \
    openai tiktoken httpx aiohttp requests \
    pandas "pyarrow>=19.0.0" omegaconf hydra-core pydantic \
    "ray[default]" "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    peft dill codetiming filelock psutil tqdm cachetools wandb
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

## 📁 Repository Layout

```text
DAGent/
├── LICENSE
├── README.md
├── figs/
│   └── figure2.png
├── agents/
│   ├── dagent.py               # Orchestrator + TaskDAG / QueryDoc / RecallTool + structural validator + reward shaping
│   ├── prompts.py              # Orchestrator, ReAct Executor, and Recall prompts
│   └── tool_spec.py            # QueryDoc finish schema, plan tool, and recall tool
├── verl/
│   └── trainer/ppo/core_algos.py   # DAGRPO advantage estimator (Eq. 5 of the paper)
└── scripts/
    ├── eval.py                 # Training-free evaluation entry (all benchmarks and workflows)
    └── train_bc_qwen3_8b.sh    # Training launcher for the five RL workflows of Table 1
```

## 🚀 Training

**1. Start the search server**

Start the search server (coming soon; it follows the FoldAgent scaffolding) on a separate machine. It downloads the corpus (`Tevatron/browsecomp-plus-corpus`) and the pre-computed embeddings (`miaolu3/browsecomp-plus`) and loads Qwen3-Embedding-8B.

```bash
cd envs && python search_server.py \
  --model Qwen/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --corpus-embedding-dataset miaolu3/browsecomp-plus \
  --host 0.0.0.0 \
  --port 8000
```

Set the environment variables:

```bash
# URL of the local search server (BrowseComp-Plus)
export LOCAL_SEARCH_URL="http://[IP-of-search-server]:8000"

# LLM judge
export JUDGE_API_KEY="your-openai-api-key"
```

**2. Download the training data**

Download the BrowseComp-Plus 680 / 150 train / test split from the [FoldAgent release](https://github.com/sunnweiwei/FoldAgent) and place it at `data/bc_train.parquet` and `data/bc_test.parquet`.

**3. Train on BrowseComp-Plus**

Train Qwen3-8B with DAGRPO (2 x H200, LoRA, asynchronous vLLM rollouts):

```bash
bash scripts/train_bc_qwen3_8b.sh dagent_dagrpo        # seed 42 (default)
bash scripts/train_bc_qwen3_8b.sh dagent_dagrpo 123    # seeds used in the paper: 42 / 123 / 777
```

Training workflows (Table 1 of the paper, training-based rows; 21 update steps, seeds 42 / 123 / 777):

| Workflow              | Paper row               | Method                                                                   |
|-----------------------|-------------------------|--------------------------------------------------------------------------|
| `dagent_dagrpo`       | DAGRPO-DAGent           | DAGent + DAGRPO (α = 0.5 off-chain credit, structural compliance regularization on) |
| `dagent_grpo`         | GRPO-DAGent             | DAGent + GRPO (α = 1.0, regularization off)                              |
| `search_branch_grpo`  | GRPO-Fold Agent         | Fold Agent + GRPO                                                        |
| `search_grpo`         | GRPO-ReAct Agent (32K)  | ReAct Agent + GRPO (default `RESPONSE_LENGTH=32768`)                     |
| `search_grpo`         | GRPO-ReAct Agent (109K) | Same workflow; set `RESPONSE_LENGTH=109568` before launching             |
| `search_summary_grpo` | GRPO-Summary Agent      | Summary Agent + GRPO                                                     |

## 📊 Evaluation

**1. Start the search server**

Needed for BrowseComp-Plus only; GAIA and xbench-DeepSearch use web tools. Same command as in Training.

**2. Set the credentials**

```bash
# Backbone model (any OpenAI-compatible endpoint; the paper used OpenRouter)
export OPENAI_API_KEY="your-backbone-api-key"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

# LLM judge (GPT-4o-mini primary, GPT-4.1 tie-breaker)
export JUDGE_API_KEY="your-openai-api-key"

# Web tools (GAIA and xbench-DeepSearch)
export GOOGLE_SEARCH_KEY="your-serper-api-key"
export JINA_API_KEYS="your-jina-api-key"
```

**3. Download the evaluation data**

| Benchmark              | Source                                                                          | Notes                                                             |
|------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------|
| BrowseComp-Plus        | https://github.com/sunnweiwei/FoldAgent                                         | 680 train / 150 test split; local search with Qwen3-Embedding-8B  |
| GAIA                   | https://github.com/MiroMindAI/MiroThinker#-benchmark-evaluation                 | 103-task text-only validation subset; web search (Serper + Jina)  |
| xbench-DeepSearch 2505 | https://github.com/MiroMindAI/MiroThinker#-benchmark-evaluation                 | 100 Chinese tasks; web search (Serper + Jina)                     |

Place the parquet files at `data/gaia.parquet` and `data/xbench.parquet`; the BrowseComp-Plus files are at `data/bc_train.parquet` and `data/bc_test.parquet` from the training step.

**4. Evaluate**

DAGent on BrowseComp-Plus with Qwen3-32B (Table 1 of the paper, Qwen3-32B block, DAGent row):

```bash
python scripts/eval.py \
  --benchmark browsecomp \
  --workflow dagent \
  --model_name qwen/qwen3-32b \
  --data_path data/bc_test.parquet \
  --local_search_url $LOCAL_SEARCH_URL \
  --num_workers 10 \
  --output_dir results
```

Other benchmarks: `--benchmark {browsecomp, gaia, xbench}`. Other backbones: `--model_name {qwen/qwen3-8b, qwen/qwen3-32b, qwen/qwen3-235b-a22b-2507}`.

Workflows (Table 1 of the paper, training-free rows):

| `--workflow`     | Extra flags                | Paper row           |
|------------------|----------------------------|---------------------|
| `dagent`         | —                          | **DAGent (ours)**   |
| `search`         | `--response_length 32768`  | ReAct Agent (32K)   |
| `search`         | `--response_length 109568` | ReAct Agent (109K)  |
| `search`         | `--enable_summary`         | Summary Agent       |
| `search_branch`  | —                          | Fold Agent          |
| `flash_searcher` | —                          | Flash-Searcher      |

The baseline workflows are coming soon.

## 📝 Citation

If you find this repository useful, please cite our paper 💗:

```bibtex
@misc{liu2026dagentevaluatethengrowplanningdeep,
      title={DAGent: Evaluate-then-Grow Planning for Deep Research Agents},
      author={Hanwen Liu and Yuanfu Sun and Qiaoyu Tan},
      year={2026},
      eprint={2609.39154},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.39154},
}
```

## 🙏 Acknowledgments

This implementation builds on [verl](https://github.com/volcengine/verl) and the search-server scaffolding of [FoldAgent](https://github.com/sunnweiwei/FoldAgent). Benchmark data comes from [FoldAgent](https://github.com/sunnweiwei/FoldAgent) (BrowseComp-Plus) and [MiroThinker](https://github.com/MiroMindAI/MiroThinker) (GAIA, xbench-DeepSearch). We also thank the open-source community for the libraries this project builds upon.

## 📄 License

This project is released under the Apache License 2.0. Please see [LICENSE](LICENSE).
