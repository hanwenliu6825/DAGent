# DAGent: Evaluate-then-Grow Planning for Deep Research Agents

![DAGent architecture](figs/figure2.png)

DAGent is a DAG-based multi-agent framework for deep research that replaces
**Plan-then-Patch** (commit a full task DAG upfront, repair it with patches)
with **Evaluate-then-Grow** incremental planning: an Orchestrator grows the
task DAG one batch at a time, conditioning each expansion on QueryDoc-grounded
status signals from already-executed nodes. A hierarchical context layer
propagates compact **QueryDocs** by default and exposes full
**InteractionTranscripts** through a `recall` tool only when needed. The
recorded DAG topology, in turn, makes well-defined a class of RL signals that
outcome-only recipes cannot use — **DAGRPO** instantiates this with a
topology-conditioned credit on Executor rollouts and a structural compliance
regularization on Orchestrator plans.

---

## Repository Layout

```
DAGent/
├── verl/
│   └── trainer/
│       └── ppo/
│           └── core_algos.py   # DAGRPO advantage estimator (Eq.5)
├── agents/
│   ├── dagent.py               # Orchestrator + TaskDAG / QueryDoc / RecallTool + validator + reward shaping
│   ├── prompts.py              # Orchestrator + ReAct Executor + Recall prompts
│   └── tool_spec.py            # QueryDoc finish schema + plan tool + recall tool
└── scripts/
    ├── eval.py                 # Training-free evaluation entry (all benchmarks + workflows)
    └── train_bc_qwen3_8b.sh    # Unified training launcher (5 RL workflows)
```

---

## Training

**1. Start Search Server**

Start the search server on a separate machine. This will download the corpus (`Tevatron/browsecomp-plus-corpus`), pre-computed embeddings (`miaolu3/browsecomp-plus`), and load the Qwen3-Embedding-8B model on available GPUs.

```bash
cd envs && python search_server.py \
  --model Qwen/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --corpus-embedding-dataset miaolu3/browsecomp-plus \
  --host 0.0.0.0 \
  --port 8000
```

Set environment variables:

```bash
# URL of the local search server (for BrowseComp-Plus)
export LOCAL_SEARCH_URL="http://[IP-of-search-server]:8000"

# For LLM-based answer grading
export JUDGE_API_KEY="your-openai-api-key"
```

**2. Download Training Data**

Download and decompress the BrowseComp-Plus 680 / 150 train / test split from the [FoldAgent release](https://github.com/sunnweiwei/FoldAgent), and place the resulting files at `data/bc_train.parquet` and `data/bc_test.parquet`.

**3. Train on BrowseComp-Plus**

Example script to train Qwen3-8B with DAGRPO:

```bash
bash scripts/train_bc_qwen3_8b.sh dagent_dagrpo
```

Available training workflows (Paper Table 1, training-based rows; 21 update steps × 3 independent seeds 42 / 123 / 777):

| Workflow                | Paper row                  | Method                                                                                    |
|-------------------------|----------------------------|-------------------------------------------------------------------------------------------|
| `dagent_dagrpo`         | DAGRPO-DAGent              | DAGent + DAGRPO (α = 0.5 multiplicative off-chain credit + structural compliance penalty) |
| `dagent_grpo`           | GRPO-DAGent                | DAGent + role-separated outcome-only GRPO (α = 1.0, structural penalty off)               |
| `search_branch_grpo`    | GRPO-Fold Agent            | Fold Agent + outcome-only GRPO                                                            |
| `search_grpo`           | GRPO-ReAct Agent (32K)     | ReAct Agent + outcome-only GRPO (default `RESPONSE_LENGTH=32768`)                         |
| `search_grpo`           | GRPO-ReAct Agent (109K)    | Same workflow; set `RESPONSE_LENGTH=109568` env var before launching                      |
| `search_summary_grpo`   | GRPO-Summary Agent         | Summary Agent + outcome-only GRPO                                                         |

---

## Evaluation

**1. Start Search Server**

(Required for BrowseComp-Plus only; GAIA and xbench-DeepSearch use web tools.)

```bash
cd envs && python search_server.py \
  --model Qwen/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --corpus-embedding-dataset miaolu3/browsecomp-plus \
  --host 0.0.0.0 \
  --port 8000
```

**2. Set credentials**

```bash
# Backbone model (any OpenAI-compatible endpoint; defaults to OpenRouter)
export OPENAI_API_KEY="your-backbone-api-key"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

# LLM judge (GPT-4o-mini primary, GPT-4.1 tie-breaker)
export JUDGE_API_KEY="your-openai-api-key"

# Web tools (GAIA + xbench)
export GOOGLE_SEARCH_KEY="your-serper-api-key"
export JINA_API_KEYS="your-jina-api-key"
```

**3. Download Evaluation Data**

| Benchmark             | Source                                                                                             | Notes                                                              |
|-----------------------|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| BrowseComp-Plus       | https://github.com/sunnweiwei/FoldAgent                                                            | 680 train / 150 test split; LocalSearch (Qwen3-Embedding-8B)       |
| GAIA                  | https://github.com/MiroMindAI/MiroThinker#-benchmark-evaluation                                    | 103-task text-only validation subset; WebSearch (Serper + Jina)    |
| xbench-DeepSearch 2505| https://github.com/MiroMindAI/MiroThinker#-benchmark-evaluation                                    | 100 Chinese tasks; WebSearch (Serper + Jina)                       |

Place the resulting parquet files at `data/gaia.parquet` and `data/xbench.parquet` (the BrowseComp-Plus train / test parquets are already at `data/bc_train.parquet` and `data/bc_test.parquet` from the training step).

**4. Evaluate**

DAGent on BrowseComp-Plus with Qwen3-32B (Paper Table 1, Qwen3-32B / DAGent row):

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

Switch the benchmark with `--benchmark {browsecomp, gaia, xbench}` and the backbone with `--model_name {qwen/qwen3-8b, qwen/qwen3-32b, qwen/qwen3-235b-a22b-2507}`.

Workflows (Paper Table 1, training-free rows):

| `--workflow`     | Extra flags                       | Paper row              |
|------------------|-----------------------------------|------------------------|
| `dagent`         | —                                 | **DAGent (ours)**      |
| `search`         | `--response_length 32768`         | ReAct Agent (32K)      |
| `search`         | `--response_length 109568`        | ReAct Agent (109K)     |
| `search`         | `--enable_summary`                | Summary Agent          |
| `search_branch`  | —                                 | Fold Agent             |
| `flash_searcher` | —                                 | Flash-Searcher         |

---

## Acknowledgements

This implementation builds on [verl](https://github.com/volcengine/verl) and
follows the search-server scaffolding from
[FoldAgent](https://github.com/sunnweiwei/FoldAgent). Benchmark data is
obtained from upstream releases: BrowseComp-Plus (train + eval) from
[FoldAgent](https://github.com/sunnweiwei/FoldAgent), GAIA and
xbench-DeepSearch from
[MiroThinker](https://github.com/MiroMindAI/MiroThinker).
