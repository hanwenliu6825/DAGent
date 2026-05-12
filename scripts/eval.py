#!/usr/bin/env python3
import asyncio
import argparse
import json
import numpy as np
import pandas as pd
import sys
import warnings
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore', message='.*fast tokenizer.*')

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from agents.utils import CallAPI, TaskContext
from verl import DataProto
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate agents on BrowseComp-Plus benchmark')
    parser.add_argument('--data_path', default='data/bc_test.parquet',
                        help='Path to test data parquet file (default: data/bc_test.parquet)')
    parser.add_argument('--output_dir', default='results',
                        help='Directory to save evaluation results (default: results)')
    parser.add_argument('--prompt_length', type=int, default=8192,
                        help='Maximum prompt length in tokens (default: 8192, matches RL training config)')
    parser.add_argument('--response_length', type=int, default=32768,
                        help='Maximum response length in tokens (default: 32768)')
    parser.add_argument('--workflow', default='dagent',
                        help='Agent workflow: dagent | search | search_branch | flash_searcher (default: dagent). '
                             'search = ReAct (add --enable_summary for Summary agent); '
                             'search_branch = Fold Agent.')
    parser.add_argument('--max_turn', type=int, default=200,
                        help='Maximum turns during training (default: 200)')
    parser.add_argument('--val_max_turn', type=int, default=200,
                        help='Maximum turns during validation/evaluation (default: 200)')
    parser.add_argument('--max_session', type=int, default=10,
                        help='Maximum branch sessions for Fold Agent during training (default: 10)')
    parser.add_argument('--val_max_session', type=int, default=10,
                        help='Maximum branch sessions for Fold Agent during validation (default: 10)')
    # Default training-free eval backbone = Qwen3-32B (paper Table 1 training-free Qwen3-32B row).
    # Other supported values: qwen/qwen3-8b, qwen/qwen3-235b-a22b-2507.
    parser.add_argument('--model_name', default='qwen/qwen3-32b',
                        help='Backbone model name (default: qwen/qwen3-32b). '
                             'Supported: qwen/qwen3-8b, qwen/qwen3-32b, qwen/qwen3-235b-a22b-2507.')
    parser.add_argument('--num_workers', type=int, default=150,
                        help='Number of parallel evaluation workers (default: 150)')
    parser.add_argument('--local_search_url', default='http://localhost:8000',
                        help='URL of the local search server (default: http://localhost:8000)')
    parser.add_argument('--enable_summary', action='store_true',
                        help='Enable summary mode (use with --workflow=search for Summary Agent baseline)')
    # DAGent specific
    parser.add_argument('--benchmark', default='browsecomp',
                        help='Benchmark: browsecomp, gaia, or xbench (default: browsecomp)')
    parser.add_argument('--dagent_max_iterations', type=int, default=30,
                        help='DAGent max orchestrator iterations (default: 30)')
    parser.add_argument('--dagent_task_timeout', type=int, default=600,
                        help='DAGent per-node timeout in seconds (default: 600)')
    # Flash-Searcher specific (baseline 4 in paper Table 1)
    parser.add_argument('--flash_searcher_max_steps', type=int, default=40,
                        help='Flash-Searcher max action steps (default: 40)')
    parser.add_argument('--flash_searcher_summary_interval', type=int, default=4,
                        help='Flash-Searcher summary interval (paper default 8 at 131K context; '
                             'adapted to 4 for our 32K context budget)')
    # Smoke test: limit to first N rows of the dataset
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit eval to first N samples (for smoke testing); None = full dataset')
    return parser.parse_args()


async def eval_one(row, config, tokenizer, model_name, process_item_fn):
    llm_client = CallAPI(url=model_name, tokenizer=tokenizer, config=config.actor_rollout_ref.rollout)
    context = TaskContext(config=config, global_step=0, llm_client=llm_client, is_train=False, tokenizer=tokenizer)

    item = DataProto()
    item.non_tensor_batch = {
        'ability': np.array([row['ability']], dtype=object),
        'extra_info': np.array([row['extra_info']], dtype=object),
        'uid': np.array([row['extra_info'].get('instance_id',
                         row['extra_info'].get('task_id',
                         str(row['extra_info'].get('original_idx', 'unknown'))))], dtype=object),
        'reward_model': np.array([row['reward_model']], dtype=object),
    }
    item.meta_info = {'generation_kwargs': {}, 'max_turn': config.actor_rollout_ref.rollout.plugin.val_max_turn}

    output = await process_item_fn(item, context)

    # plan §4.7: persist efficiency stats (dagent_stats for DAGent, env_stats for others)
    extra = output[0].extra_fields if output else {}
    stats = extra.get('dagent_stats') or extra.get('env_stats') or {}

    result = {
        'instance_id': row['extra_info'].get('instance_id',
                       row['extra_info'].get('task_id',
                       str(row['extra_info'].get('original_idx', 'unknown')))),
        'data_source': row.get('data_source', row['extra_info'].get('dataset', 'unknown')),
        'level': row['extra_info'].get('level', None),
        'score': output[0].reward_score if output else 0,
        'status': 'success' if output else 'failed',
        'stats': dict(stats) if stats else {}
    }
    return result


async def worker(worker_id, rows, args, pbar, shared_scores):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    plugin_config = {
        'workflow': args.workflow,
        'max_turn': args.max_turn,
        'val_max_turn': args.val_max_turn,
        'max_session': args.max_session,
        'val_max_session': args.val_max_session,
        'session_timeout': 5400,
        'process_reward': None,
        'max_traj': None,
        'must_finish': False,
        'double_check': False,
        'must_search': False,
        'enable_summary': args.enable_summary
    }

    # DAGent specific config
    if args.workflow == 'dagent':
        plugin_config.update({
            'dagent_max_iterations': args.dagent_max_iterations,
            'dagent_task_timeout': args.dagent_task_timeout,
        })

    # Flash-Searcher specific config
    if args.workflow == 'flash_searcher':
        plugin_config.update({
            'flash_searcher_max_steps': args.flash_searcher_max_steps,
            'flash_searcher_summary_interval': args.flash_searcher_summary_interval,
        })

    config = OmegaConf.create({
        'actor_rollout_ref': {'rollout': {
            'prompt_length': args.prompt_length,
            'response_length': args.response_length,
            'plugin': plugin_config
        }}
    })

    # Dynamic import based on workflow
    if args.workflow == 'dagent':
        from agents.dagent import process_item as process_item_fn
    elif args.workflow == 'flash_searcher':
        from agents.flash_searcher import process_item as process_item_fn
    else:
        from agents.fold_agent import process_item as process_item_fn

    results = []
    for row in rows:
        result = await eval_one(row, config, tokenizer, args.model_name, process_item_fn)
        results.append(result)
        shared_scores.append(result['score'])
        avg_score = np.mean(shared_scores)
        pbar.set_postfix({'avg_score': f"{avg_score:.3f}", 'id': result['instance_id']})
        pbar.update(1)

    return results


def main():
    args = parse_args()
    os.environ["LOCAL_SEARCH_URL"] = args.local_search_url

    # Load data
    df = pd.read_parquet(args.data_path)
    print(f"Loaded {len(df)} items")
    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit).reset_index(drop=True)
        print(f"[--limit {args.limit}] Truncated to {len(df)} items for smoke test")

    # GAIA / xbench benchmark: override ability to WebSearch
    if args.benchmark == 'gaia':
        print("[GAIA] Setting ability='WebSearch' for all items")
        df['ability'] = 'WebSearch'
    elif args.benchmark == 'xbench':
        print("[xbench] Setting ability='WebSearch' for all items")
        df['ability'] = 'WebSearch'

    # Split for workers
    chunk_size = len(df) // args.num_workers
    chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size if i < args.num_workers-1 else len(df)]
              for i in range(args.num_workers)]

    # Run workers with progress bar
    async def run_all():
        shared_scores = []
        with tqdm(total=len(df), desc="Evaluating", unit="item") as pbar:
            tasks = [worker(i, [chunks[i].iloc[j] for j in range(len(chunks[i]))], args, pbar, shared_scores)
                     for i in range(args.num_workers)]
            return await asyncio.gather(*tasks)

    all_results = asyncio.run(run_all())
    results = [r for worker_results in all_results for r in worker_results]

    # Summary overall
    avg_score = np.mean([r['score'] for r in results])
    print(f"\n{'='*60}")
    print(f"Overall - Avg Score: {avg_score:.4f}, Success: {sum(r['status']=='success' for r in results)}/{len(results)}")

    # Summary by data_source
    from collections import defaultdict
    by_source = defaultdict(list)
    for r in results:
        by_source[r['data_source']].append(r['score'])

    print(f"\nBy Data Source:")
    for source in sorted(by_source.keys()):
        scores = by_source[source]
        print(f"  {source}: {np.mean(scores):.4f} ({len(scores)} items)")

    # Summary by level (useful for GAIA)
    by_level = defaultdict(list)
    for r in results:
        level = r.get('level')
        if level is not None:
            by_level[level].append(r['score'])

    if by_level:
        print(f"\nBy Level:")
        for level in sorted(by_level.keys()):
            scores = by_level[level]
            correct = int(sum(scores))
            total = len(scores)
            print(f"  Level {level}: {np.mean(scores):.4f} ({np.mean(scores)*100:.2f}%) - {correct}/{total}")

    # Save
    Path(args.output_dir).mkdir(exist_ok=True)
    output_file = Path(args.output_dir) / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_by_source = {src: {'avg_score': float(np.mean(scores)), 'count': len(scores)}
                         for src, scores in by_source.items()}
    summary_by_level = {lvl: {'avg_score': float(np.mean(scores)), 'count': len(scores)}
                        for lvl, scores in by_level.items()} if by_level else {}
    json.dump({'avg_score': avg_score, 'by_source': summary_by_source,
               'by_level': summary_by_level, 'results': results},
              open(output_file, 'w'), indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()