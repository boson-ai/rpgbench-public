import argparse
import json

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from numbers import Number
import os
from typing import List, Dict, Callable
from rpgbench.evaluation import (
    evaluation_fact,
    evaluation_length,
    evaluation_tipi,
    evaluation_consistency,
    evaluation_mechanics,
    evaluation_action,
    evaluation_interestingness
)
from rpgbench.evaluation.evaluation_utils import (
    mean_score_dict,
    get_short_results
)
from rpgbench.models import get_model
from tqdm import tqdm


EVALUATION_DICT = {
    "LEN": evaluation_length,
    "FAC": evaluation_fact,
    "PER": evaluation_tipi,
    "PERd": evaluation_consistency,
    "INT": evaluation_interestingness,
    "ACT": evaluation_action,
    "MEC": evaluation_mechanics,
}



def trajectory_evaluation(games: List[Dict], game_trajectories: List[List[Dict]], evaluation_tasks: Dict[str, Callable], model_type:str, model_name:str, api_key:str=None, num_concurrent_jobs=128):
    results = [{evaluation_task: None for evaluation_task in evaluation_tasks} for _ in range(len(games))]
    full_results = [{evaluation_task: None for evaluation_task in evaluation_tasks} for _ in range(len(games))]
    futures = {}
    model = get_model(
        model_type=model_type,
        model_name_or_path=model_name,
        api_keys=api_key.split(",")
    )
    llm_judge = partial(
        model.completion,
        model=model_name,
        temperature=0.0,
        max_tokens=16384,
        top_p=1.0,
        verification=None
    )
    with ThreadPoolExecutor(num_concurrent_jobs) as executor:
        for i, (game, game_trajectory) in enumerate(zip(games, game_trajectories)):
            for evaluation_task, evaluation in evaluation_tasks.items():
                futures[executor.submit(evaluation, history=game_trajectory, game=game, llm_judge=llm_judge)] = (i, evaluation_task)
        for future in tqdm(as_completed(futures), total=len(futures)):
            trajectory_id, evaluation_task = futures[future]
            current_full_results, current_results = future.result()
            results[trajectory_id][evaluation_task] = current_results
            full_results[trajectory_id][evaluation_task] = current_full_results
    return full_results, results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-file", type=str, default=os.path.join("data", "games", "games.jsonl"))
    parser.add_argument("--game-trajectory-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory to save results. Please use an empty directory, since old files can be overwritten.")

    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--evaluation-tasks", type=str, default="ALL", help="Comma separated list of evaluation tasks to run. Options: LEN, FAC, PER, PERd, INT, ACT, MEC, or ALL for all tasks")

    parser.add_argument("--model-type", type=str, default="openai", choices=["openai", "claude", "gemini", "deepseek"])
    parser.add_argument("--model-name", type=str, default="gpt-4o")
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--num-concurrent-jobs", type=int, default=128)

    return parser.parse_args()


def prepare_evaluation_tasks(evaluation_tasks: str):
    if evaluation_tasks == "ALL":
        return EVALUATION_DICT
    tasks = evaluation_tasks.split(",")
    return {task: EVALUATION_DICT[task] for task in tasks}


def main():
    args = parse_args()
    with open(args.game_file, "rt") as f:
        games = [json.loads(t) for t in f]
    with open(args.game_trajectory_file, "rt") as f:
        game_trajectories = [json.loads(line) for line in f]
    
    valid_indices = [i for i, t in enumerate(game_trajectories) if len(t) > 5]
    print(f'Evaluating {len(valid_indices)} trajectories')
    max_round = args.max_rounds * 2 # taking into account both user and model turns
    games = [games[i] for i in valid_indices]
    game_trajectories = [game_trajectories[i][:max_round] for i in valid_indices]

    print("start_evaluation")
    results_full, results = trajectory_evaluation(
        games,
        game_trajectories,
        prepare_evaluation_tasks(args.evaluation_tasks),
        args.model_type,
        args.model_name,
        args.api_key,
        args.num_concurrent_jobs
    )

    print("saving results")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_full_file = os.path.join(
        args.output_dir,
        f"full_results.jsonl"
    )
    save_file = os.path.join(
        args.output_dir,
        f"results.jsonl"
    )
    save_summary_file = os.path.join(
        args.output_dir,
        f"summary_results.json"
    )
    save_short_file = os.path.join(
        args.output_dir,
        f"short_results.json"
    )

    with open(save_full_file, "wt") as f:
        for result in results_full:
            f.write(json.dumps(result) + "\n")
    
    with open(save_file, "wt") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    results_to_average = [r for r in results if None not in r.values()]
    average_results = mean_score_dict(results_to_average)
    with open(save_summary_file, "wt") as f:
        json.dump(average_results, f, indent=4)
    
    short_results = get_short_results(average_results)
    with open(save_short_file, "wt") as f:
        json.dump(short_results, f, indent=4)
    

if __name__ == "__main__":
    main()