import argparse
import json
import os
import random
import time
from rpgbench.models import get_model
from rpgbench.game import prepare_game_prompt, parse_game_output


def parse_args():
    parser = argparse.ArgumentParser()

    model_args = parser.add_argument_group('Model Arguments')
    model_args.add_argument('--model_type', type=str, default='openai', choices=["claude", "deepseek", "gemini", "openai", "ray"], help='Model type')
    model_args.add_argument('--model_name_or_path', type=str, help='Model name or path. See api provider websites for api models.')
    model_args.add_argument('--api_key', type=str, default='', help='API key')
    model_args.add_argument('--ray_headnode', type=str, default='', help='Ray cluster headnode for vllm models')
    model_args.add_argument('--ray_n_gpus_per_model', type=int, default=1, help='Tensor parallel size for vllm models (number of gpus per model)')
    model_args.add_argument('--ray_n_gpus_total', type=int, default=1, help='Total number of GPUs to use for vllm models')
    
    sampling_args = parser.add_argument_group('Sampling Arguments')
    sampling_args.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature')
    sampling_args.add_argument('--top_p', type=float, default=0.9, help='Top p sampling')
    sampling_args.add_argument('--max_tokens', type=int, default=16384, help='Max tokens to generate')

    game_args = parser.add_argument_group("Game parameters")
    game_args.add_argument("--max_rounds", type=int, default=10, help="Maximum number of rounds to run")
    
    file_args = parser.add_argument_group('File Arguments')
    file_args.add_argument('--output_dir', type=str, default='data/game_simulation_generation', help='Output directory to save generated games')

    args = parser.parse_args()
    return args


def get_first_round_role(args):
    if args.model_type == "ray" and "gemma" in args.model_name:
        return "user"
    elif args.model_type in ["claude", 'gemini']:
        return "user"
    else:
        return "system"


def run_games(args):
    with open(os.path.join("data", "games", "games.jsonl"), "rt") as f:
        games = [json.loads(t) for t in f]

    system_prompts = [prepare_game_prompt(game) for game in games]
    rounds = [[{"role": get_first_round_role(args), "content": p}] for p in system_prompts]
    prompt_indices = list(range(len(system_prompts)))
    results = [None for _ in range(len(system_prompts))]
    model = get_model(
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        api_keys=args.api_key,
        ray_headnode=args.ray_headnode,
        ray_n_gpus_per_model=args.ray_n_gpus_per_model,
        ray_n_gpus_total=args.ray_n_gpus_total
    )
    rng = random.Random(42)
    for i_round in range(args.max_rounds):
        print(f"Round {i_round+1}")
        if args.model_type == "vllm":
            _round_outputs = model.batch_completion(rounds, batch_size=2, sampling_kwargs={"n": 1, "temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens})
            round_outputs = []
            for output in _round_outputs:
                try:
                    round_outputs.append(output[0][-1]["content"])
                except Exception as e:
                    round_outputs.append("")
        else:
            round_outputs = model.batch_completion(rounds, model=args.model_name_or_path, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
            if args.model_type == "claude": # Claude models need more request intervals
                time.sleep(30)
        next_indices = []
        next_rounds = []
        for idx, round_history, round_output in zip(prompt_indices, rounds, round_outputs):
            round_history.append({"role": "assistant", "content": round_output})
            try:
                _, state = parse_game_output(round_output)
                next_actions = state['choices']
            except Exception as e:
                print(f"Error parsing model output: {e}. Output: {state}")
                next_actions = None
            if next_actions:
                next_action = rng.choice(next_actions)
                round_history.append({"role": "user", "content": next_action})
                next_rounds.append(round_history)
                next_indices.append(idx)
            else:
                results[idx] = round_history
        rounds = next_rounds
        prompt_indices = next_indices
        if len(rounds) == 0:
            break
    for idx, round_history in zip(prompt_indices, rounds):
        results[idx] = round_history

    if args.model_type == "vllm":
        model.shutdown()
    return results


def main():
    args = parse_args()
    results = run_games(args)

    model_name_or_path = args.model_name_or_path.rstrip('/')
    if '/' in model_name_or_path:
        model_name_or_path = model_name_or_path.split('/')[-1]
    model_name_or_path = model_name_or_path.replace('_', '-')
    save_file = os.path.join(
        args.output_dir,
        f"{model_name_or_path}_t{args.temperature}_p{args.top_p}_m{args.max_tokens}_r{args.max_rounds}.jsonl"
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(save_file, "wt") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    main()