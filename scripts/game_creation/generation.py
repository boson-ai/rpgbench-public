import argparse
import json
import os

from importlib import resources
from rpgbench.models import get_model
from rpgbench.game import verify_and_extract_game


with resources.open_text('rpgbench.static', 'game_creation_prompt.txt') as f:
    gc_prompt = f.read()

with resources.open_text('rpgbench.static', 'game_creation_examples.json') as f:
    gc_examples = json.load(f)

with resources.open_text('rpgbench.static', 'game_json_schema.json') as f:
    game_schema = json.load(f)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Game Creation Geneartion')

    model_args = parser.add_argument_group('Model Arguments')
    model_args.add_argument('--model_type', type=str, default='openai', choices=["claude", "deepseek", "gemini", "openai", "ray"], help='Model type')
    model_args.add_argument('--model_name_or_path', type=str, help='Model name or path. See api provider websites for api models.')
    model_args.add_argument('--api_key', type=str, default='', help='API key')
    model_args.add_argument('--ray_headnode', type=str, default='', help='Ray cluster headnode for vllm models')
    model_args.add_argument('--ray_n_gpus_per_model', type=int, default=1, help='Tensor parallel size for vllm models (number of gpus per model)')
    model_args.add_argument('--ray_n_gpus_total', type=int, default=1, help='Total number of GPUs to use for vllm models')

    sampling_args = parser.add_argument_group('Sampling Arguments')
    sampling_args.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    sampling_args.add_argument('--top_p', type=float, default=1.0, help='Top p sampling')
    sampling_args.add_argument('--max_tokens', type=int, default=8192, help='Max tokens to generate')

    file_args = parser.add_argument_group('File Arguments')
    file_args.add_argument('--output_dir', type=str, default='data/game_creation_generation', help='Output directory to save generated games')

    return parser.parse_args()


def sanity_check(args):
    if args.model_type == 'ray':
        if not args.ray_headnode:
            raise ValueError('ray_headnode must be specified for ray models')
    else:
        if not args.api_key:
            raise ValueError('api_key must be specified for api models')


def main():
    # Parse arguments
    args = parse_arguments()
    sanity_check(args)
    # Load character wikipedia pages
    with open(os.path.join('data', 'game_creation', 'characters.jsonl'), 'rt') as f:
        characters = [json.loads(line) for line in f]
    prompts = []
    for character in characters:
        character_content = f"{character['name']}\n{character['description']}"
        prompt = gc_prompt.format(
            wikicontent=character_content,
            schema=game_schema
        )
        messages = gc_examples + [{ "role": "user", "content": prompt}]
        prompts.append(messages)
    # Prepare model
    model = get_model(
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        api_keys=args.api_key,
        ray_headnode=args.ray_headnode,
        ray_n_gpus_per_model=args.ray_n_gpus_per_model,
        ray_n_gpus_total=args.ray_n_gpus_total
    )
    # Generate games
    if args.model_type != 'ray': # For API models
        outputs = model.batch_completion(
            batch=prompts,
            model=args.model_name_or_path,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    else: # For Ray models
        _outputs = model.batch_completion(
            messages=prompts,
            batch_size=32,
            sampling_kwargs={
                "n": 1,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens
            }
        )
        outputs = []
        for output in _outputs:
            try:
                _outputs.append(output[0][-1]["content"])
            except Exception as e: # A ray or vllm error occurs
                _outputs.append("")
    # Save games
    model_name_or_path = args.model_name_or_path.rstrip('/')
    if '/' in model_name_or_path:
        model_name_or_path = model_name_or_path.split('/')[-1]
    model_name_or_path = model_name_or_path.replace('_', '-')
    save_file = os.path.join(
        args.output_dir,
        f"{model_name_or_path}_t{args.temperature}_p{args.top_p}_m{args.max_tokens}.jsonl"
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(save_file, 'wt') as f:
        for output in outputs:
            error, parsed = verify_and_extract_game(output)
            if not error:
                f.write(json.dumps(parsed) + "\n")
            else:
                f.write(json.dumps({"error" : f"{error}", "response" : output}) + "\n")


if __name__ == '__main__':
    main()