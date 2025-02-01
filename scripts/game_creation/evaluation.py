import argparse
import json
import os

from functools import partial
from multiprocessing import Pool
from rpgbench.evaluation import game_bfs_check


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--games',
        type=str,
        help='Path to generated games JSONL file',
        required=True
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/game_creation_evaluation',
        help='Output directory to save evaluation results'
    )
    parser.add_argument(
        '--max_states',
        type=int,
        default=10000000,
        help='Maximum number of states to explore'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=5,
        help='Number of workers to use'
    )
    return parser.parse_args()


def sanity_check(args):
    if os.path.exists(args.output_dir) and os.path.samefile(os.path.dirname(args.games), args.output_dir):
        raise ValueError('games and output_dir must be different')

def summarize_results(results):

    formatted_games = [r for r in results if r['terminate_reason'] != 'error']
    fcr = len(formatted_games) / len(results)
    
    valid_games = [r for r in results if (
        r['terminate_reason'] != 'error' and \
        len(r['unreachable_event']) == 0 and \
        len(r['unreachable_scene']) == 0  and \
        r['win_state_count'] > 0 and \
        r['lose_state_count'] > 0)]
    vcr = len(valid_games) / len(results)

    w_success = len([r for r in results if (
        r['terminate_reason'] != 'error' and \
        r['win_state_count'] > 0)]) / max(1, len(formatted_games))
    
    w_lose = len([r for r in results if (
        r['terminate_reason'] != 'error' and \
        r['lose_state_count'] > 0)]) / max(1, len(formatted_games))
    
    reachability = len([r for r in results if (
        r['terminate_reason'] != 'error' and \
        len(r['unreachable_event']) == 0 and \
        len(r['unreachable_scene']) == 0)]) / max(1, len(formatted_games))

    average_win_lost_ratio = sum([t['win_state_count'] / max(1, t['lose_state_count']) for t in valid_games]) / max(1, len(valid_games))

    average_win_lost_length_ratio = sum([t['average_win_depth'] / max(1, t['average_lose_depth']) for t in valid_games]) / max(1, len(valid_games))

    return {
        'fcr': fcr,
        'vcr': vcr,
        'w_success': w_success,
        'w_lose': w_lose,
        'reachability': reachability,
        'average_win_lost_ratio': average_win_lost_ratio,
        'average_win_lost_length_ratio': average_win_lost_length_ratio
    }


def main():
    # Parse arguments
    args = parse_arguments()
    sanity_check(args)
    # Load generated games
    with open(args.games, 'rt') as f:
        games = [json.loads(line) for line in f]
    # Evaluate games
    bfs_check = partial(game_bfs_check, max_states_explored=args.max_states)

    with Pool(args.num_workers) as p:
        results = p.map(bfs_check, games)
    
    # Save evaluation results
    base_game_filename = os.path.basename(args.games).rsplit('.', 1)[0]
    save_filename = os.path.join(args.output_dir, f'{base_game_filename}_full.jsonl')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(save_filename, 'wt') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    summary_filename = os.path.join(args.output_dir, f'{base_game_filename}_summary.json')
    with open(summary_filename, 'wt') as f:
        json.dump(summarize_results(results), f, indent=4)


if __name__ == '__main__':
    main()