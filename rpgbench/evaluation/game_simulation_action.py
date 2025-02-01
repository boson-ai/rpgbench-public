import json
import json_repair
from importlib import resources
from jinja2 import Template
from rpgbench.game import parse_game_output
from transformers import AutoTokenizer
from typing import Callable, Dict, List, Optional


with resources.open_text('rpgbench.static', 'game_action_rubrics.json') as f:
    _choice_rubrics = json.load(f)

with resources.open_text('rpgbench.static', 'game_action_prompt.txt') as f:
    _choices_prompt = Template(f.read())


def format_history(history):
    for idx, turn in enumerate(history):
        if idx % 2 == 0:
            assert turn['role'] == 'assistant', f"Role mismatch at index {idx}"
        else:
            assert turn['role'] == 'user', f"Role mismatch at index {idx}"
    if len(history) % 2 == 1:
        history.append({"role": "assistant", "content": ""})
    rounds = []
    for idx in range(0, len(history), 2):
        rounds.append({
            "assistant": history[idx]['content'],
            "user": history[idx+1]['content']
        })
    return rounds


def load_history_single(single_trajectory):
    return single_trajectory[0]['content'], format_history(single_trajectory[1:])


def extract_history_and_choices(whole_history, idx):
    relative_history = whole_history[:idx + 1]
    current_round = whole_history[idx]['assistant']
    game_section, state = parse_game_output(current_round)
    return relative_history, state['choices']


def _render_choice_and_check_length(tokenizer, rubric, instruction, history, choices, max_length):
    prompt = _choices_prompt.render(
        rubric=rubric,
        instruction=instruction,
        history=history,
        choices=choices,
    )
    chat_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return prompt, len(tokenizer.tokenize(chat_prompt)) <= max_length


def _try_to_fit_choice_in_length(tokenizer, rubric, instruction, history, choices, max_length):
    max_valid = 0
    min_invalid = len(history)
    assert min_invalid > 0
    prompt, is_valid = _render_choice_and_check_length(tokenizer, rubric, instruction, history, choices, max_length)
    if is_valid:
        return prompt, min_invalid
    pointer = (max_valid + min_invalid) // 2
    last_valid_prompt = None
    while max_valid < min_invalid - 1:
        prompt, is_valid = _render_choice_and_check_length(tokenizer, rubric, instruction, history[:pointer], choices, max_length)
        if is_valid:
            max_valid = pointer
            pointer = (max_valid + min_invalid) // 2
            last_valid_prompt = prompt
        else:
            min_invalid = pointer
            pointer = (max_valid + min_invalid) // 2
    if max_valid == 0:
        return None, 0
    return last_valid_prompt, max_valid


def get_choices_prompts_single_game(trajectory, tokenizer_name=None, max_length=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    game_prompts = []    # game_prompts[round_idx] is a list of prompts (for each rubric)
    game_instruction, game_history = load_history_single(trajectory)
    for idx in range(len(game_history) - 1):  # last round has no choice
        try:
            history, choices = extract_history_and_choices(game_history, idx)
        except Exception as e:
            break
        round_prompts = []
        for r, rubric in _choice_rubrics.items():
            p, valid = _try_to_fit_choice_in_length(tokenizer, rubric, game_instruction, history, choices, max_length)
            assert valid
            round_prompts.append(p)
        game_prompts.append(round_prompts)
    return game_prompts, None


def parse_choice_scores(output_list):
    assert len(output_list) == 3
    errors = 0
    reasons = []
    scores = []
    for text in output_list:
        result = json_repair.loads(text[text.find("{") :])
        try:
            reasons.append(result['reason'])
            scores.append(float(result['score']))
        except Exception as e:
            reasons.append("")
            scores.append(-1)
            errors += 1
    return reasons, scores, errors


def evaluation_action(
    history: Optional[List[str]] = None, 
    response: Optional[str] = None, 
    game: Optional[Dict] = None,
    llm_judge: Optional[Callable] = None
):
    assert history, "History must be provided"
    assert llm_judge is not None
    tokenizer_name = "/fsx/models/Meta-Llama-3-8B-Instruct"
    total_errors = 0
    game_prompts, n_discarded = get_choices_prompts_single_game(history, tokenizer_name, 32768 - 1)
    game_reasons, game_scores, game_judges = [], [], []
    for round_id, prompts in enumerate(game_prompts):
        judges = [llm_judge([{"role": "user", "content": p}]) for p in prompts]
        reasons, scores, errors = parse_choice_scores(judges)
        game_reasons.append(reasons)
        game_scores.append(scores)
        game_judges.append(judges)
        total_errors += errors

    total_score_sum, total_score_count = 0, 0
    game_rounds = []
    for round_idx, scores in enumerate(game_scores):
        total_score_sum += sum(scores)
        total_score_count += len(scores)
        game_rounds.append({
            "round_idx": round_idx,
            "scores": scores,
            "reason": game_reasons[round_idx],
            "judge": game_judges[round_idx],
        })
    final_score = total_score_sum / total_score_count if total_score_count > 0 else 0
    return game_rounds, {"action": final_score}
    