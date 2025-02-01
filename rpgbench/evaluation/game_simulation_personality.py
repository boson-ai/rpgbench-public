import json
import math
import warnings

from importlib import resources
from typing import Callable, Dict, List, Literal, Optional

from rpgbench.evaluation.evaluation_utils import get_game_section
from rpgbench.utils.json_schema import verify_and_extract

with resources.open_text("rpgbench.static", "game_character_personality_tipi_prompt.txt") as f:
    tipi_prompt = f.read()

with resources.open_text("rpgbench.static", "game_character_personality_tipi_json_schema.json") as f:
    tipi_json_schema = json.load(f)
    tipi_json_schema_str = "```json\n" + json.dumps(tipi_json_schema, indent=2, ensure_ascii=False) + "\n```"

with resources.open_text('rpgbench.static', 'game_character_personality_consistency_prompt.txt') as f:
    _game_character_consistency_prompt = f.read()

with resources.open_text('rpgbench.static', 'game_character_personality_consistency_json_schema.json') as f:
    _game_character_personality_schema = json.load(f)
    _game_character_personality_schema_str = "```json\n" + json.dumps(_game_character_personality_schema, indent=2, ensure_ascii=False) + "\n```"


def compute_personality_scores(personality_estimation, personality):
    scores = {}
    for trait, value in personality.items():
        if isinstance(personality_estimation[trait], dict):
            prediction = personality_estimation[trait]['rate']
        else:
            prediction = personality_estimation[trait]
        if isinstance(value, dict):
            value = value['rate']
        scores[trait] = math.fabs(prediction - value)
    scores['overall'] = math.sqrt(sum([s ** 2 for s in scores.values()]))
    return scores


def get_5scale_big5_score(score: Dict[str, int]):
    extraversion = score['A'] + 8 - score['F']
    agreeableness = score['G'] + 8 - score['B']
    conscientiousness = score['C'] + 8 - score['H']
    neuroticism = score['I'] + 8 - score['D']
    openness = score['E'] + 8 - score['J']
    def rescale(value):
        return (value + 1) / 3
    return {
        "extraversion": rescale(extraversion),
        "agreeableness": rescale(agreeableness),
        "conscientiousness": rescale(conscientiousness),
        "neuroticism": rescale(neuroticism),
        "openness": rescale(openness)
    }


def evaluation_tipi(history: Optional[List[str]] = None, response: Optional[str] = None, game: Optional[Dict] = None, llm_judge: Optional[Callable] = None):
    assert game is not None
    assert llm_judge is not None
    npc_name = game["main_npc_name"]
    character_str = f"Character: {npc_name}\nRelevant Content:\n"
    for round in history:
        if round["role"] == "assistant":
            character_str += f"{get_game_section(round['content'])}\n"
    prompt = tipi_prompt.format(schema=tipi_json_schema_str, character=character_str)
    judgement = None
    score = None
    for i in range(3):
        judgement = llm_judge([{"role": "user", "content": prompt}])
        error, parsed_judgement = verify_and_extract(judgement, tipi_json_schema)
        if error is not None:
            continue
        parsed_judgement = get_5scale_big5_score(parsed_judgement)
        score = compute_personality_scores(parsed_judgement, game["main_npc_description"]["big5_personality_traits"])
        return judgement, score
    return judgement, score


def evaluation_consistency(history: Optional[List[str]] = None, response: Optional[str] = None, game: Optional[Dict] = None, llm_judge: Optional[Callable] = None):
    assert game is not None
    personality_str = f'Main NPC: {game["main_npc_name"]}\n'
    for trait, value in game["main_npc_description"]["big5_personality_traits"].items():
        personality_str += f"**{trait}**:\n"
        personality_str += f"Rating (1 - 5): {value['rate']}\n"
        personality_str += f"Description: {value['description']}\n"

    game_narrative_str = ''
    for round in history:
        if round['role'] == 'assistant':
            game_narrative_str += get_game_section(round['content']) + '\n'
    prompt = _game_character_consistency_prompt.format(
        game_narrative = game_narrative_str,
        npc_personality = personality_str,
        schema = _game_character_personality_schema_str
    )
    judgement = None
    score = None
    for i in range(3):
        judgement = llm_judge([{"role": "user", "content": prompt}])
        error, score = verify_and_extract(judgement, _game_character_personality_schema)
        if error is not None:
            continue
        score['overall'] = {
            "rate": sum([v['rate'] for v in score.values()]) / len(score),
            "description": ""
        }
        return judgement, score
    return judgement, score