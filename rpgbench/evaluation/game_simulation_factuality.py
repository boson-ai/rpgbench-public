import json

from importlib import resources
from typing import Literal, Optional, List, Dict, Callable

from rpgbench.evaluation.evaluation_utils import get_game_section
from rpgbench.utils.json_schema import verify_and_extract

with resources.open_text('rpgbench.static', 'game_character_facts_json_schema.json') as f:
    _game_character_fact_schema = json.load(f)

with resources.open_text('rpgbench.static', 'game_character_facts_example.json') as f:
    _game_character_fact_example = json.load(f)

with resources.open_text('rpgbench.static', 'game_character_facts_prompt.txt') as f:
    _game_character_facts_prompt = f.read()

def compute_fact_scores(fact_estimation):
    scores = {'align': 0, 'contradict': 0}
    for fact in fact_estimation:
        if fact['judgement'] == 'align':
            scores['align'] += 1
        elif fact['judgement'] == 'contradict':
            scores['contradict'] += 1
    
    return {
        "align_ratio": scores['align'] / len(fact_estimation),
        "contradict_ratio": scores['contradict'] / len(fact_estimation),
        "informative_ratio": (scores['align'] + scores['contradict']) / len(fact_estimation),
        "align_relative": 0 if scores['align'] + scores['contradict'] == 0 else scores['align'] / (scores['align'] + scores['contradict'])
    }

def evaluation_fact(history: Optional[List[str]] = None, response: Optional[str] = None, game: Optional[Dict] = None, llm_judge: Optional[Callable] = None):
    npc_name = game["main_npc_name"]
    npc_facts = game['main_npc_description']['additional_facts']
    character_str = f"Character: {npc_name}\nRelevant Content:\n"
    for round in history:
        if round["role"] == "assistant":
            character_str += f"{get_game_section(round['content'])}\n"
    prompt = _game_character_facts_prompt.format(
        npc=npc_name,
        facts="\n".join(f"fact_id: {i}\nfact:{t}\n\n" for i,t in enumerate(npc_facts)),
        game_content=character_str,
        schema="```json\n" + json.dumps(_game_character_fact_schema, indent=2, ensure_ascii=False) + "\n```",
        example="```json\n" + json.dumps(_game_character_fact_example, indent=2, ensure_ascii=False) + "\n```"
    )
    judgement = None
    score = None
    for i in range(3):
        judgement = llm_judge([{"role": "user", "content": prompt}])
        error, parsed_judgement = verify_and_extract(judgement, _game_character_fact_schema)
        if error is not None:
            continue
        score = compute_fact_scores(parsed_judgement)
        return judgement, score
    return judgement, score 