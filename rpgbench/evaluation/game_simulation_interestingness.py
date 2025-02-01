import json

from importlib import resources
from typing import Literal, Optional, List, Dict, Callable

from rpgbench.evaluation.evaluation_utils import get_game_section
from rpgbench.utils.json_schema import verify_and_extract

with resources.open_text('rpgbench.static', 'game_interestingness_json_schema.json') as f:
    _game_interestingness_schema = json.load(f)

with resources.open_text('rpgbench.static', 'game_interestingness_prompt.txt') as f:
    _game_interestingness_prompt = f.read()

def evaluation_interestingness(history: Optional[List[str]] = None, response: Optional[str] = None, game: Optional[Dict] = None, llm_judge: Optional[Callable] = None):
    character_str = ""
    for round in history:
        if round["role"] == "assistant":
            character_str += f"{get_game_section(round['content'])}\n"
    prompt = _game_interestingness_prompt.format(
        game_content=character_str,
        schema="```json\n" + json.dumps(_game_interestingness_schema, indent=2, ensure_ascii=False) + "\n```"
    )
    judgement = None
    parsed_judgement = None
    for i in range(3):
        judgement = llm_judge([{"role": "user", "content": prompt}])
        error, parsed_judgement = verify_and_extract(judgement, _game_interestingness_schema)
        if error is not None:
            continue
        return judgement, parsed_judgement
    return judgement, parsed_judgement