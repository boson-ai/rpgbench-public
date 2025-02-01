import nltk
from typing import Literal, Optional, List, Dict, Callable
from rpgbench.evaluation.evaluation_utils import get_game_section

def evaluation_length(history: Optional[List[str]] = None, response: Optional[str] = None, game: Optional[Dict] = None, llm_judge: Optional[Callable] = None):
    assert history is not None
    rounds = [round for round in history if round["role"] == "assistant"]
    lengths = [len(nltk.word_tokenize(get_game_section(round["content"]))) for round in rounds]
    return lengths, sum(lengths) / len(lengths) if lengths else 0