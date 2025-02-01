from typing import List, Dict, Literal
from numbers import Number
import random
import numpy as np
import math


def get_game_section(response: str):
    if "===GAME START===" in response:
        response = response.split("===GAME START===")[1]
    if "===GAME END===" in response:
        response = response.split("===GAME END===")[0]
    return response


def sample_mean_scores(instance_scores:List[Dict[Literal["scores", "weights"], List]], num_samples=1000):
    """
    instance_scores: List of dictionaries of the following keys:
        - scores: List of numbers or tuples (correct, total)
        - weights: List of numbers or None (uniform weights)
    
    Returns: float

    Note that, if `scores` are list of numbers, this function directly computes weighted means of `scores`, and then averages them.
    If `scores` are list of tuples, this function samples `num_samples` from each instance scores, which result in `num_samples` lists of instance scores. 
    Then, it computes the mean of each instance score list, and then average over `num_samples` lists.
    """
    assert isinstance(instance_scores, list)
    if len(instance_scores) == 0:
        return 0.0
    assert isinstance(instance_scores[0], dict) and "scores" in instance_scores[0] and isinstance(instance_scores[0]["scores"], list)

    if isinstance(instance_scores[0]['scores'][0], Number):
        return sum([np.average(instance['scores'], weights=instance.get('weights', None)) for instance in instance_scores]) / len(instance_scores)
    else:
        samples = []
        for instance in instance_scores:
            scores = instance['scores']
            weights = instance.get('weights', None)
            if weights is None:
                weights = [1] * len(scores)
            sample = random.choices(scores, weights=weights, k=num_samples)
            samples.append(sample)
        sample_scores = []
        for i in range(num_samples):
            sample_scores.append(mean_score([sample[i] for sample in samples]))
        return mean_score(sample_scores)


def mean_score(scores):
    """
    Accepts 3 types of inputs:
    - list of numbers
    - list of tuples (correct, total)
    - list of dictionaries with the same keys and values being numbers or tuples (correct, total)
    """
    assert isinstance(scores, list)
    if len(scores) == 0:
        return 0.0
    if isinstance(scores[0], Number):
        return sum(scores) / len(scores)
    elif isinstance(scores[0], dict):
        if (len(scores[0]) == 2 and "scores" in scores[0] and "weights" in scores[0]) or (len(scores[0]) == 1 and "scores" in scores[0]):
            return sample_mean_scores(scores)
        else:
            raise ValueError("Invalid dictionary format")
    else: # assume list of tuples (correct, total)
        try:
            return sum(t[0] for t in scores) / max(sum(t[1] for t in scores), 1)
        except Exception as e:
            raise ValueError(f"Error in computing mean score: {e}")


def get_routes(score_dict):
    """
    given a score dictionary, returns all the possible routes to the scores
    a score will be:
    - a number
    - a tuple (correct, total)
    - a dictionary with "scores" and optionally "weights" keys
        - the "scores" key will be a list of numbers or tuples
        - the "weights" key (if exsits) will be a list of numbers or None
        - if a dictionary with "scores" and optionally "weights" keys is found, but failed the above conditions, raises a warning that "don't use 'scores' and 'weights' in non-score dicti
onaries", and consider them as route elements
    """

    all_routes = []

    def dfs(d, routes):
        if isinstance(d, Number):
            all_routes.append(routes)
            return
        elif isinstance(d, tuple) and len(d) == 2 and isinstance(d[0], Number) and isinstance(d[1], Number):
                all_routes.append(routes)
                return
        # assert isinstance(d, dict), str(d)
        if not isinstance(d, dict):
            return
    
        if "scores" in d:
            weight_check = ("weights" not in d) or (isinstance(d["weights"], list) and isinstance(d["weights"][0], Number))
            if (
                weight_check and \
                isinstance(d["scores"], list) and \
                isinstance(d["scores"][0], (Number, tuple))
            ):
                    all_routes.append(routes)
                    return
        
        for k, v in d.items():
            dfs(v, routes + (k,))
    
    dfs(score_dict, ())
    return all_routes


def mean_score_dict(score_dicts):
    assert isinstance(score_dicts, list)
    assert len(score_dicts) > 0
    assert all(isinstance(d, dict) for d in score_dicts)
    routes = get_routes(score_dicts[0])
    new_score_dict = {}
    for route in routes:
        route_scores = []
        for score_dict in score_dicts:
            pointer = score_dict
            for r in route:
                pointer = pointer[r]
            route_scores.append(pointer)
        
        mean_scores = mean_score(route_scores)
        pointer = new_score_dict
        for r in route[:-1]:
            if r not in pointer:
                pointer[r] = {}
            pointer = pointer[r]
        pointer[route[-1]] = mean_scores
    return new_score_dict


def get_short_results(summary_results):
    return {
        "LEN": summary_results["LEN"],
        "FAC": summary_results["FAC"]["align_relative"],
        "PER": 1 - summary_results["PER"]["overall"] / 4 / math.sqrt(5),
        "PERd": (summary_results["PERd"]["overall"]["rate"] - 1) / 4,
        "ACT": (summary_results["ACT"]["action"] - 1) / 4,
        "INT": (summary_results["INT"]["interestingness"] - 1) / 4,
        "MEC": summary_results['MEC']['final_error_free_rounds_strict'],
        "MEC - ECE": summary_results['MEC']['event_condition_check_error'],
        "MEC - VUE": summary_results['MEC']['var_update_error_strict']
    }