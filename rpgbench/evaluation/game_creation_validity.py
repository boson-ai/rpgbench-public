from collections import deque
from copy import deepcopy
import math
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass


class DictObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictObject(value))
            else:
                setattr(self, key, value)
                
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictObject):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


@dataclass
class GameState:
    variables: Dict[str, int]
    hidden_variables: Dict[str, int]
    events_completed: Dict[str, int]
    path: List[str]

class GameSimulator:
    def __init__(self, state: GameState, game_data: Dict):
        self.game_state = {
            "variables": state.variables,
            "hidden_variables": state.hidden_variables,
        }
        self.game_data = game_data
    
    def check_variable_validity(self, state) -> None:
        for v in self.game_data['state_variables']:
            variable_name = v['value_name']
            value = state.variables[variable_name]
            min, max = int(v['min_value']), int(v['max_value'])
            if value < min:
                state.variables[variable_name] = min
            elif value > max:
                state.variables[variable_name] = max
        for v in self.game_data['hidden_variables']:
            variable_name = v['value_name']
            value = state.hidden_variables[variable_name]
            min, max = int(v['min_value']), int(v['max_value'])
            if value < min:
                state.hidden_variables[variable_name] = min
            elif value > max:
                state.hidden_variables[variable_name] = max
    
    def pre_event_check(self) -> Dict[int, bool]:
        for check in self.game_data["pre_event_checks"]:
            run_check = True
            for condition in check['condition']:
                if not self._evaluate_condition(condition):
                    run_check = False
                    break
            if run_check:
                self.game_state['variable'], self.game_state['hidden_variables'] = self._apply_effects(check['effect'])
                if check['check_name'] == "If Succeeded":
                    return {"succeeded": True, "failed": False}
                if check['check_name'] == "If Failed":
                    return {"succeeded": False, "failed": True}
        return {"succeeded": False, "failed": False}
    
    def get_available_events(self) -> List[Dict]:
        available_events = []
        
        for event in self.game_data["events"]:
            can_enter = True
            for condition in event["entering_condition"]:
                if not self._evaluate_condition(condition):
                        can_enter = False
                        break
            if can_enter:
                available_events.append(event)
        return available_events
    
    def handle_event(self, event: Dict):
        success = True
        for condition in event["succeed_condition"]:
            if not self._evaluate_condition(condition):
                success = False
                break
            
        if success:
            state = self._apply_effects(event["succeed_effect"])
        else:
            state = self._apply_effects(event["fail_effect"])
        
        return state
    
    def _apply_effects(self, effects: List[str]):
        v, h = DictObject(self.game_state['variables']), DictObject(self.game_state['hidden_variables'])
        for effect in effects:
            try:
                exec(effect)
            except Exception as e:
                raise Exception(f"Error applying effect: {e}")
        return (v.to_dict(), h.to_dict())
    
    def _evaluate_condition(self, condition: str) -> bool:
        v, h = DictObject(self.game_state['variables']), DictObject(self.game_state['hidden_variables'])
        try:
            return eval(condition)
        except:
            raise Exception(f"Error evaluating condition: {condition}")


def state_hash(state):
    return (
        tuple(sorted(state.variables.items())),
        tuple(sorted(state.hidden_variables.items())) 
    )


def bfs(game_data: Dict, max_win: int = 1, max_states_explored: int = 1000) -> Tuple[List[GameState], List[GameState], Set[str], int]:
    visited_states = set()
    states_explored = 0
    terminate_reason = 'queue_empty'
    queue = deque([
        GameState(
            variables = {v['value_name']:int(v["initial_value"]) for v in game_data["state_variables"]},
            hidden_variables = {v['value_name']:int(v.get("initial_value", 0)) for v in game_data["hidden_variables"]},
            events_completed = {e["unique_id"]:0 for e in game_data["events"]},
            path = []
        )
    ])
    win_set, lose_set, events = [], [], set()
    while queue:
        states_explored += 1
        if states_explored % max_states_explored == 0: 
            terminate_reason = 'reach_max_states_explored'
            break
        current_state = queue.popleft()
        if state_hash(current_state) in visited_states:
            continue
        visited_states.add(state_hash(current_state))
        game = GameSimulator(current_state, game_data)
        pre_check = game.pre_event_check()
        if pre_check["succeeded"]:
            win_set.append(current_state)
            if len(win_set) == max_win:
                terminate_reason = 'reach_max_win_count'
                break
            continue
        if pre_check["failed"]:
            lose_set.append(current_state)
            continue
        available_events = game.get_available_events() 
        for event in available_events:
            events.add(event["unique_id"])
            new_state = deepcopy(current_state)
            new_state.variables, new_state.hidden_variables = game.handle_event(event)
            new_state.path = current_state.path + [event["unique_id"]]
            new_state.events_completed[event["unique_id"]] += 1
            game.check_variable_validity(new_state)
            queue.append(new_state)
    print(f"Explored {states_explored} states.")  
    depth = len(current_state.path)
    return win_set, lose_set, events, depth, states_explored, terminate_reason

def analysis_result(game, win_set, lose_set, events, depth, states_explored, terminate_reason):
    print(f"Last Depth: {depth}")
    unreachable_event = []
    reachable_scene = set()
    for event in game['events']:
        if not event['unique_id'] in events:
            unreachable_event.append(event['unique_id'])
        else:
            if event['scene']:
                reachable_scene.update(event['scene'])
    print(f"Unreachable events: {unreachable_event}")
    print(f"Unreachable scenes: {[scene['unique_id'] for scene in game['scenes'] if scene['unique_id'] not in reachable_scene]}")
    
    print(f"\n=== Lose Analysis ===")
    print(f"Lose State Count: {len(lose_set)}")
    average_lose_depth, min_lose_depth, max_lose_depth, average_lose_complete_event = 0, 0, 0, {}
    if len(lose_set) > 0:
        average_lose_depth = sum([len(lose_sample.path) for lose_sample in lose_set]) / len(lose_set)
        print(f"Average Lose Depth: {average_lose_depth}")
        min_lose_depth = min([len(lose_sample.path) for lose_sample in lose_set])
        print(f"Min Lose Depth: {min_lose_depth}")
        max_lose_depth = max([len(lose_sample.path) for lose_sample in lose_set])
        print(f"Max Lose Depth: {max_lose_depth}")
        
        lose_complete_event = {k['unique_id']:[] for k in game['events']}
        for lose_sample in lose_set:
            for event_id, count in lose_sample.events_completed.items():
                    lose_complete_event[event_id].append(count)
        average_lose_complete_event = {k:sum(v)/len(v) for k, v in lose_complete_event.items()}
        print(f"Average Lose Complete Event: {average_lose_complete_event}")
    
    print(f"\n=== Win Analysis ===")
    print(f"Win State Count: {len(win_set)}")
    average_win_depth, min_win_depth, max_win_depth, average_win_complete_event = 0, 0, 0, {}
    if len(win_set) > 0:
        average_win_depth = sum([len(win_sample.path) for win_sample in win_set]) / len(win_set)
        print(f"Average Win Depth: {average_win_depth}")
        min_win_depth = min([len(win_sample.path) for win_sample in win_set])
        print(f"Min Win Depth: {min_win_depth}")
        max_win_depth = max([len(win_sample.path) for win_sample in win_set])
        print(f"Max Win Depth: {max_win_depth}")
        
        win_complete_event = {k['unique_id']:[] for k in game['events']}
        for win_sample in win_set:
            for event_id, count in win_sample.events_completed.items():
                win_complete_event[event_id].append(count)
        average_win_complete_event = {k:sum(v)/len(v) for k, v in win_complete_event.items()}
        print(f"Average Win Complete Event: {average_win_complete_event}")

    win_list = []
    for i, win_sample in enumerate(win_set):
        win_list.append({
            "id": i+1,
            "path": win_sample.path,
            "depth": len(win_sample.path),
            "events_completed": win_sample.events_completed
        })
    return {
        "terminate_reason":  terminate_reason,
        "explored_states": states_explored,
        "last_depth": depth,
        "unreachable_event": unreachable_event,
        "unreachable_scene": [scene['unique_id'] for scene in game['scenes'] if scene['unique_id'] not in reachable_scene],
        "format_ids_check": validate_unique_ids(game),
        "lose_state_count": len(lose_set),
        "average_lose_depth": average_lose_depth,
        "min_lose_depth": min_lose_depth,
        "max_lose_depth": max_lose_depth,
        "average_lose_complete_event": average_lose_complete_event,
        "win_state_count": len(win_set),
        "average_win_depth": average_win_depth,
        "min_win_depth": min_win_depth,
        "max_win_depth": max_win_depth, 
        "average_win_complete_event": average_win_complete_event,
        "win_list": win_list
    }


def validate_unique_ids(parsed_game):
    results = {}
    for key, prefix in [("scenes", "S"), ("state_variables", "V"), ("hidden_variables", "H"), ("events", "E"), ("pre_event_checks", "P")]:
        ids = [item["unique_id"] for item in parsed_game.get(key, [])]
        correct_format = all(id.startswith(prefix) and id[1:].isdigit() for id in ids)
        sequential = ids == [f"{prefix}{str(i).zfill(3)}" for i in range(1, len(ids) + 1)]
        results[key] = {"correct_format": correct_format, "sequential": sequential}
    return results


def game_bfs_check(game_data, max_states_explored):
    error_output = {
        "terminate_reason":  'error',
        "explored_states": [],
        "last_depth": None,
        "unreachable_event": [],
        "unreachable_scene": [],
        "format_ids_check": None,
        "lose_state_count": None,
        "average_lose_depth": None,
        "min_lose_depth": None,
        "max_lose_depth": None,
        "average_lose_complete_event": None,
        "win_state_count": None,
        "average_win_depth": None,
        "min_win_depth": None,
        "max_win_depth": None, 
        "average_win_complete_event": None,
        "win_list": []
    }
    try:
        if "error" in game_data.keys():
            return error_output
        win_set, lose_set, events, depth, states_explored, terminate_reason = bfs(
            game_data, math.inf, max_states_explored
        )
        return analysis_result(
            game_data, win_set, lose_set, events, depth, states_explored, terminate_reason
        )
    except Exception as e:
        return error_output