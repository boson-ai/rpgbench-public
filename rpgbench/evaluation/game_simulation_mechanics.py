import json
from rpgbench.game import parse_game_output
import copy
from typing import Dict, List, Optional, Callable
import numpy as np


class GameLoader:
    def __init__(self, json_obj):
        self.data = json_obj
        self.state_variables = {var["value_name"]: int(var.get("initial_value", 0)) for var in self.data["state_variables"]}
        self.hidden_variables = {var["value_name"]: int(var.get("initial_value", 0)) for var in self.data["hidden_variables"]}
        self.var_mapping = {}  # Map from id to variable name
        for key in ["state_variables", "hidden_variables"]:
            for var in self.data.get(key, []):  # Use .get() to safely handle missing keys
                self.var_mapping[var["unique_id"]] = var["value_name"]
        self.scenes = {scene["unique_id"]: scene for scene in self.data["scenes"]}
        self.start_scene = self.get_start_scene()
        self.events = {event["unique_id"]: event for event in self.data["events"]}
        self.pre_event_checks = {checks["unique_id"]: checks for checks in self.data["pre_event_checks"]}

    def load_json(self, json_file):
        with open(json_file, "r") as file:
            return json.load(file)

    def get_start_scene(self):
        """Identify the unique start scene and ensure there is exactly one."""
        start_scenes = [scene for scene in self.data["scenes"] if scene.get("scene_type") == "Start Scene"]
        if len(start_scenes) != 1:
            raise ValueError(f"Expected exactly one start scene, found {len(start_scenes)}")
        return start_scenes[0]["unique_id"]  # Return the unique_id of the start scene

    def get_game_state(self):
        """Initialize the game state."""
        return {
            "state_variables": self.state_variables.copy(),
            "hidden_variables": self.hidden_variables.copy(),
        }


class StateWrapper:
    def __init__(self, data):
        self._data = data  # Store the original dictionary internally

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'StateWrapper' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "_data":  # Allow setting the internal dictionary
            super().__setattr__(name, value)
        elif name in self._data:
            self._data[name] = value
        else:
            raise AttributeError(f"'StateWrapper' object has no attribute '{name}'")

    def __repr__(self):
        return repr(self._data)


class GameChecker:
    def __init__(self, loader):
        self.loader = loader
        self.scenes = loader.scenes
        self.events = loader.events
        self.pre_event_checks = loader.pre_event_checks
        self.visited_states = set()
        self.reachable_scenes = set()
        self.variable_ranges = self.get_variable_ranges()

    def get_variable_ranges(self):
        """Extract min and max ranges for all state and hidden variables."""
        ranges = {}
        for var in self.loader.data["state_variables"]:
            ranges[var["value_name"]] = (int(var["min_value"]), int(var["max_value"]))
        for var in self.loader.data["hidden_variables"]:
            ranges[var["value_name"]] = (int(var["min_value"]), int(var["max_value"]))
        return ranges

    def validate_ranges(self, state):
        """Check if all variables are within their specified min/max ranges."""
        for var_name, (min_val, max_val) in self.variable_ranges.items():
            if var_name in state["state_variables"]:
                value = state["state_variables"][var_name]
                if not (min_val <= value <= max_val):
                    return False
            elif var_name in state["hidden_variables"]:
                value = state["hidden_variables"][var_name]
                if not (min_val <= value <= max_val):
                    return False
        return True

    def ensure_valid_ranges(self, state):
        """Ensure all variables are within their min/max ranges."""
        for var_name, (min_val, max_val) in self.variable_ranges.items():
            if var_name in state["state_variables"]:
                state["state_variables"][var_name] = max(
                    min_val, min(state["state_variables"][var_name], max_val)
                )
            elif var_name in state["hidden_variables"]:
                state["hidden_variables"][var_name] = max(
                    min_val, min(state["hidden_variables"][var_name], max_val)
                )

    def evaluate_conditions(self, conditions, state):
        """Evaluate whether all conditions for an event are satisfied."""
        state_variables = StateWrapper(state["state_variables"])
        hidden_variables = StateWrapper(state["hidden_variables"])
        for condition in conditions:
            if not eval(condition, {"v": state_variables, "h": hidden_variables}):
                return False
        return True

    def apply_effects(self, effects, state):
        """Apply event effects to update state variables."""
        state_variables = StateWrapper(state["state_variables"])
        hidden_variables = StateWrapper(state["hidden_variables"])
        for effect in effects:
            exec(effect, {"v": state_variables, "h": hidden_variables})
    
    def run_pre_event_checks(self, state):
        """Run all pre_event_checks to see if any game finishes"""
        if_exit = False
        for checks in self.pre_event_checks.values():
            if self.evaluate_conditions(checks.get("condition", []), state):
                self.apply_effects(checks.get("effect", []), state)
                if_exit |= (checks.get("effect", "Exit") == "Exit")  # assume all checks are Exit 
        return state, if_exit

    def parse_state(self, state):
        state_variables = {self.loader.var_mapping[var["value_id"]]: int(var.get("current_value", 0)) for var in state["state_variables"] if var["value_id"] in self.loader.var_mapping}
        hidden_variables = {self.loader.var_mapping[var["value_id"]]: int(var.get("current_value", 0)) for var in state["hidden_variables"] if var["value_id"] in self.loader.var_mapping}
        return {
            "state_variables": state_variables.copy(),
            "hidden_variables": hidden_variables.copy(),
        }

    def check_state_update(self, curr, prev, prev_valid, err, debug=False):
        state_var_err_range, hidden_var_err_range = 0, 0
        state_var_err_complete, hidden_var_err_complete = 0, 0
        for var in self.loader.state_variables: 
            if curr['state_variables'][var] == prev_valid['state_variables'][var]:
                continue
            elif curr['state_variables'][var] == prev['state_variables'][var]:
                if debug: print(f"ERROR: variable **{var}** matches between prev & curr, but not in valid range.")
                state_var_err_range += 1
            else: 
                if debug: print(f"ERROR: variable **{var}** does not match between prev & curr.")
                state_var_err_complete += 1
        for var in self.loader.hidden_variables:
            if curr['hidden_variables'][var] == prev_valid['hidden_variables'][var]:
                continue
            elif curr['hidden_variables'][var] == prev['hidden_variables'][var]:
                if debug: print(f"ERROR: variable **{var}** matches between prev & curr, but not in valid range.")
                hidden_var_err_range += 1
            else: 
                if debug: print(f"ERROR: variable **{var}** does not match between prev & curr.")
                hidden_var_err_complete += 1
        err[-1]["state_var_update_error_range_invalid"] += state_var_err_range
        err[-1]["hidden_var_update_error_range_invalid"] +=  hidden_var_err_range
        err[-1]["state_var_update_error_complete_mismatch"] += state_var_err_complete
        err[-1]["hidden_var_update_error_complete_mismatch"] += hidden_var_err_complete

    def validate_game_per_round(self, trajectory, debug=False):
        """
        1. Perform pre-event checks to check if game is terminating prematurely or not terminating when it should have.
        2. For non-"OTHER" events with type "Start," validate that the event's "entering_condition" is satisfied.
        3. For non-"OTHER" events with type "End" and outcome "Success," ensure the event's "succeed_condition" is met, then apply the "succeed_effect."
        4. For non-"OTHER" events with type "End" and outcome "Fail," confirm that the event's "succeed_condition" is *not* met, then apply the "fail_effect."
        5. After applying effects to the previous round's state, extract the current round's state and validate consistency: ensure prev_state + curr_round_effects == curr_state 
        """
        max_error_dict = {
            "pre_event_check_error": len(trajectory),
            "event_condition_check_error": 0,
        }
        all_errors = []
        prev_state = self.loader.get_game_state()
        for round_id in range(len(trajectory)):
            if debug: print(f"=================================START OF ROUND {round_id}===============================")
            error_dict = {
                "num_events": 0,
                "general_format_error": 0,
                "pre_event_check_error": 0,
                "event_condition_check_error": 0,
                "state_var_update_error_range_invalid": 0,
                "hidden_var_update_error_range_invalid": 0,
                "state_var_update_error_complete_mismatch": 0,
                "hidden_var_update_error_complete_mismatch": 0,
            }
            all_errors.append(error_dict)
            # extract current state, all events in this round, and next state if available
            curr_state = parse_game_output(trajectory[round_id]['content'])[1]
            if not curr_state or (len(curr_state) == 1 and 'event_plan' in curr_state): 
                if debug: print(f"ERROR: curr_state parse failed.")
                all_errors[-1]["general_format_error"] += 1 
                return all_errors, max_error_dict
            if any(var not in self.parse_state(curr_state)['state_variables'] for var in self.loader.state_variables) or any(var not in self.parse_state(curr_state)['hidden_variables'] for var in self.loader.hidden_variables):
                if debug: print(f"ERROR: curr_state missing some variables.")
                all_errors[-1]["general_format_error"] += 1
                return all_errors, max_error_dict
            events = curr_state.get('event_plan', [])  # extract event from output
            if debug: print("Events:", [f"{e['event_id']}-{e['type']}" for e in events])
            events = [e for e in events if e['event_id'] != 'OTHER']  # skip OTHER
            max_error_dict["event_condition_check_error"] += len(events) * 2  # each event can make 2 errors at most
            error_dict["num_events"] = len(events)
            curr_state = self.parse_state(curr_state)
            # 1. run pre-event checks
            prev_state, if_exit = self.run_pre_event_checks(prev_state)
            if round_id + 1 < len(trajectory) and if_exit:
                if debug: print(f"ERROR: Round {round_id} should have reached end-game already, but still have next round")
                all_errors[-1]["pre_event_check_error"] += 1
            elif round_id == len(trajectory) - 1 and not if_exit:
                if debug: print(f"ERROR: Round {round_id} did not reached end-game yet, but does not have next round")
                all_errors[-1]["pre_event_check_error"] += 1
            # handle all events
            for event in events: 
                event_id = event['event_id']
                if event_id not in self.events: 
                    if debug: print(f"ERROR: event {event_id} not found in all event ids")
                    all_errors[-1]["general_format_error"] += 1  # can be more, but we can avoid this with json schema check
                    return all_errors, max_error_dict
                # 2. events with type "Start", we validate the event's "entering_condition" is met
                if event['type'] == 'Start': 
                    enterable = self.evaluate_conditions(self.events[event_id].get("entering_condition", []), prev_state)
                    if not enterable: 
                        if debug: print(f"ERROR: event {event_id} does not match entering_condition")
                        all_errors[-1]["event_condition_check_error"] += 1
                        continue
                # 3. events with type "End" and outcome "Success", we validate the event's "succeed_condition" is met, then apply "succeed_effect"
                elif event['type'] == 'End' and event['outcome'] == 'Success': 
                    succeed = self.evaluate_conditions(self.events[event_id].get("succeed_condition", []), prev_state)
                    if not succeed: 
                        if debug: print(f"ERROR: event {event_id} does not match succeed_condition, but has outcome Success")
                        all_errors[-1]["event_condition_check_error"] += 1
                        continue
                    self.apply_effects(self.events[event_id].get("succeed_effect", []), prev_state)
                # 4. events with type "End" and outcome "Fail", we validate the event's "succeed_condition" is not met, then apply "fail_effect"
                elif event['type'] == 'End' and event['outcome'] == 'Fail': 
                    succeed = self.evaluate_conditions(self.events[event_id].get("succeed_condition", []), prev_state)
                    if succeed: 
                        if debug: print(f"ERROR: event {event_id} matches succeed_condition, but has outcome Fail")
                        all_errors[-1]["event_condition_check_error"] += 1
                        continue
                    self.apply_effects(self.events[event_id].get("fail_effect", []), prev_state)
                else: 
                    if debug: print(f"ERROR: unreadable event combination of type {event['type']} & outcome {event['outcome']}")
                    all_errors[-1]["general_format_error"] += 1
                    return all_errors, max_error_dict
            # 5. finished all event execution, check if state match
            prev_state_valid_range = copy.deepcopy(prev_state)
            self.ensure_valid_ranges(prev_state_valid_range)
            self.check_state_update(curr_state, prev_state, prev_state_valid_range, all_errors)
            if debug: print(f"=================================END OF ROUND {round_id}=================================")
            # update prev_state
            prev_state = self.parse_state(parse_game_output(trajectory[round_id]['content'])[1])
        return all_errors, max_error_dict

    def process_errors(self, all_errors, max_errors, trajectory, weights=(1,1,1)):
        """
        Helper function to parse and print error information.
        
        At each round, there are 3 main kinds of error types (in addition to format-err):
        1. pre_event_check: can happen max=1 per round
        2. event_condition: can happen #events * 2 per round (enter condition + success/fail condition)
        3. var_update: can happen #state + #hidden per round. TYPES: strict (with range check) or normal (just match check)

        For each error type, we get a error rate number between [0,1], and combine with weights (which also sums to 1.0)
        
        This gives one error value for each round, then we pool over the trajectory to get a single number.
        """
        weights = tuple(weights[i] / sum(weights) for i in range(3))
        if all_errors[-1]['general_format_error']:
            all_errors.pop()
        terminate_with_err = (len(trajectory) > len(all_errors))
        # Summary stats of errors - implement what I say in the function comments
        scores_raw, score_strict = [], []
        event_condition_average, var_update_average, var_update_strict_average = 0, 0, 0
        for round_idx, errors in enumerate(all_errors):
            pre_event_check = 0#errors['pre_event_check_error']
            assert 0 <= pre_event_check <= 1
            event_condition = errors['event_condition_check_error'] / max(1, errors['num_events'])
            assert 0 <= event_condition <= 1
            var_update = (errors['state_var_update_error_complete_mismatch'] + errors['hidden_var_update_error_complete_mismatch']) / (len(self.loader.hidden_variables) + len(self.loader.state_variables))
            var_update_strict = var_update + (errors['state_var_update_error_range_invalid'] + errors['hidden_var_update_error_range_invalid']) / (len(self.loader.hidden_variables) + len(self.loader.state_variables))
            assert 0 <= var_update <= 1 and 0 <= var_update_strict <= 1
            scores_raw.append(pre_event_check * weights[0] + event_condition * weights[1] + var_update * weights[2])
            score_strict.append(pre_event_check * weights[0] + event_condition * weights[1] + var_update_strict * weights[2])
            event_condition_average += event_condition
            var_update_average += var_update
            var_update_strict_average += var_update_strict
        return {
            "weights": weights,
            "terminated_with_format_error": terminate_with_err,
            "num_rounds": len(trajectory),
            "num_valid_rounds": len(all_errors),
            "per_round_error": scores_raw,
            "per_round_error_strict": score_strict,
            "final_error_rate": 1 if len(scores_raw) == 0 else np.mean(scores_raw),
            "final_error_rate_strict": 1 if len(scores_raw) == 0 else np.mean(score_strict),
            "final_error_free_rounds": sum(x == 0 for x in scores_raw) / max(1, len(scores_raw)),
            "final_error_free_rounds_strict": sum(x == 0 for x in score_strict) / max(1, len(score_strict)),
            "pre_event_check_error": sum(x['pre_event_check_error'] for x in all_errors),
            "event_condition_check_error": event_condition_average / max(1, len(all_errors)),
            "var_update_error": var_update_average / max(1, len(all_errors)),
            "var_update_error_strict": var_update_strict_average / max(1, len(all_errors)),
        }


def evaluation_mechanics(
    history: Optional[List[str]] = None, 
    response: Optional[str] = None, 
    game: Optional[Dict] = None, 
    llm_judge: Optional[Callable] = None
):
    assert history and game, "Trajectory and Game must be provided"
    trajectory = [msg for msg in history if msg['role'] == 'assistant']
    loader = GameLoader(game)
    checker = GameChecker(loader)
    all_errors, max_error_dict = checker.validate_game_per_round(trajectory)
    error_summary = checker.process_errors(all_errors, max_error_dict, trajectory, weights=(1,1,1))
    return all_errors, error_summary