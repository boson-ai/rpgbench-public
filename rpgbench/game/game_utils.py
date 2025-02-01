import json
import random
from importlib import resources
from rpgbench.utils.json_schema import verify_and_extract

with resources.open_text('rpgbench.static', 'game_json_schema.json') as f:
    _game_schema = json.load(f)

with resources.open_text('rpgbench.static', 'game_state_json_schema.json') as f:
    _game_state_schema = json.load(f)
    
with resources.open_text('rpgbench.static', 'game_event_plan_json_schema.json') as f:
    _game_event_plan_schema = json.load(f)

with resources.open_text('rpgbench.static', 'game_simulation_screenplay.txt') as f:
    _game_screenplay_example = f.read()

with resources.open_text('rpgbench.static', 'game_simulation_prompt.txt') as f:
    _game_prompt = f.read()


def verify_and_extract_game(json_string):
    """
    Attempts to parse a JSON string representing a game and validates it against the predefined schema.
    
    Parameters:
        json_string (str): The JSON string to be parsed and validated.
    
    Returns:
        tuple: A tuple containing an error and the parsed JSON object. If the error is None, the JSON object is valid.
    
    Raises:
        ValueError: If failed to parse JSON string using available methods
    """

    return verify_and_extract(json_string, _game_schema)


def verify_and_extract_game_state(json_string):
    """
    Attempts to parse a JSON string representing a game state and validates it against the predefined schema.
    
    Parameters:
        json_string (str): The JSON string to be parsed and validated.
    
    Returns:
        tuple: A tuple containing an error and the parsed JSON object. If the error is None, the JSON object is valid.
    
    Raises:
        ValueError: If failed to parse JSON string using available methods
    """

    return verify_and_extract(json_string, _game_state_schema)


def verify_and_extract_game_event_plan(json_string):
    """
    Attempts to parse a JSON string representing a game event output and validates it against the predefined schema.
    
    Parameters:
        json_string (str): The JSON string to be parsed and validated.
    
    Returns:
        tuple: A tuple containing an error and the parsed JSON object. If the error is None, the JSON object is valid.
    
    Raises:
        ValueError: If failed to parse JSON string using available methods
    """

    return verify_and_extract(json_string, _game_event_plan_schema)


def format_game(game):
    def format_scenes(scenes):
        formatted = ""
        for scene in scenes:
            formatted += f"### Scene Name: {scene.get('scene_name', 'N/A')}\n"
            formatted += f"- **Unique ID**: {scene.get('unique_id', 'N/A')}\n"
            formatted += f"- **Scene Description**: {scene.get('background_description', 'N/A')}\n"
            formatted += f"- **Scene Type**: {scene.get('scene_type', 'N/A')}\n\n"
        return formatted

    def format_values(values):
        formatted = ""
        for value in values:
            formatted += f"### Value Name: {value.get('value_name', 'N/A')}\n"
            formatted += f"- **Unique ID**: {value.get('unique_id', 'N/A')}\n"
            formatted += f"- **Description**: {value.get('description', 'N/A')}\n\n"
            formatted += f"- **Initial Value**: {value.get('initial_value', 'N/A')}\n"
            formatted += f"- **Min Value**: {value.get('min_value', 'N/A')}\n"
            formatted += f"- **Final Value**: {value.get('max_value', 'N/A')}\n"
        return formatted
    
    def format_events(events):
        formatted = ""
        for event in events:
            entering_conditions = event.get('entering_condition', [])
            if len(entering_conditions) > 0:
                entering_conditions_str = " AND ".join(entering_conditions)
            else:
                entering_conditions_str = "N/A"
            succeed_conditions = event.get('succeed_condition', [])
            if len(succeed_conditions) > 0:
                succeed_conditions_str = " AND ".join(succeed_conditions)
            else:
                succeed_conditions_str = "N/A"
            succeed_effects = event.get('succeed_effect', [])
            if len(succeed_effects) > 0:
                succeed_effects_str = " AND ".join(succeed_effects)
            else:
                succeed_effects_str = "N/A"
            fail_effects = event.get('fail_effect', [])
            if len(fail_effects) > 0:
                fail_effects_str = " AND ".join(fail_effects)
            else:
                fail_effects_str = "N/A"
            formatted += f"### Event Name: {event.get('event_name', 'N/A')}\n"
            formatted += f"- **Unique ID**: {event.get('unique_id', 'N/A')}\n"
            formatted += f"- **Entering Condition**: {entering_conditions_str}\n"
            formatted += f"- **Succeed Condition**: {succeed_conditions_str}\n"
            formatted += f"- **Succeed Effect**: {succeed_effects_str}\n"
            formatted += f"- **Fail Effect**: {fail_effects_str}\n\n"
            formatted += f"- **Remark**: {event.get('explanations', 'N/A')}\n"
        return formatted
    
    def format_rules(rules):
        formatted = ""
        for rule in rules:
            conditions = rule.get('condition', [])
            if len(conditions) > 0:
                conditions_str = " AND ".join(conditions)
            else:
                conditions_str = "N/A"
            effects = rule.get('effect', [])
            if len(effects) > 0:
                effects_str = " AND ".join(effects)
            else:
                effects_str = "N/A"
            formatted += f"### Rule Name: {rule.get('check_name', 'N/A')}\n"
            formatted += f"- **Unique ID**: {rule.get('unique_id', 'N/A')}\n"
            formatted += f"- **Condition**: {conditions_str}\n"
            formatted += f"- **Effect**: {effects_str}\n\n"
            formatted += f"- **Remark**: {rule.get('explanation', 'N/A')}\n"
        return formatted
            

    sections = []
    
    player_name = game.get('player_name')
    player_desc = game.get('player_description')
    if player_name and player_desc:
        sections.append("## Player Character")
        sections.append(f"**Name**: {player_name}")
        sections.append(f"**Description**: {player_desc}")
        sections.append("")

    main_npc_name = game.get('main_npc_name', 'No Main NPC')
    main_npc_desc = game.get('main_npc_description', 'No Main NPC')
    if main_npc_name != 'No Main NPC':
        sections.append("## Main NPC")
        if main_npc_name:
            sections.append(f"**Name**: {main_npc_name}")
        if main_npc_desc:
            sections.append(f"**Description**: {main_npc_desc['text']}")
            sections.append(f"**Personality Traits**:")
            for trait, value in main_npc_desc['big5_personality_traits'].items():
                sections.append(f"**{trait}**:")
                sections.append(f"Rating (1 - 5): {value['rate']}")
                sections.append(f"Description: {value['description']}")
            if len(main_npc_desc['additional_facts']) > 0:
                sections.append(f"**NPC Facts**")
                for fact in main_npc_desc['additional_facts']:
                    sections.append(fact)
        sections.append("")

    game_world = game.get('game_world')
    if game_world:
        sections.append("## Game World")
        sections.append(f"{game_world.strip()}")
        sections.append("")

    objectives = game.get('game_objectives')
    if objectives:
        sections.append("## Game Objectives")
        sections.append(f"{objectives.strip()}")
        sections.append("")

    scenes = game.get('scenes')
    if scenes:
        sections.append("## Scenes")
        sections.append(format_scenes(scenes))
        sections.append("")

    state_variables = game.get('state_variables')
    if state_variables:
        sections.append("## State Variables")
        sections.append(format_values(state_variables))
        sections.append("")

    hidden_variables = game.get('hidden_variables')
    if hidden_variables:
        sections.append("## Hidden Variables")
        sections.append(format_values(hidden_variables))
        sections.append("")
    
    events = game.get('events')
    if events:
        sections.append("## Events")
        sections.append(format_events(events))
        sections.append("")
    
    rules = game.get('pre_event_check')
    if rules:
        sections.append("## Rules to Check Before Next Event")
        sections.append(format_rules(rules))
        sections.append("")

    game_formatted = "\n".join(sections)
    return game_formatted

def prepare_game_prompt(game):
    """
    Generates a prompt for a large language model to act as an RPG game engine based on the provided game information.

    Args:
        game (dict): A dictionary containing game-related information such as characters, world details, objectives, scenes, and value systems.

    Returns:
        str: A formatted string prompt that includes the organized game information and instructions for the LLM to generate the game output in the specified format.
    """

    schema = json.dumps(_game_state_schema, indent=2, ensure_ascii=False)
    event_plan_schema = json.dumps(_game_event_plan_schema, indent=2, ensure_ascii=False)
    example_screenplay = _game_screenplay_example
    game_formatted = format_game(game)
    prompt = _game_prompt.format(schema=schema, example_screenplay=example_screenplay, game_formatted=game_formatted, event_plan_schema=event_plan_schema)
    return prompt


def parse_game_output(game_output):
    """
    Parses the LLM output into Game and State sections based on delimiters.

    Args:
        game_output (str): The complete output string from the LLM.

    Returns:
        tuple: A tuple containing the Game section and the State as a Python dictionary.
    """
    event_plan_start = "===EVENT PLAN START==="
    event_plan_end = "===EVENT PLAN END==="
    game_start = "===GAME START==="
    game_end = "===GAME END==="
    state_start = "===STATE START==="
    state_end = "===STATE END==="
    
    try:
        event_plan = game_output.split(event_plan_start)[1].split(event_plan_end)[0].strip()
        if event_plan.startswith("```json"):
            event_plan = event_plan[len("```json"):]
        if event_plan.endswith("```"):
            event_plan = event_plan[:-len("```")]
        _, event_plan = verify_and_extract_game_event_plan(event_plan.strip())
        if _ is not None:
            event_plan = {}
    except (ValueError, Exception):
        _, event_plan = None, {}

    try:
        game_section = game_output.split(game_start)[1].split(game_end)[0].strip()
    except (IndexError, Exception):
        game_section = ""

    try:
        state_raw = game_output.split(state_start)[1].split(state_end)[0].strip()
        if state_raw.startswith("```json"):
            state_raw = state_raw[len("```json"):]
        if state_raw.endswith("```"):
            state_raw = state_raw[:-len("```")]
        _, state = verify_and_extract_game_state(state_raw.strip())
        if _ is not None:
            state = {}
    except (ValueError, Exception):
        _, state = None, {}
    
    state['event_plan'] = event_plan

    return game_section, state

def format_game_short(game):
    sections = []
    
    player_name = game.get('player_name')
    player_desc = game.get('player_description')
    if player_name and player_desc:
        sections.append("## Player Character")
        sections.append(f"**Name**: {player_name}")
        sections.append(f"**Description**: {player_desc}")
        sections.append("")

    main_npc_name = game.get('main_npc_name', 'No Main NPC')
    main_npc_desc = game.get('main_npc_description', 'No Main NPC')
    if main_npc_name != 'No Main NPC':
        sections.append("## Main NPC")
        if main_npc_name:
            sections.append(f"**Name**: {main_npc_name}")
        if main_npc_desc:
            sections.append(f"**Personality Traits**:")
            for trait, value in main_npc_desc['big5_personality_traits'].items():
                sections.append(f"**{trait}**:")
                sections.append(f"Rating (1 - 5): {value['rating']}")
                sections.append(f"Description: {value['description']}")
            if len(main_npc_desc['additional_facts']) > 0:
                sections.append(f"**NPC Facts**")
                for fact in main_npc_desc['additional_facts']:
                    sections.append(fact)
        sections.append("")

    game_world = game.get('game_world')
    if game_world:
        sections.append("## Game World")
        sections.append(f"{game_world.strip()}")
        sections.append("")

    objectives = game.get('game_objectives')
    if objectives:
        sections.append("## Game Objectives")
        sections.append(f"{objectives.strip()}")
        sections.append("")
    
    return "\n".join(sections)


