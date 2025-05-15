# RPGBench
Evaluation of LLMs as RPG Game Engines


## Overview

We introduce two tasks: Game Creation and Game Simulation.

### Game Creation Entry Points:
Generation: `scripts/game_creation/generation.py`

Usage:
```bash
PYTHONPATH=. python scripts/game_creation/generation.py \
    --model_type openai \
    --model_name_or_path gpt-4o \
    --api_key {api_key} \
    --max_tokens 16384 \
    --output_dir ./data/game_creation_generation
```

Evaluation: `scripts/game_creation/evaluation.py`

Usage:
```bash
PYTHONPATH=. python scripts/game_creation/evaluation.py \
    --games ./data/game_creation_generation/gpt-4o-mini_t0.0_p1.0_m16384.jsonl # output from generation.py
```


### Game Simulation Entry Points:
Generation: `scripts/game_simulation/generation.py`

Usage:
```bash
PYTHONPATH=. python scripts/game_simulation/generation.py \
    --model_type openai \
    --model_name_or_path gpt-4o \
    --api_key {api_key} \
    --max_tokens 16384 \
    --output_dir ./data/game_simulation_generation
```

Evaluation: `scripts/game_simulation/evaluation.py`

Usage (with the debug files as an example):
```bash
PYTHONPATH=. python scripts/game_simulation/evaluation.py \
    --game-file ./data/evaluation_debug/games.jsonl \
    --game-trajectory-file ./data/evaluation_debug/trajectories.jsonl \
    --output-dir data/evaluation_debug/ \
    --model-name gpt-4o \
    --api-key {api_key} \
    --evaluation-tasks ALL
```

## Environment Setup
We provide `conda_env.yml` for environment setup. You can create the environment by running:
```bash
conda env create -f conda_env.yml
```

## Data
Please refer to our paper: "RPGBench: Evaluating Large Language Models as Role-Playing Game Engines" for further details.
### Game Creation
The task of game creation is to generate a game from a Wikipedia page about a character (which will be the main NPC of the game). We provide a dataset of characters in `data/game_creation/characters.jsonl`.
### Game Simulation
We introduce the task of RPG game simulation. Games are in `data/games/games.jsonl`, which are all model-generated games in the Game Creation task with proper validity checks.
### Game Trajactories
Game trajectories of various models are presented in `data/game_simulation_generation/`.

### Hugging Face Dataset
Datasets are also uploaded to [RPGBench on Hugging Face](https://huggingface.co/datasets/DongmingShenDS/RPGBench).

## Models

We currently support API models for generation and evaluation. In order to use local models, we recommend one of the following options

- Host an open-ai compatible API with serving packages (e.g., `vLLM`).
- Use `ray` clusters. We have supporting scripts for this setup in `rpgbench/models/ray/`, but you need to setup your own ray clusters depending on your requirements.
