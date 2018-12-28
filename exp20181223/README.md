# Experiment: Value Agents, Tile Coding and MountainCar-v0

https://gym.openai.com/envs/MountainCar-v0/

An implementation of Monte Carlo and a 1-step implementation of Sarsa and Q-learning was used
to "solve" MountainCar-v0, scoring > -110 reward per episode on average for 100 episodes. See https://github.com/openai/gym/wiki/Leaderboard for the winning conditions of MountainCar and
other environments.

## Installation


Install python packages into a virtual environment:

```
python3 -m venv ~/deploy/2018/learning_reinforcement_learning/venv
source ~/deploy/2018/learning_reinforcement_learning/venv/bin/activate
pip install -r requirements.txt
```

## Usage

```
pythonw agent.py --train --restore-latest
pythonw agent.py --test --restore-latest
pythonw agent.py --visualize --restore-latest # plot q-values for mountain car problem

```
