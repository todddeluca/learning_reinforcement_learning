# Experiment: More Deep Q-Learning with MountainCar-v0 

https://gym.openai.com/envs/MountainCar-v0/

## Installation


Install python packages into a virtual environment:

```
python3 -m venv ~/deploy/2018/learning_reinforcement_learning/venv
source ~/deploy/2018/learning_reinforcement_learning/venv/bin/activate
pip install -r requirements.txt
```

## Usage

Use `pythonw` instead of `python` on Mac OS X.

```
pythonw agent.py --train --restore-latest
pythonw agent.py --test --restore-latest
pythonw agent.py --test --no-render # quickly evaluate model performance
pythonw agent.py --test --restore-model=<path-to-checkpoint>
pythonw agent.py --visualize --restore-latest # plot q-values for mountain car problem

```
