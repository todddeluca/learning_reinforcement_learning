# Experiment: Policy Gradient Agent with MountainCar-v0 

https://gym.openai.com/envs/MountainCar-v0/


## Installation


Install python packages into a virtual environment:

```
python3 -m venv ~/deploy/2018/learning_reinforcement_learning/venv
source ~/deploy/2018/learning_reinforcement_learning/venv/bin/activate
pip install -r requirements.txt
```

## Usage

```
pythonw agent.py --train [--restore-latest]

pythonw agent.py --test --restore-latest --no-render

pythonw agent.py --visualize --restore-latest

```
