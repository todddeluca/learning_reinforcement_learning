# Experiment: Policy Gradients LVCA

Here the game is to have a cellular automata learn rules which follow the 
predator-prey population dynamics of a Lotka-Volterra differential function system.
ODEs.

Also on the list are some improvements to the policy gradients agent and training
code: tensorflow metrics and checkpoints


## Installation

You might need to install ffmpeg to get saving movies to work.

Install python packages into a virtual environment:

```
python3 -m venv ~/deploy/2018/learning_reinforcement_learning/venv
source ~/deploy/2018/learning_reinforcement_learning/venv/bin/activate
pip install -r requirements.txt
```

## Usage

```
python train.py

```
