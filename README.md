# learning_reinforcement_learning

From 0 to ... in code.


## To Do

- A Deep Policy Gradient method on MountainCar-v0.
- Cross entropy method, like Parameter-Exploring Policy Gradients (PEPG, 2009) or OpenAI approach, on MountainCar-v0
- Recurrent/memory models (to make non-markovian observations more markovian)
- Add 100-episode moving average statistic
- Track best saved model

## Maybe Do

- n-step Q-learning
- n-step Sarsa
- REINFORCE
- A2C
- PPO

- RL course: implement berkeley grid worlds
- RL course: implement berkeley pacman as Gym Env
  - pixel-based implementation: https://github.com/sohamghosh121/PacmanGym
- classic control (openai gym)

- World models
- Curiosity driven exploration


## exp20181216 Vanilla Policy Gradients and Hot Forests

After my frustrations with using OpenAI Baselines deepq on 2x2 hot forests (what was I doing wrong?) and with the coming of the end of the semester, I can finally kick back and write a vanilla policy gradient agent in tensorflow 2.0 style.

It was an exciting moment when I pieced together the training loop and saw the model learn how to play the 2x2 version of the game! Yes, I know, just 2x2, but you got to start somewhere...

There is still a lot of future work to do.


## exp20181217 Vanilla Policy Gradients and Lotka-Volterra Cellular Automata

A cellular automata is implemented herein as a 3x3 convolutional layer followed by N 1x1 convolutional layers.  This network outputs the logits for each cell's next state.  An action is sampled from these logits to produce the next state of the automata.  Therefore the action space is the product (in the algebraic type sense) of the individually sampled "actions" (the next state of the cell).  The stochastic policy, pi(a|s), becomes the mathematical product of the probability of each individual cells actions.  

Anyway, like MoCS HW 3, this formulation did not lead to policies that I thought were good.  No matter the reward function I tried, it seemed like the agent learns policies that lead to the extinction of one or another species (predators or prey).  Also the spatial dynamics are decidedly mixed. Herds of animals are not chasing each other across the tundra. It looks more like static on a screen.

This attempt to get an agent to learn cellular automata rules that follow the orbital dynamics of a Lotka-Volterra system failed for one reason or another.  There is a lot of future work to do to get this to work:

- Try learning known cellular automata rules in order to debug the learning process
- Try using memory or very wide neighborhoods to learn rules that obey orbits
- Try a different loss function, that captures more of the features we want, like non-extinct populations or spatial dynamics, or population dynamics over a longer time period.


## exp20181219 Policy Gradients and MountainCar-v0 and CartPole-v0

Trying my hand at some OpenAI Gym environments. CartPole-v0 is easy. MountainCar-v0 is hard. Added Monte Carlo returns and a number of other features to the Policy Gradient Agent.


## exp20181220 Homebrew Q-Learning and MountainCar-v0

Since policy gradients explores randomly and never gets a reward signal, it never updates its policy. What about Q-learning. In Sutton and Barto Chap 10, they tackle MountainCar with Coarse Coding (Tiling) and ~n-step Sarsa. In this experiment I try 1-step Q-learning with a deep network (1-hidden layer) and with a linear layer on top of a tiling encoding.  The tiling approach worked (after ~40 "epochs" of training) and then got worse after that. The deep network degraded to a linear function, which


## exp20181223 MountainCar-v0, Sarsa, Q-learning, Monte Carlo, Tile Coding and Linear Approximation

Stepping back from the deep networks for a bit, in this experiment I implemented 1-step Sarsa and Q-learning, and Monte Carlo algorithms.  I used a linear approximation model with tile coding to solve the MountainCar-v0 environment. In the Sutton and Barto book, Sarsa is used with tile coding.

This worked well for MountainCar-v0. Applying it to other environments might require a different embedding. For example, it did not initially work for Taxi-v2, an environment designed to test hierarchical learning.


## exp20181228 MountainCar-v0, Evolutionary Strageties type of method

Under Construction.


## exp20190104 MountainCar-v0, Deep Q-Learning

I fixed the Deep Q-Learning implementation from exp20181220, by:

- adding a proper replay buffer
- using a target network for Q(s', a'), which also fixed the gradients.

I tried a fully connected network, which has not yet worked ok. 

I also tried a 2d conv net by embedding the input as a 2d multichannel image,
which worked ok.

I tried more (residual) layers, though only 4 residual blocks. The best
network took ~300,000 training steps.

A 1d conv net worked best of all so far



