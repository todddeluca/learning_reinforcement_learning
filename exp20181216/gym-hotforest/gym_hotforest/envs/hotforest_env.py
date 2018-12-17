import gym
from gym import spaces
# from gym import error, utils
from gym.utils import seeding

import numpy as np
import skimage.measure
import sys


# Useful references for creating a custom gym environment
# gym.Env: https://github.com/openai/gym/blob/master/gym/core.py
# Example environment -- soccer: https://github.com/openai/gym-soccer/blob/master/gym_soccer/envs/soccer_env.py
# https://stackoverflow.com/questions/45068568/is-it-possible-to-create-a-new-gym-environment-in-openai
# Gym README: https://github.com/openai/gym/
# Gym Docs: https://gym.openai.com/docs/

def make_board(length):
    '''
    return: 2d array of zeros (an empty board)
    '''
    board = np.zeros((length, length), dtype=int)
    return board


def spark_probs(length, l_factor=10):
    '''
    Where will the spark land on the 2d board?
    p(i,j) = e^{-i/l} * e^{-j/l}, where l = length / l_factor.

    return: 2d array of board position spark probability distribution
    '''
    scale = length / l_factor
    idxs = np.indices((length, length))
    probs = np.exp(-1 * idxs[0] / scale) * np.exp(-1 * idxs[1] / scale).astype(np.float64)
    return probs / np.sum(probs) # normalize probs so it sums to (almost) 1.0


def board_costs(board):
    '''
    For every position on the board, calculate the cost as number of trees
    which will burn if a fire starts at that position.

    The connected components (forests) of the board are found, the size of each component is found,
    and each position in the board is annotated with the size of the component it is a member of (or 0)
    if it is empty (not a part of the component).

    returns: costs, same shape as board. each position contains the number of trees that
    would burn if a fire started at that position.
    '''
    labels = skimage.measure.label(board, connectivity=1).ravel() # label each connected component with an id
    counts = np.bincount(labels) # count the size of each component
    counts[0] = 0 # treeless squares (labels==0) are NOT a big forest, despite what bincount might imply.
    costs = counts[labels].reshape(board.shape)
    return costs


def sample_yield(board, ps, n=None):
    '''
    Calculate the yield, which is the fraction of the board planted in trees minus the
    expected fraction of the board that will burn from a spark starting a fire

    ps: spark probability distribution
    n: number of samples to average over (defaults to a single sample). n > 1 means multiple sparks
     will be sampled and the average yield will be returned. If n is None, the true expected spark cost
     is used.
    return: a sample yield, the percentage of the board that is planted in trees minus the percentage of
      the board that burns as a result of a sampled spark.
    '''
    costs = board_costs(board)
    if n is None:
        exp_cost = np.sum(ps * costs)
    else:
        # light n sparks and average how many trees burn
        exp_cost = np.mean(np.random.choice(costs.ravel(), size=n, replace=True, p=ps.ravel()))

    yld = (np.sum(board) - exp_cost) / np.product(board.shape) # yield is a fraction of board size
    return yld


class HotforestEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, length=4, l_factor=10, num_yield_samples=None):
        '''
        length: length of one side of the square board
        l_factor: scaling factor affecting spark probability distribution.
        num_yield_samples: None returns true expected yield (using spark
        probability dist). An integer (e.g. 1, 20) returns a yield calculated
        from that many spark samples.
        '''
        self.length = length # how to pass configuration to environment?
        self.l_factor = l_factor
        self.num_yield_samples = num_yield_samples

        self.ps = spark_probs(self.length, self.l_factor)
        # actions are planting a tree at any position on the board as a 1d vector
        self.action_space = spaces.Discrete(self.length**2)
        # observation is the board as a 2d matrix of trees and empty spaces.
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.length, self.length), dtype=np.uint8)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.t += 1
        
        # print('action:', action, 'board:', self.board.ravel())
        # convert 1d action (tree position) into 2d board position tuple
        # for a 4x4 board, action 0 -> (0,0) and action 6 -> (1,2)
        tree = np.unravel_index(action, self.board.shape)

        # GAME VERSION: end game for illegal moves
        # end the game for illegal moves
        if self.board[tree] == 1:
            reward = -10 # reward for "losing" the game by overplanting a tree
            done = True
        else:
            # plant tree, calc reward
            self.board[tree] = 1
            reward = sample_yield(self.board, self.ps, n=self.num_yield_samples)
            # episode is done when board is full or max_t timesteps have elapsed.
            done = self.t >= self.max_t or np.sum(self.board) == self.board_size

        return (self.board, reward, done, {})

        # reward
        if self.board[tree] == 1:
#             reward = 0 # large negative reward for planting a tree on top of an existing tree
            reward = sample_yield(self.board, self.ps, n=self.num_yield_samples) # same reward for overplanting tree
        else:
            # plant tree, calc reward
            self.board[tree] = 1
            reward = sample_yield(self.board, self.ps, n=self.num_yield_samples)

        # episode is done after max_t tree plantings.
#         done = self.t >= self.max_t

        # episode is done when board is full or max_t timesteps have elapsed.
        done = self.t >= self.max_t or np.sum(self.board) == self.board_size
        
        # reward for completing the board
        if done and np.sum(self.board) == self.board_size:
#             reward += self.max_t # full board bonus to encourage not planting over trees.
            pass

#         # episode is done when board is fully planted with trees
#         done = np.sum(self.board) == np.product(self.board.shape)
#         print(f't: {self.t} max_t: {self.max_t} reward: {reward:0.2}', 'done:', done, 'board:', self.board.ravel())

#         reward += -1.0 # penalty for dawdling
        return (self.board, reward, done, {})


    def reset(self):
        '''
        Reset the environment state. Return initial observation.
        '''
        self.board = make_board(self.length) # empty the board
        self.board_size = np.product(self.board.shape)
        self.t = 0
#         self.max_t = 1000 # allow game to go on for a long time to teach agent to end it early
#         self.max_t = np.product(self.board.shape) + 100 # length of a "game" is max_t tree plantings
        self.max_t = np.product(self.board.shape) # length of a "game" is max_t tree plantings
        return self.board

    def render(self, mode='human'):
        '''
        For humans, visualize something. For others, return data.
        '''
        outfile = sys.stdout
        print(self.board, file=outfile)
