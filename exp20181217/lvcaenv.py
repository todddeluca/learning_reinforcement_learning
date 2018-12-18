import argparse
import numpy as np


DEBUG = False

'''
On the woes of scaling population deltas:

It is possible, I think, to get populations to remain within the range [0, 1], and possibly to get
the populations to sum to 1, by fiddling with the differential equations.

See:
- https://en.wikipedia.org/wiki/Competitive_Lotka%E2%80%93Volterra_equations
- http://www.tiem.utk.edu/~gross/bioed/bealsmodules/competition.html

I have not been able to figure out:
- how to do this
- how to get orbiting dynamics. 

Future work. :-)

The problem with target_deltas, the predicted changes from the lotka-volterra differential equations,
is that they predict absolute population changes larger than the size of the board. They do not
"sum to one".  This is illustrated on this small 4x4 board:
[[2 0 1 2]
 [2 0 1 0]
 [2 0 2 2]
 [0 2 1 2]]
The next observed board is:
[[2 0 1 2]
 [2 0 1 0]
 [2 0 2 2]
 [0 2 1 2]]
Therefore the populations of [empty, prey, and predator] states are:
pops: [5 3 8] next_pops: [5 3 8]
The observed change are, well, none in this case, and the target changes are larger than
possible on a board with only 16 cells:
obs_deltas: [0 0] target_deltas: [-30.  16.]
This leads to a VERY large mean squared error.
mse reward: 578.0

This is the thinking behind the cosine lotka reward, which ignores the size of a change in
favor of a direction
'''

def make_scaled_rmse_lotka_reward(alpha, beta, delta, gamma, num_states=3):
    def reward_func(obs, next_obs):
        obs_deltas, target_deltas = get_scaled_lotka_deltas(obs, next_obs, alpha, beta, delta, gamma, num_states)

        # reward: - root mean squared error
        reward = -np.sqrt(np.mean((obs_deltas - target_deltas)**2))

        if DEBUG:
            print('reward:', reward)

        return reward
    return reward_func
    

def get_scaled_lotka_deltas(obs, next_obs, alpha, beta, delta, gamma, num_states):
    # populations as fraction of board covered
    size = np.product(obs.shape)
    pops = (np.bincount(obs.ravel(), minlength=num_states) / size)[1:]
    next_pops = (np.bincount(next_obs.ravel(), minlength=num_states) / size)[1:]

    # staring at the orbits at
    # https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations#Phase-space_plot_of_a_further_example
    # I want populations to be in the range 0..3 for predator and 0..5 for prey
    # In particular, I want 50% predator and 50% prey to be around 2 and 3.5 respectively
    # once scaled.
    # note: these numbers are specific to alpha=2/3, beta=4/3, delta=1, gamma=1
    scale = np.array([3.5/0.5, 2.0/0.5]) # state 0 is ignored
    scaled_pops = pops * scale

    # actual prey and predator population changes
    obs_deltas = (next_pops - pops).astype(float)

    # predicted prey and predator population changes
    target_prey_delta = alpha * scaled_pops[0] - beta * scaled_pops[0] * scaled_pops[1]
    target_predator_delta = delta * scaled_pops[0] * scaled_pops[1] - gamma * scaled_pops[1]
    scaled_target_deltas = np.array([target_prey_delta, target_predator_delta])
    target_deltas = scaled_target_deltas / scale

    if DEBUG:
        print('pops:', pops, 'next_pops:', next_pops)
        print('scaled_pops:', scaled_pops)
        print('obs_deltas:', obs_deltas)
        print('scaled_target_deltas:', scaled_target_deltas)
        print('target_deltas:', target_deltas)

    return obs_deltas, target_deltas


def get_lotka_deltas(obs, next_obs, alpha, beta, delta, gamma, num_states):
    pops = np.bincount(obs.ravel(), minlength=num_states)
    next_pops = np.bincount(next_obs.ravel(), minlength=num_states)

    # actual prey and predator population changes
    obs_deltas = (next_pops - pops).astype(float)[1:]

    # predicted prey and predator population changes
    target_prey_delta = alpha * pops[1] - beta * pops[1] * pops[2]
    target_predator_delta = delta * pops[1] * pops[2] - gamma * pops[2]
    target_deltas = np.array([target_prey_delta, target_predator_delta])

    if DEBUG:
        print('pops:', pops, 'next_pops:', next_pops)
        print('obs_deltas:', obs_deltas, 'target_deltas:', target_deltas)

    return obs_deltas, target_deltas
    

def make_mse_lotka_reward(alpha, beta, delta, gamma, num_states=3, rmse=False):
    def reward(obs, next_obs):
        '''
        obs: a CA board
        next_obs: the CA board at the next time step
        rmse: True if reward is -RMSE, False if reward is MSE
        returns: -1 * MSE between the population changes of predator and prey predicted by 
          the lotka-volterra model and the observed changes.

        Calculate mean sqaured error between the current deltas (changes in population)
        and target derivatives from the Lotke-Volterra equations.

        https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations

        Return -MSE, so the lower the error, the higher the reward.

        empty: index 0
        prey: index 1
        predator: index 2

        deltas: the change in population proportions from one time step to the next
        props: the population proportions, [0..1], for empty, prey, and predator
        alpha: prey birth rate
        beta: prey death rate from predators
        delta: predator birth rate from predation
        gamma: predator death rate

        '''
        obs_deltas, target_deltas = get_lotka_deltas(obs, next_obs, alpha, beta, delta, gamma, num_states)
        
        # mse between target and observed
        mse = np.mean((obs_deltas - target_deltas)**2)
        if rmse:
            rwd = -np.sqrt(mse)
        else:
            rwd = -mse
        
        
        if DEBUG:
            print('reward:', rwd)

        return rwd
    
    return reward


def make_cosine_lotka_reward(alpha, beta, delta, gamma, debug=False, num_states=3, noise=None, scaled=False):
    '''
    Cosine reward gives a reward for moving in the direction of the lotka-volterra derivatives.
    This sidesteps the problem of determining how far to move in the direction.
    '''
    def reward(obs, next_obs):
        if scaled:
            obs_deltas, target_deltas = get_scaled_lotka_deltas(
                obs, next_obs, alpha, beta, delta, gamma, num_states)
        else:
            obs_deltas, target_deltas = get_lotka_deltas(obs, next_obs, alpha, beta, delta, gamma, num_states)
        
        # add some jitter to avoid the case where obs_deltas is all zeros.
        if noise:
            obs_deltas += np.random.normal(scale=noise, size=obs_deltas.shape)
            target_deltas += np.random.normal(scale=noise, size=target_deltas.shape)
        
        # cosine between the delta vectors.  deltas should be in the same
        # direction as target_deltas
        rwd = (np.dot(obs_deltas, target_deltas) / 
                  (np.linalg.norm(obs_deltas) * np.linalg.norm(target_deltas)))
        if np.isnan(rwd):
            print('cosine_lotka_reward: nan reward')
            print(rwd, obs_deltas, target_deltas)
            rwd = 0 # what should I return?
            
        if DEBUG:
            print('cosine reward:', rwd)

        return rwd
    
    return reward


def random_state_dist(num_states=3):
    ps = np.random.random(num_states)
    ps = ps / np.sum(ps)
    return ps


def init_board(length, ps):
    '''
    length: the width and height of the board
    ps: state probability distribution. Each cell state is sampled from this distribution.
    return: board is a 2d array containing the state of every cell
    '''
    return np.random.choice(len(ps), size=(length, length), p=ps)


class LvcaEnv:
    '''
    Lotka-Voltera Cellular Automata Environment
    '''
    def __init__(self, length, alpha=2/3, beta=4/3, delta=1, gamma=1, reward_type='mse',
                 episode_len=None, cosine_noise=None):
        '''
        length: board is a length x length matrix of cells
        alpha: prey birth rate
        beta: prey death rate from predation
        delta: predator birth rate from predation
        gamma: predator death rate
        reward_type: 'mse' or 'cosine'. The penalty for differing from the lotka-volterra 
        episode_len: if not None, env.step() will return done=True after episode_len steps.
          otherwise env will be continuous.
        cosine_noise: for reward type 'cosine', this is the std dev of normal noise added to
         observed population changes to avoid NaN rewards when the population does not change.
         If None, no noise is added. Suggested noise is 0.1?
        '''
        self.length = length
        self.num_states = 3
        self.episode_len = episode_len
        self.reset() # initialize the board
        
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        print('Fixed point of Lotka-Volterra equations')
        print('Predator alpha/beta = ', alpha/beta)
        print('Prey:   gamma/delta =', gamma/delta)
        print(f'lotka-voltera params: alpha {alpha}, beta {beta}, delta {delta}, gamma {gamma}')
    
        if reward_type == 'mse':
            self.reward_func = make_mse_lotka_reward(alpha, beta, delta, gamma)
        elif reward_type == 'rmse':
            self.reward_func = make_mse_lotka_reward(alpha, beta, delta, gamma, rmse=True)
        elif reward_type == 'scaled_rmse':
            self.reward_func = make_scaled_rmse_lotka_reward(alpha, beta, delta, gamma)
        elif reward_type == 'cosine':
            self.reward_func = make_cosine_lotka_reward(alpha, beta, delta, gamma, noise=cosine_noise)
        elif reward_type == 'scaled_cosine':
            self.reward_func = make_cosine_lotka_reward(alpha, beta, delta, gamma, noise=cosine_noise, scaled=True)
        else:
            raise Exception('Unknown reward type', reward_type)
            
    def step(self, action):
        '''
        action: next board, shape (length, length)
        '''
        self.t += 1 # increment episode time steps
        reward = self.reward_func(self.board, action)
        self.board = action
        done = False if self.episode_len is None else (self.t >= self.episode_len)
        info = {}
        return self.board, reward, done, info
    
    def reset(self):
        self.t = 0
        self.board = init_board(self.length, ps=random_state_dist(self.num_states))
        return self.board
    
    def render(self, mode='human'):
        if mode == 'human':
            print(self.board)
            
        return self.board
                

def main():
    global DEBUG
    DEBUG = True
    num_eps = 10
    env = LvcaEnv(4, episode_len=10, reward_type='rmse', cosine_noise=0.001)
    for i in range(num_eps):
        board = env.reset()
        env.render()
        rewards = []
        while True:
            board = board.copy()
            board[np.random.choice(board.shape[0]), np.random.choice(board.shape[1])] = np.random.choice(3)
            board, reward, done, _ = env.step(board)
            env.render()
            rewards.append(reward)
            if done:
                break
                
        print('episode rewards:', rewards)
        print('mean:', np.mean(rewards))
        
    

if __name__ == '__main__':
    main()
