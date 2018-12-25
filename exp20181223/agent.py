'''
Linear Approximation, Tiling, and SARSA or Q-learning or Monte Carlo on MountainCar-v0

Like Example 10.1 from the Sutton and Barto book.

DONE
====


  
TODO
====


RESULTS
=======

model02: scores around -122 reward per episode (averaged over 100 episodes).
num_episodes = 500
    learning_rate=1e-2
    epsilon = 0.05 # exploration rate
    gamma = 0.95 # discount rate

model02: scores around -136. gamma=0.95 did not work
    num_episodes = 1000
    learning_rate=1e-2
    epsilon = 0.05 # exploration rate
    gamma = 0.99 # discount rate
    kind = 'montecarlo'

model03: Wins! 100 episode average -108.93
    num_episodes = 2000
    learning_rate=1e-2
    epsilon = 0.05 # exploration rate 0.1 works too.
    gamma = 1.0 # 0.99 # discount rate
    kind = 'montecarlo'

model04: Wins! 100 episode avg -104.54
    num_episodes ~ 8000-10000
    learning_rate=1e-3
    epsilon = 0.1 # exploration rate
    gamma = 1.0 # 0.99 # discount rate
    kind = 'sarsa' # 'montecarlo'

model05: Wins! 100 episode avg -103.65
    num_episodes = 1000
    learning_rate=2e-2
    epsilon = 0.1 # exploration rate
    gamma = 1.0 # 0.99 # discount rate
    kind = 'montecarlo' # montecarlo or sarsa

model06: Wins! 100 episode average -105.31
    num_episodes = 1000
    learning_rate=2e-2
    epsilon = 0.1 # exploration rate
    gamma = 1.0 # 0.99 # discount rate
    kind = 'qlearning' # 'montecarlo' # montecarlo or sarsa


'''


from pathlib import Path
import pickle
import argparse
import datetime
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

# import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tf.enable_eager_execution()

import gym


def make_tiling_2d(x_min, x_max, y_min, y_max, num_tilings, freq=None, offsets=(1,3)):
    '''
    This is approximately like the description of the tiling used in Sutton and Barto.
    
    freq: the coarseness of the tiling. The tile dimensions are 1/freq of the range of each dimension.
     Default, if None, is to use freq == num_tilings
    num_tilings: number of times the x,y range is tiled
    offsets: the distances tilings are offset from each other.
    
    returns: a tuple of lists of the coordinates of each tile. There are num_tilings * freq**2 tiles
    '''
    freq = num_tilings if freq is None else freq
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_tile_size = x_range / freq
    y_tile_size = y_range / freq
    x_offset_len = x_tile_size / freq # "fundamental unit" is range / (freq**2)
    y_offset_len = y_tile_size / freq
    max_units = freq * freq # the length of the range in fundamental units
    
    # center locations in fundamental units
    x_centers = []
    y_centers = []
    for i_t in range(num_tilings):
        for i_x in range(freq):
            for i_y in range(freq):
                x_centers.append((i_x * freq + (i_t * offsets[0])) % max_units) # location in fundamental units
                y_centers.append((i_y * freq + (i_t * offsets[1])) % max_units)
    
    x_centers = np.array(x_centers) * x_offset_len + x_min
    y_centers = np.array(y_centers) * y_offset_len + y_min
    tile_x_min = x_centers - (x_tile_size / 2)
    tile_x_max = x_centers + (x_tile_size / 2)
    tile_y_min = y_centers - (y_tile_size / 2)
    tile_y_max = y_centers + (y_tile_size / 2)
    return (tile_x_min, tile_x_max, tile_y_min, tile_y_max)

    
def make_mountain_car_tiling(x_min=-1.2, x_max=0.6, y_min=-0.07, y_max=0.07, freq=8):
    '''
    freq: the number adjacent tiles that cover the whole range in one dimension. The
    number of tiles in 2 dimensions is freq**2.  Also freq offset tilings are generated
    leading to a total of freq**3 tiles that cover the range a total of freq times.
    '''
    return make_tiling_2d(x_min, x_max, y_min, y_max, num_tilings=freq)


def tile_encode(obs, tiling):
    '''
    obs: input obsevation, shape (2,). observations are 2 dimensional
    Sutton and Barto used an 8 8x8 tiling with (1, 3) asymmetrical offsets 
    with Sarsa to learn the Mountain Car task.
    See Section 9.5.4 Tile Coding and Example 10.1 Mountain Car Task in the book.
    
    returns: shape (num_tiles,), where tiles that cover the point are 
      on (1.0) and the other tiles are off (0.0).
    '''
    # shapes (num_tiles,)
    tile_x_min, tile_x_max, tile_y_min, tile_y_max = tiling
    obs_x = obs[0]
    obs_y = obs[1]
    
#     print('obs_x', obs_x)
#     print('obs_y', obs_y)
    
    obs_in_x = (obs_x >= tile_x_min) & (obs_x < tile_x_max)
    obs_in_y = (obs_y >= tile_y_min) & (obs_y < tile_y_max)
    obs_in_tile = (obs_in_x & obs_in_y)
    return obs_in_tile.astype(np.float32)
    

class IdentityEmbedding():
    def __init__(self, num_features):
        self.num_features = num_features
        
    def __call__(self, x):
        return x

    
class OneHotEmbedding():
    def __init__(self, num_features):
        self.num_features = num_features
        self.one_hot = np.eye(num_features)
        
    def __call__(self, x):
        '''
        x: a state index
        return: one hot encoded vector
        '''
        return self.one_hot[x]

    
class TilingEmbedding2d():
    
    def __init__(self, tiling):
        '''
        tiling: a list of 2d tiles, as a tuple of lists of tile x_min, x_max, y_min and y_max.
        '''
        self.tiling = tiling
        self.num_features = len(tiling[0])
        
    def __call__(self, x):
        '''
        x: shape (2,), a 2d input state
        return: shape (num_tiles,), the tile coding of x.
        '''
        return tile_encode(x, self.tiling)
    

class LinearApproximationModel():
    def __init__(self, num_actions, embedding):
        '''
        '''
        super().__init__()
        self.num_actions = num_actions
        self.embedding = embedding
        self.num_features = self.embedding.num_features + 1 # include bias
        self.weights = np.zeros((self.num_actions, self.num_features))

    def __call__(self, x):
        '''
        x: shape (obs_dims,)
        returns: (action-values, state_embeding)
        '''
        features = np.append(self.embedding(x), 1.0) # include bias
        qvalues = np.dot(self.weights, features) # (num_actions,)
        return qvalues, features #
        
    def get_action(self, qvalues, epsilon=0.0):
        '''
        qvalues: shape (num_actions,)
        '''
        if np.random.random() > epsilon:
            return np.argmax(qvalues)
        else:
            return np.random.choice(self.num_actions)

    
class TilingLinearSarsaModel():
    def __init__(self, num_actions, tiling, learning_rate, gamma):
        '''
        '''
        super().__init__()
        self.num_actions = num_actions
        self.tiling = tiling
        self.num_features = len(tiling[0]) + 1 # include bias
        self.weights = np.zeros((self.num_actions, self.num_features))
        self.learning_rate = learning_rate
        self.gamma = gamma

    def __call__(self, x, n=10):
        '''
        x: shape (obs_dims,)
        '''
        features = np.append(tile_encode(x, self.tiling), 1.0) # include bias
        qvalues = np.dot(self.weights, features) # (num_actions,)
        return qvalues, features #
    
    def get_action(self, qvalues, epsilon=0.0):
        '''
        qvalues: shape (num_actions,)
        '''
        if np.random.random() > epsilon:
            return np.argmax(qvalues)
        else:
            return np.random.choice(self.num_actions)

    def fit(self, features, qvalues, action, reward, done, next_qvalues, next_action):
        '''
        Update model weights using w = w + alpha * (r + gamma * q(s,a) - q(s',a')) * x(s)
        
        features: (num_features,) embedding of current observation
        '''
        target = reward + (self.gamma * next_qvalues[next_action] if not done else 0)
        weight_update = self.learning_rate * (target - qvalues[action]) * features
#         print('weights[action]:', self.weights[action])
#         print('weight_update:', weight_update)
#         print('target:', target)
#         print('qvalues[action]:', qvalues[action])
#         print('self.gamma * qvalues[action]:', self.gamma * qvalues[action])
#         print('features.shape:', features.shape)
#         print('self.weights.shape:', self.weights.shape)
        self.weights[action] += weight_update
    

def visualize_model(model):
    '''
    Visualize qvalues for mountain car model, to compare to plots in Sutton & Barto.
    Points are colored according to the action selected by the greedy policy.
    '''
    positions = np.linspace(-1.2, 0.6, 51)
    speeds = np.linspace(-0.07, 0.07, 51)
    results = []
    observations = []
    for position in positions:
        for speed in speeds:
            observations.append((position, speed))
    qvalues = np.array([model(obs)[0] for obs in observations])

    print('some qvalues:', qvalues[:10])
    max_qvalues = np.amax(qvalues, axis=-1)
    max_actions = np.argmax(qvalues, axis=-1)
    x, y = np.array(list(zip(*observations)))
    z = max_qvalues
            
    from mpl_toolkits.mplot3d import Axes3D
    # actions: 0, 1, 2 are left, null, right, respectively
    colors = ['#dd3333', '#333333', '#33dd33']
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', colors)
    norm = matplotlib.colors.Normalize(vmin=0,vmax=2)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
#     ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    ax.scatter(x, y, z, c=max_actions, cmap=cmap, norm=norm)
    plt.title('green=right,grey=null,red=left')
    plt.show()
     

class MyCheckpoint():
    '''
    saving and loading of models.
    '''
    def save(self, model, checkpoint_prefix):
        path = Path(str(checkpoint_prefix) + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.pkl')
        path.parent.mkdir(exist_ok=True)
        with open(path, 'wb') as fh:
            pickle.dump(model, fh)
            
        return path
    
    def latest_path(self, checkpoint_dir):
        d = Path(checkpoint_dir)
        return list(sorted(d.glob('*.pkl')))[-1]
    
    def restore(self, checkpoint_file):
        with open(checkpoint_file, 'rb') as fh:
            return pickle.load(fh)


def discounted_returns(rewards, gamma=1.0, normalize=False):
    '''
    Compute the discounted returns (possibly normalized) for 
    the rewards of an episode
    '''
    disc_returns = np.zeros(len(rewards))
    # gamma = 1.0
    r_t = 0
    for i in range(len(rewards) - 1, -1, -1):
        r_t = rewards[i] + gamma * r_t
        disc_returns[i] = r_t
    
    if normalize:
        disc_returns = (disc_returns - np.mean(disc_returns)) / np.std(disc_returns)
        
    return disc_returns


def run(args):

    exp_id = 'exp20181223'
    model_id = 'model06'
    data_dir = Path('/Users/tfd/data/2018/learning_reinforcement_learning') / exp_id
    num_episodes = 10000
    learning_rate=1e-3
    epsilon = 0.01 # exploration rate
    gamma = 1.0 # 0.99 # discount rate
    kind = 'montecarlo' # montecarlo or sarsa or qlearning
    checkpoint_dir = data_dir / f'{model_id}_checkpoints'
    checkpoint_prefix = checkpoint_dir / 'ckpt'
    # different tensorboard log dir for each run
    log_dir = data_dir / f'log/{model_id}/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    checkpoint = MyCheckpoint()
    
    # make environment
#     env = gym.make('MountainCar-v0')
#     env = gym.make('CartPole-v0')
#     num_observation_dims = env.observation_space.shape[0]
#     env = gym.make('Taxi-v2')
    env = gym.make('FrozenLake-v0')
    num_observation_dims = env.observation_space.n
    
    num_actions = env.action_space.n
#     embedding = IdentityEmbedding(num_observation_dims)
    embedding = OneHotEmbedding(num_observation_dims) # for doing table-based RL with a linear model
#     embedding = TilingEmbedding2d(make_mountain_car_tiling(freq=8)) # 8 is the number of tilings from Sutton & Barto
    model = LinearApproximationModel(num_actions, embedding)

    if args.restore_latest or args.restore_model:
        model_file = args.restore_model if args.restore_model else checkpoint.latest_path(checkpoint_dir)
        print('restoring', model_file)
        model = checkpoint.restore(model_file)
    
    if args.visualize:
        visualize_model(model)
    elif args.model_summary:
        num_epoch = 1
        num_episodes = 1
        # dummy training to initialize model (without saving a checkpoint)
        train(model, env, num_episodes=1)
        model.summary() # print summary and graph of model
    elif args.train:
        # use tensorboard to log training progress
#         summary_writer = tf.contrib.summary.create_file_writer(str(log_dir), flush_millis=5000)
#         with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        train(model, env, num_episodes=num_episodes,
              checkpoint=checkpoint, checkpoint_prefix=checkpoint_prefix,
              epsilon=epsilon, gamma=gamma, learning_rate=learning_rate, kind='sarsa')
    elif args.test:
        test(model, env, render=args.render)
    else:
        print('doing nothing.')


def train(model, env, num_episodes, 
          checkpoint=None, checkpoint_prefix=None, 
          epsilon=0.1, gamma=0.95, learning_rate=1e-3,
          kind='sarsa'):
    '''
    SARSA training
    Model predicts q-values. Every step, update model using temporal difference error.
    On-policy.
    kind: sarsa, montecarlo
    '''
        
    print('start training')

    num_actions = env.action_space.n

    for i_epi in range(num_episodes):
        if i_epi % 1 == 0:
            print('i_epi', i_epi)
            
        done = False
        episode_features = []
        episode_qvalues = []
        episode_actions = []
        episode_rewards = []
        obs = env.reset() # shape (obs_dims,)
        qvalues, features = model(obs) # return features so they don't need to be recomputed in model.fit
        action = model.get_action(qvalues, epsilon=epsilon)
#         print('obs:', obs)
#         print('features:', features)
#         print('qvalues:', qvalues)
#         print('action:', action)

        while not done:

            next_obs, reward, done, info = env.step(action)
            episode_features.append(features)
            episode_qvalues.append(qvalues)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            next_qvalues, next_features = model(next_obs)
            next_action = model.get_action(next_qvalues, epsilon=epsilon)
                        
            if kind == 'sarsa' or 'qlearning':
                if kind == 'sarsa':
                    target = reward + (gamma * next_qvalues[next_action] if not done else 0)
                elif kind == 'qlearning':
                    target = reward + (gamma * np.amax(next_qvalues) if not done else 0)
                weight_update = learning_rate * (target - qvalues[action]) * features
                model.weights[action] += weight_update

            obs = next_obs
            qvalues = next_qvalues
            features = next_features
            action = next_action

        if kind == 'montecarlo':
            epi_returns = discounted_returns(episode_rewards, gamma=gamma)
            for features, qvalues, action, return_ in zip(episode_features, episode_qvalues, episode_actions, epi_returns):
                update = learning_rate * (return_ - qvalues[action]) * features
                model.weights[action] += weight_update
                

        episode_reward = np.sum(episode_rewards)
        print('episode_reward:', episode_reward)
            
#         print('last obs:', obs)


        # save (checkpoint) the model every n steps
        if (i_epi + 1) % 100 == 0:
            if checkpoint:
                cp_path = checkpoint.save(model, checkpoint_prefix)
                print('saved checkpoint to', cp_path)



def test(model, env, num_episodes=100, epsilon=0.0, render=False):
    '''
    Play game for num_episodes episodes. Use greedy policy (epsilon == 0).
    '''
    episode_rewards = []
    for i_epi in range(num_episodes):
        episode_reward = 0
        done = False
        obs = env.reset() # shape (obs_dims,)
        while not done:
            qvalues, features = model(obs) # return features so they don't need to be recomputed in model.fit
            action = model.get_action(qvalues, epsilon=epsilon)
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            obs = next_obs
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        
    print('num_episodes:', num_episodes)
    print('mean episode reward:', np.mean(episode_rewards))


def test_tiling():
    '''
    Don't use code blindly; debug it first.
    '''
    tiling = make_mountain_car_tiling(freq=2)
    x_mins, x_maxes, y_mins, y_maxes = tiling
    
    # Print tiles and make sure values look sane
    # values should be within the x and y ranges (plus a little extra for the tile size)
    # the difference between min and max values of tiles should equal the size of the tile
    # which is 1/2 the x range (or y range) for freq=2
    print('tiling:', tiling) 

    random_obs_x = np.random.random(10) * 1.8 - 1.2
    random_obs_y = np.random.random(10) * 0.14 - 0.07
    observations = np.array([(0,0), (-1.2, -0.07), (0.6, 0.07)] + list(zip(random_obs_x, random_obs_y)))
    encodings = np.array([tile_encode(obs, tiling) for obs in observations])
    print('encodings:', encodings)

    # https://matplotlib.org/gallery/statistics/errorbars_and_boxes.html#sphx-glr-gallery-statistics-errorbars-and-boxes-py
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Create list for all the error patches
    errorboxes = []

    # Loop over data points; create box from errors at each point
    for x_min, x_max, y_min, y_max in zip(*tiling):
        rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min)
        errorboxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor='r', alpha=0.1,
                         edgecolor=None)

    # Add the bounds of the mountain car input space
    rect = Rectangle((-1.2, -0.07), 1.8, 0.14, linewidth=1, edgecolor='b', facecolor='none')
    ax.add_patch(rect)

    # Add collection to axes
    ax.add_collection(pc)
    ax.set_xlim(left=-2.2, right=1.5)
    ax.set_ylim(top=0.15, bottom=-0.15)
    
    # add observations
    for ob in observations:
        plt.plot(ob[0], ob[1], 'bo')
        
    plt.show()
    


def main():
#     print('tf version:', tf.VERSION)
#     print('tf keras version:', tf.keras.__version__)
# #     print(dir(tf))
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-summary', default=False, action='store_true')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--no-render', dest='render', default=True, action='store_false')
    parser.add_argument('--restore-latest', default=False, action='store_true')
    parser.add_argument('--restore-model', metavar='PATH', help='Specify a model path to restore')
    parser.add_argument('--visualize', default=False, action='store_true')
    args = parser.parse_args()
#     print(args)
#     test_tiling()
#     return
    run(args)
        


if __name__ == '__main__':
    main()
