'''
Deep Q-Learning, MountainCar-v0 and CartPole-v0


DONE
====

Visualize Q-values, like in Example 10.1 from the Sutton and Barto book.

  
TODO
====


RESULTS
=======

model01: elu, num_epoch = 1000, num_episodes = 20, train_epochs = 2, learning_rate=1e-4, epsilon = 0.1, gamma = 1.0
looks very linear. positive q-values (when ALL rewards are negative) is a bad sign, though if model is trapped in small area of the state space and learning a linear q-value function, those positive q-values could be for areas of the state space never explored. learning_rate too high? epsilon too low? gamma too high?
model02: oops.
model03: tanh activation, 10 episodes, 1 training epoch, gamma=0.95. q-valeus are negative, so that is good. plotted q-values look like a plane with a positive slope for position (greater position = greater q-value) and no slope for velocity. Looks like the deep model is devolving into a linear model. Why?
model04: Use random tile coding with linear model in attempt to approximate the Sutton and Barto approach
model05: Use non-random overlapping tile coding in a better attempt to approximate S&B, learning_rate=1e-2, epsilon = 0.2. It wins!!! learning rate? epsilon? tiling?
'''


from pathlib import Path
import argparse
import datetime
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

import gym


def make_tiling_2d(x_min, x_max, y_min, y_max, num_tilings, freq=None, offsets=(1,3)):
    '''
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

    

def make_random_tiling_2d(x_min, x_max, y_min, y_max, freq, tiles):
    '''
    tiles: the number of random tiles
    freq: the coarseness of the tiling. The tile dimensions are 1/freq of the range of each dimension.
    
    Create a set of tiles randomly distributed across the 2 dimensions. The tile centers are uniformly
    distributed between the min and max bounds given.  The tile bounds can extend beyond the min and max
    bounds given.
    '''
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_tile_size = x_range / freq
    y_tile_size = y_range / freq
    x_centers = tf.random_uniform(shape=(tiles,)) * x_range + x_min
    y_centers = tf.random_uniform(shape=(tiles,)) * y_range + y_min
    tile_x_min = x_centers - (x_tile_size / 2)
    tile_x_max = x_centers + (x_tile_size / 2)
    tile_y_min = y_centers - (y_tile_size / 2)
    tile_y_max = y_centers + (y_tile_size / 2)
    return (tile_x_min, tile_x_max, tile_y_min, tile_y_max)


def make_mountain_car_tiling(x_min=-1.2, x_max=0.6, y_min=-0.07, y_max=0.07, freq=8, tiles=512, kind='random'):
    if kind == 'random':
        return make_random_tiling_2d(x_min, x_max, y_min, y_max, freq, tiles)
    elif kind == 'offset':
        return make_tiling_2d(x_min, x_max, y_min, y_max, num_tilings=freq)
    else:
        raise Exception('Unknown tiling kind', kind)


def tile_encode(obs, tiling):
    '''
    obs: input tensor. shape (batch_size, 2). observations are 2 dimensional
    Sutton and Barto used an 8 8x8 tiling with (1, 3) asymmetrical offsets 
    with Sarsa to learn the Mountain Car task.
    See Section 9.5.4 Tile Coding and Example 10.1 Mountain Car Task in the book.
    
    returns: (batch_size, num_tiles), where, for each point in the batch, tiles 
      that cover the point are on (1.0) and the other tiles are off (0.0).
    '''
    # shapes (num_tiles,)
    tile_x_min, tile_x_max, tile_y_min, tile_y_max = tiling
    batch_size = obs.shape[0]
    obs_x = tf.cast(tf.expand_dims(obs[:,0], -1), dtype=tf.float32) # shape (batch_size, 1), for broadcasting
    obs_y = tf.cast(tf.expand_dims(obs[:,1], -1), dtype=tf.float32) # shape (batch_size, 1), for broadcasting
    
#     print('obs_x', obs_x)
#     print('obs_y', obs_y)
    
    # Expand tilings to (batch_size, num_tiles)
    # Broadcast observation comparisons across tilings.
    # observation is in the x bounds of a tile if tile_x_min <= obs_x < tile_x_max
    big_tile_x_min = tf.cast(tf.tile(tf.expand_dims(tile_x_min, axis=0), multiples=(batch_size, 1)), dtype=tf.float32)
    big_tile_x_max = tf.cast(tf.tile(tf.expand_dims(tile_x_max, axis=0), multiples=(batch_size, 1)), dtype=tf.float32)
    
#     print('big_tile_x_min:', big_tile_x_min)
#     print('big_tile_x_max:', big_tile_x_max)
    
    obs_in_x = tf.logical_and(
        tf.greater_equal(obs_x, big_tile_x_min),
        tf.less(obs_x, big_tile_x_max)) # (batch_size, num_tiles)

    # observation is in the y bounds of a tile if tile_y_min <= obs_y < tile_y_max
    big_tile_y_min = tf.cast(tf.tile(tf.expand_dims(tile_y_min, axis=0), multiples=(batch_size, 1)), dtype=tf.float32)
    big_tile_y_max = tf.cast(tf.tile(tf.expand_dims(tile_y_max, axis=0), multiples=(batch_size, 1)), dtype=tf.float32)
    
#     print('big_tile_y_min:', big_tile_y_min)
#     print('big_tile_y_max:', big_tile_y_max)
    
    obs_in_y = tf.logical_and(
        tf.greater_equal(obs_y, big_tile_y_min),
        tf.less(obs_y, big_tile_y_max))
    
    obs_in_tile = tf.logical_and(obs_in_x, obs_in_y)
    return tf.cast(obs_in_tile, dtype=tf.float32)
    
    
class AgentModel(tf.keras.Model):
    '''
    Input: state
    Output: Q(s,a) for each action
    '''
    def __init__(self, num_actions, n_h=128, activation='elu', n_l=1, tiling=None):
        '''
        num_states: 
        '''
        super().__init__()
        self.num_actions = num_actions
        self.activation = activation
        self.n_l = n_l
        self.n_h = n_h
        self.tiling = tiling
        if not self.tiling:
            self.dense1 = tf.keras.layers.Dense(n_h, activation=activation)
        self.dense2 = tf.keras.layers.Dense(num_actions) # Q(s,a)

    def call(self, x):
        if not self.tiling:
            x = self.dense1(x)
            tf.contrib.summary.histogram('dense1_act', x)
            tf.contrib.summary.histogram('dense1_weights', self.dense1.weights[0])
            tf.contrib.summary.histogram('dense1_bias', self.dense1.weights[1])
        else:
            x = tile_encode(x, self.tiling)
            
        x = self.dense2(x)
        tf.contrib.summary.histogram('dense2_act', x)
        tf.contrib.summary.histogram('dense2_weights', self.dense2.weights[0])
        tf.contrib.summary.histogram('dense2_bias', self.dense2.weights[1])
        return x


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


def value_loss(qvalues, actions, rewards, next_qvalues, gamma=1.0):
    '''
    Loss is MSE between Temporal Difference target and qvalues of selected actions, 
    that is, between (r + gamma * max_a' q(s',a')) and q(s,a)
    
    qvalues: batch of action values for state, shape (batch_size, num_actions)
    actions: batch of sampled actions, shape (batch_size,)
    rewards: batch of rewards, shape (batch_size,)
    qvalues: batch of action values for next state, shape (batch_size, num_actions)  
    '''
    targets = rewards + gamma * tf.reduce_max(next_qvalues, axis=-1)
    actuals = tf.reduce_sum(tf.one_hot(actions, depth=qvalues.shape[-1]) * qvalues, axis=-1)
    loss = tf.reduce_mean((targets - actuals)**2) # openai deepq uses huber loss instead of MSE
    return loss    


def sample_action(qvalues, epsilon=0.0):
    '''
    qvals: shape (batch_size, num_actions)
    '''
    ps = tf.random_uniform(shape=qvalues.shape[:-1]) # (batch_size,)
    random_actions = tf.squeeze(tf.multinomial(tf.ones_like(qvalues), num_samples=1), axis=-1) # (batch_size,)
    max_actions = tf.argmax(qvalues, axis=-1)
    selected_actions = tf.where(tf.greater_equal(ps, epsilon), max_actions, random_actions)
    return selected_actions


def visualize_model(model):
    '''
    Visualize qvalues for mountain car model, to compare to plots in Sutton & Barto
    '''
    positions = np.linspace(-1.2, 0.6, 21)
    speeds = np.linspace(-0.07, 0.07, 21)
    results = []
    observations = []
    for position in positions:
        for speed in speeds:
            observations.append((position, speed))
    obs_batch = np.array(observations, dtype=np.float32)
    qvalues = model.predict(obs_batch)
    print('some qvalues:', qvalues[:10])
    max_qvalues = tf.reduce_max(qvalues, axis=-1).numpy()
    x, y = np.array(list(zip(*observations)))
    z = max_qvalues
            
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
    plt.show()
     

def run(args):


    exp_id = 'exp20181220'
    model_id = 'model05'
    data_dir = Path('/Users/tfd/data/2018/learning_reinforcement_learning') / exp_id
    batch_size = 32
    num_epoch = 1000 # number of buffers filled and then trained on
    num_episodes = 10 # number of episodes per buffer
    train_epochs = 1 # number of times to train model on each buffer
    learning_rate=1e-2
    epsilon = 0.2 # exploration rate
    gamma = 0.95 # discount rate
    checkpoint_dir = data_dir / f'{model_id}_checkpoints'
    checkpoint_prefix = checkpoint_dir / 'ckpt'
    # different tensorboard log dir for each run
    log_dir = data_dir / f'log/{model_id}/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    activation = 'tanh' #'elu'
    tiling_kind = 'offset' #'random' # None
    
    # make environment
    env = gym.make('MountainCar-v0')
#     env = gym.make('CartPole-v0')
    num_actions = env.action_space.n
    
    if tiling_kind == 'random':
        tiling = make_mountain_car_tiling(kind='random')
    elif tiling_kind == 'offset':
        tiling = make_mountain_car_tiling(kind='offset')
    else:
        tiling = None
        
    model = AgentModel(num_actions, activation=activation, tiling=tiling)    
#     optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)    
    global_step = tf.train.get_or_create_global_step()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, global_step=global_step)

    if args.restore_latest or args.restore_model:
        model_file = args.restore_model if args.restore_model else tf.train.latest_checkpoint(checkpoint_dir)
        print('restoring', model_file)
        checkpoint.restore(model_file)
    
    if args.visualize:
        visualize_model(model)
    elif args.model_summary:
        num_epoch = 1
        num_episodes = 1
        # dummy training to initialize model (without saving a checkpoint)
        train(model, env, optimizer, global_step, 
              num_epoch=1, num_episodes=1, batch_size=batch_size, train_epochs=1)
        model.summary()
    elif args.train:
        # tensorboard setup
        summary_writer = tf.contrib.summary.create_file_writer(str(log_dir), flush_millis=5000)

        with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            train(model, env, optimizer, global_step, 
                  num_epoch, num_episodes, batch_size, train_epochs,
                  checkpoint, checkpoint_prefix, epsilon=epsilon, gamma=gamma)
    elif args.test:
        test(model, env, checkpoint, checkpoint_dir)
    else:
        print('doing nothing.')

        
def train(model, env, optimizer, global_step, num_epoch, num_episodes, batch_size, train_epochs,
          checkpoint=None, checkpoint_prefix=None, epsilon=0.1, gamma=1.0):
        
    print('start training')
    print('global_step:', global_step.numpy())
    # lets do this thing!
    for i_epoch in range(num_epoch):
        print('epoch', i_epoch)

        # fill replay buffer with experiences
        buffer = []
        rewards = []
        episode_reward_totals = []
        for i_epi in range(num_episodes):
            episode_rewards = []
            episode_obs = []
            episode_actions = []
            obs = env.reset()
            done = False
            while not done:
                obs_batch = obs[None].astype(np.float32) # (batch_size=1, obs_dims)
#                 print('obs_batch', obs_batch)
                qvalues = model.predict(obs_batch) # (batch_size, num_actions)
#                 print('qvalues:', qvalues)
                action = sample_action(qvalues, epsilon)[0].numpy() # batch size of 1
#                 print('action:', action)
                next_obs, reward, done, info = env.step(action)
                episode_rewards.append(reward)
                rewards.append(reward)
                episode_obs.append(obs)
                episode_actions.append(action)
                buffer.append((obs, action, reward, next_obs))
                obs = next_obs
#                 env.render()
                if done:
                    episode_reward_totals.append(np.sum(episode_rewards))
                    # monte carlo returns with no discounting

        print('last obs:', obs)

        mean_episode_reward = np.mean(episode_reward_totals)
        print('mean_episode_reward:', mean_episode_reward)
        tf.contrib.summary.scalar('mean_episode_reward', mean_episode_reward)

        # make dataset from experience buffer
        gen = lambda: (exp for exp in buffer)
        ds = (tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32, tf.float32, tf.float32))
              .shuffle(len(buffer))
              .batch(batch_size, drop_remainder=True))

        print('training model...', end='')
        losses = []
        for i_te in range(train_epochs):
            print(f'{i_te}...', end='')
            for i_batch, batch in enumerate(ds):
                global_step.assign_add(1) # increment global step
                observations, actions, rewards, next_obervations = batch
                
                with tf.GradientTape() as tape:
                    qvalues = model(observations)
                    next_qvalues = model(next_obervations)
                    loss = value_loss(qvalues, actions, rewards, next_qvalues, gamma=gamma)
                    tf.contrib.summary.scalar('loss', loss)
                    losses.append(loss)
                    tf.contrib.summary.scalar('batch_mean_reward', tf.reduce_mean(rewards))

                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables),
                                          global_step=global_step)

        print('done')
        print('mean training loss:', np.mean(losses))

        # save (checkpoint) the model every 10 epochs
        if (i_epoch + 1) % 10 == 0:
            if checkpoint:
                cp_path = checkpoint.save(file_prefix=checkpoint_prefix)
                print('saved checkpoint to', cp_path)



def test(model, env, checkpoint, checkpoint_dir, num_episodes=100):
    
    # play game
    for i_epi in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            obs_batch = obs[None].astype(np.float32)
            qvalues = model.predict(obs_batch)
            action = sample_action(qvalues, epsilon=0.0)[0].numpy() # greedy, not epsilon-greedy
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            env.render()


def test_tiling():
    '''
    Don't use code blindly; debug it first.
    '''
    tiling = make_mountain_car_tiling(freq=8, tiles=10)
    x_mins, x_maxes, y_mins, y_maxes = tiling
    
    # Print tiles and make sure values look sane
    # values should be within the x and y ranges (plus a little extra for the tile size)
    # the difference between min and max values of tiles should equal the size of the tile
    # which is 1/2 the x range (or y range) for freq=2
    print('tiling:', tiling) 

    random_obs_x = np.random.random(10) * 1.8 - 1.2
    random_obs_y = np.random.random(10) * 0.14 - 0.07
    obs = np.array([(0,0), (-1.2, -0.07), (0.6, 0.07)] + list(zip(random_obs_x, random_obs_y)))
    encodings = tile_encode(obs, tiling)
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
    for ob in obs:
        plt.plot(ob[0], ob[1], 'bo')
        
    plt.show()
    


def main():
    print('tf version:', tf.VERSION)
    print('tf keras version:', tf.keras.__version__)
#     print(dir(tf))
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-summary', default=False, action='store_true')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
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
