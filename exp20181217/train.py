'''
Policy Gradients, LVCA, Eager Tensorflow

Useful example of tensorflow code:
- https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/eager/python/examples/generative_examples/dcgan.ipynb

Why does the agent suck so bad?
- I'm overfitting the data
- The agent is learning to maximize the reward but the reward function is hard for a CA to learn
  - need a new reward function?
  
DONE
====

- fixed major bug, where instead of saving (observation, action, reward) to buffer,
  I was saving (next_observation, action, reward). The observation did not correspond
  to the action or reward!
- use baseline: (r - mean(r)/std(r)) or (r - mean(r)) or r - v(s)
- tensorboard metrics:
  - mean reward (buffering) over time
  - mean entropy (buffering) over time
  - mean loss (training) over time
  
TODO
====

- periodic padding of CA board: https://stackoverflow.com/questions/39088489/tensorflow-periodic-padding
- use return: this is a temporal problem. worth rewarding actions that lead to future good things.
- output model description: graph, number of params
- env/reward in tensorflow. for this game, env just gives reward, since action is next board.
  - reward
- memory
  - model input: memory + board board
  - model output: next memory + next board
- neighborhood: stack conv layers to create larger neighborhood. does that improve performance?
- sanity check: learn to immitate the stochastic predator-prey model (sppm). reward is -cross-entropy between agent action and sppm next board
- tensorboard metrics:
  - population trajectories over time?
- replay buffer: get right balance between gathering experiences and training model.
  - use old experiences, like DQN or SAC?
- hyperparam tuning

RESULTS
=======

model01, cosine error, board length 8, model01, strategy: looks ok? wonder what the trajectories look like?
model02, mse error, length 6, strategy: kill the sheep then alternate empty and predators every round
model03, rmse error, length 8, strat: kill most of the sheep, split board between empty and predator square that alternate each round
model04, scaled rmse, length 8, strat: kill most of the sheep, alternating 50/50 empty/predator squares.
model05, scaled cosine, length 8
model06, scaled cosine, length 8, fixed obs-action mismatch in buffer, centered rewards, 8-step episodes
model07, cosine, length 8, centered rewards, 8-step episodes. strat: mostly pred, less empty, few prey.
model08, cosine, bigger neighborhood (kernel size 5, 5-hood). very quickly learns all 2's.
model08, scaled rmse, bigger neighborhood. 
model11, cosine, 5-hood, periodic boundaries
'''


from pathlib import Path
import argparse
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

import lvcaenv




def make_movie(boards, movie_path):
    '''
    Save a movie of sequential boards.
    boards: list of boards
    '''
    # https://medium.com/@Vincentropy/a-practical-intro-to-colors-in-python-496737f23568
    # create color maps and normalization
    # colors: empty, prey, predator -> green, white, black
#     colors = ['#61dd4e', '#eeeeee', '#111111']
    colors = ['#dddddd', '#3333dd', '#dd3333']
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', colors)
    norm = matplotlib.colors.Normalize(vmin=0,vmax=2)

    # make a movie figure
    movie_writer = animation.writers['ffmpeg'](fps=4)
    fig = plt.figure()
    ax = fig.add_subplot(111) # create a single subplot
    board = boards[0] # peek at first board to create image object
    im = ax.imshow(board, cmap=cmap, norm=norm)#, vmin=0, vmax=1) # use vmin and vmax if board does not have full range of values

    with movie_writer.saving(fig, outfile=movie_path, dpi=100):
        for board in boards:
            im.set_data(board)
            movie_writer.grab_frame()
            
    plt.close(fig)


def plot_rewards_rule_deltas_and_populations(rewards=None, rule_deltas=None, 
                                             state_props_list=None, prop_trajectory_states=None,
                                             start_trim=0, prop_scatter=False, entropies=None,
                                             local_rewards=None, num_states=3):
    '''
    rewards: list of rewards
    entropies: list of entropies
    state_props_list: each element of list is the populations of the different states at a time step.
    start_trim: index to start plotting at.  Useful for ignoring the first few timesteps where
      error and other values fluctuate wildly.
    prop_trajectory_states: If not none, a pair of states (e.g. (1,2)) to plot the state trajectory for.
    '''
    if rewards:
        plt.plot(rewards[start_trim:]) # ignore first few so plot is on a better scale, easier to see reward trends
        plt.title('Rewards vs Time')
        plt.show()
        
    if local_rewards:
        plt.plot(local_rewards[start_trim:])
        plt.title('Mean Local Rewards vs Time')
        plt.show()
        
    if rule_deltas:
        plt.plot(rule_deltas[start_trim:])
        plt.title('Rule changes vs Time')
        plt.show()
        
    if entropies:
        print('entropies:', entropies)
        plt.plot(entropies[start_trim:])
        plt.title('Mean Rule Entropy vs Time')
        plt.xlabel('time step')
        plt.ylabel('mean rule entropy (bits)')
        plt.show()

    if state_props_list:
        # convert state_props_list from a list of ndarrays to an ndarray, 
        # where each row is state populations at a point in time
        # and each column is population of a state over time
        ar = np.concatenate(state_props_list).reshape((len(state_props_list), num_states))
        
        for i in range(num_states):
            plt.plot(ar[start_trim:, i], label=i) # plot the population of each state over time

        plt.title('State population (proportion) vs Time')
        plt.legend()
        plt.show()
        
        # predator-prey specific
        if prop_trajectory_states:
            state_1, state_2 = prop_trajectory_states
            # colors = tuple(np.array(range(start_trim, len(ar))) / (len(ar) - start_trim))
            # plt.scatter(ar[start_trim:, 1], ar[start_trim:, 2], c=colors)
            # plt.plot(ar[start_trim:, 1], ar[start_trim:, 2])
            
            # Color the trajectory by plotting each line segment separately as a different color.
            # a little slow ... but better than nothing.
            # Line color starts blue and fades to red at the end.
            c1 = np.linspace(0,1,(len(ar) - start_trim))**2
            c2 = np.zeros(len(ar) - start_trim)
            c3 = np.linspace(1,0,(len(ar) - start_trim))**2
            # print('len(T)', len(T))
            if not prop_scatter:
                for i in range(len(ar)-start_trim-1):
                    # print(ar[i:(i+1), 1])
                    # print('i', i)
                    j = start_trim + i
                    plt.plot(ar[j:(j+2), state_1], ar[j:(j+2), state_2], color = [c1[i], c2[i], c3[i]])
            else:
                plt.scatter(ar[start_trim:, state_1], ar[start_trim:, state_2], color=list(zip(c1,c2,c3)))
            plt.title('State Population Trajectory')
            plt.xlabel(f'Population of state {state_1}')
            plt.ylabel(f'Population of state {state_2}')
            plt.show()

#             norm_pop_1 = ar[start_trim:, state_1] / np.mean(ar[start_trim:, state_1])
#             norm_pop_2 = ar[start_trim:, state_2] / np.mean(ar[start_trim:, state_2])
#             norm_pop_ratio = norm_pop_1 / norm_pop_2
#             plt.plot(norm_pop_ratio, label='pop ratio') # plot the population of each state over time
#             plt.title('Population-Ratio vs Time')
#             plt.legend()
#             plt.show()

            norm_pop_1 = ar[start_trim:, state_1] / np.mean(ar[start_trim:, state_1])
            norm_pop_2 = ar[start_trim:, state_2] / np.mean(ar[start_trim:, state_2])
            plt.plot(norm_pop_1, label=f'pop {state_1}') # plot the population of each state over time
            plt.plot(norm_pop_2, label=f'pop {state_2}') # plot the population of each state over time
            plt.title('Normalized Population vs Time')
            plt.legend()
            plt.show()

#             plt.plot(ar[start_trim:, 1] - ar[start_trim:, 2], label='pop diff') # plot the population of each state over time
#             plt.title('Population-difference vs Time')
#             plt.legend()
#             plt.show()

            

    if rewards:
        print('mean rewards:', np.mean(rewards))
        
    if local_rewards:
        print('mean local_rewards:', np.mean(local_rewards))
        
    for i in range(num_states):
        print(f'mean population state {i}: {np.mean(ar[:, i])}')


class CAModel(tf.keras.Model):
    '''
    To be a cellular automata, this model only looks at neighborhood information once,
    via a Conv2d layer with a (3, 3) kernel
    '''
    def __init__(self, num_states, kernel_size=3, periodic=False):
        '''
        num_states: 
        '''
        super().__init__()
        self.num_states = num_states
        self.periodic = periodic
        if periodic:
            padding = 'valid'
            self.pad_size = (kernel_size - 1) // 2
        else:
            padding = 'same'
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=kernel_size, strides=(1, 1), padding=padding,
                                            activation='elu') # neighborhood
        self.conv2 = tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='elu')
        self.conv3 = tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='elu')
        self.conv4 = tf.keras.layers.Conv2D(num_states, (1, 1), strides=(1, 1), padding='same') # logits

    def call(self, x):
        if self.periodic:
            x = periodic_padding(x, padding=self.pad_size)
        x = tf.expand_dims(x, axis=-1)
        x = self.conv1(x)
        tf.contrib.summary.histogram('conv1_act', x)
        tf.contrib.summary.histogram('conv1_w', self.conv1.weights[0])
        tf.contrib.summary.histogram('conv1_b', self.conv1.weights[1])
        x = self.conv2(x)
        tf.contrib.summary.histogram('conv2_act', x)
        tf.contrib.summary.histogram('conv2_w', self.conv2.weights[0])
        tf.contrib.summary.histogram('conv2_b', self.conv2.weights[1])
        x = self.conv3(x)
        tf.contrib.summary.histogram('conv3_act', x)
        tf.contrib.summary.histogram('conv3_w', self.conv3.weights[0])
        tf.contrib.summary.histogram('conv3_b', self.conv3.weights[1])
        x = self.conv4(x)
        tf.contrib.summary.histogram('conv4_act', x)
        tf.contrib.summary.histogram('conv4_w', self.conv4.weights[0])
        tf.contrib.summary.histogram('conv4_b', self.conv4.weights[1])
        return x
   

def periodic_padding(x, padding=1):
    '''
    x: shape (batch_size, d1, d2)
    return x padded with periodic boundaries. i.e. torus or donut
    '''
    d1 = x.shape[1] # dimension 1: height
    d2 = x.shape[2] # dimension 2: width
    p = padding
    # assemble padded x from slices
    #            tl,tc,tr
    # padded_x = ml,mc,mr
    #            bl,bc,br
    top_left = x[:, -p:, -p:] # top left
    top_center = x[:, -p:, :] # top center
    top_right = x[:, -p:, :p] # top right
    middle_left = x[:, :, -p:] # middle left
    middle_center = x # middle center
    middle_right = x[:, :, :p] # middle right
    bottom_left = x[:, :p, -p:] # bottom left
    bottom_center = x[:, :p, :] # bottom center
    bottom_right = x[:, :p, :p] # bottom right
    top = tf.concat([top_left, top_center, top_right], axis=2)
    middle = tf.concat([middle_left, middle_center, middle_right], axis=2)
    bottom = tf.concat([bottom_left, bottom_center, bottom_right], axis=2)
    padded_x = tf.concat([top, middle, bottom], axis=1)
    return padded_x
    

def policy_gradient_loss(logits, actions, rewards):
    '''
    Loss is: -A * log(p), where A is the advantage. A simple form of advantage is normalized reward.
    A more sophisticated form is U_t - V(s_t), where U_t is monte-carlo return
    or Q(a_t,s_t), the action value.

    rewards: batch of rewards, shape (batch_size,)
    logits: batch of action logits, shape (batch_size, length, length, num_states)
    actions: batch of sampled boards, shape (batch_size, length, length)
    '''
    # log pi(a|s)
    cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=actions # one-hot encoding happens within sparse_softmax_cross_entropy_with_logits
    )
    losses = tf.reduce_sum(rewards * tf.reduce_sum(tf.reduce_sum(cross_entropies, axis=-1), axis=-1))
    return losses    


def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator


def policy_entropy(logits, mean=False):
    '''
    Compute the policy entropy, - \sum_a p(a) log p(a), for each policy in batch
    logits: shape (batch_size, ..., num_actions)
    returns: shape (batch_size,), each element is entropy of corresponding policy in batch
    '''
    # -np.sum(action_probs * np.log2(action_probs), axis=-1)

    pi = tf.nn.softmax(logits, axis=-1)
#     print('pi:', pi)
    entropies = tf.reduce_sum(-1 * pi * log2(pi), axis=-1)
    if mean:
        entropies = tf.reduce_mean(entropies)
        
    return entropies
    

def sample_board(logits):
    # Using tensorflow eager execution for fun and education
    # logits shape: (batch_size, length, length, num_classes)
    # return shape: (batch_size, length, length), a batch of sampled boards
    # https://stackoverflow.com/questions/39432164/sample-from-a-tensor-in-tensorflow-along-an-axis
    dims = list(tf.shape(logits))
    num_states = dims[-1] # last dimension
    logits = tf.reshape(logits, (-1, num_states))
    board = tf.multinomial(logits=logits, num_samples=1)
    board = tf.reshape(board, dims[:-1])
    return board
    

def play(model, env, episodes=1, movie_path=None):
    rewards = []
    populations = []
    entropies = []
    for i_eps in range(episodes):
        
        obs = env.reset()
        episode_observations = [obs]
        population = np.bincount(obs.ravel(), minlength=env.num_states) / (env.length ** 2)
        populations.append(population)
        
        actions = []
        max_timesteps = 10000
        for i in range(max_timesteps):
            env.render()
            # obs_batch is (1, length, length) shape array of observations
            obs_batch = obs[None].astype(np.float64) # prepend batch dimension to make batch of size 1
            # logits is (1, num_actions) shape array of action logits
            logits = model.predict(obs_batch)
            entropy = policy_entropy(logits, mean=True)
#             print('entropy:', entropy)
            entropies.append(entropy)
#             print('logits:', logits)
            action = sample_board(logits)[0].numpy() # batch size of 1
#             print('action:', action)
            next_obs, reward, done, info = env.step(action)
            print('reward:', reward)
            rewards.append(reward)
            actions.append(action)
            
            obs = next_obs
            episode_observations.append(obs)
            population = np.bincount(obs.ravel(), minlength=env.num_states) / (env.length ** 2)
            populations.append(population)
            print('population:', population)
            if done:
                break

        env.render()
        if movie_path:
            make_movie(episode_observations, movie_path)
            
        print('rewards:', rewards)
        # print('actions:', actions)
        
    return rewards, populations, entropies


def run(args):


    exp_id = 'exp20181217'
    model_id = 'model11'
    data_dir = Path('/Users/tfd/data/2018/learning_reinforcement_learning') / exp_id
    batch_size = 32
    num_episodes = 64
    episode_len = 16 # 128 # number of time steps of cellular automata
    buffer_size = num_episodes * episode_len
    shuffle_size = buffer_size # 
    buffer_epochs = 10 # number of times to collect a buffer of experiences
    train_epochs = 2 # number of times to train model on each buffer
    num_epoch = 100
    learning_rate=1e-5
    length = 8 # length of cellular automata board
    reward_type = 'cosine' #'scaled_rmse' # 'mse' #'cosine'
    cosine_noise = 0.001 # avoid divide by zero issues with cosine reward
    start_trim = 0 # use for trimming off first n datapoints when generating plots.
    checkpoint_dir = data_dir / f'{model_id}_checkpoints'
    checkpoint_prefix = checkpoint_dir / 'ckpt'
    log_dir = data_dir / f'log/{model_id}/1'
    num_states = 3
    kernel_size = 5 # 3 # size of neighborhood
    periodic = True # board has periodic boundaries
    
    # tensorboard setup
    global_step = tf.train.create_global_step()
    summary_writer = tf.contrib.summary.create_file_writer(str(log_dir), flush_millis=10000)
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        model = CAModel(num_states, kernel_size=kernel_size, periodic=periodic)
        
#     optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)    
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    
    if args.model_summary:
        length = 4
        episode_len = 1
        num_epoch = 1
        num_episodes = 1
        # make environment
        env = lvcaenv.LvcaEnv(length=length, episode_len=episode_len, 
                              reward_type=reward_type, cosine_noise=cosine_noise)
        # dummy training to initialize model (without saving a checkpoint)
        train(model, env, optimizer, num_epoch=1, num_episodes=1, shuffle_size=shuffle_size, batch_size=batch_size, train_epochs=1)
        model.summary()
        return # don't train dummy model?

    if args.train:
        # make environment
        env = lvcaenv.LvcaEnv(length=length, episode_len=episode_len, 
                              reward_type=reward_type, cosine_noise=cosine_noise)
        with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            train(model, env, optimizer, num_epoch, num_episodes, shuffle_size, batch_size, train_epochs,
                 checkpoint, checkpoint_prefix)

    if args.test:
        length = 32
        episode_len = 128
        # make environment
        env = lvcaenv.LvcaEnv(length=length, episode_len=episode_len, 
                              reward_type=reward_type, cosine_noise=cosine_noise)
        test(model, env, checkpoint, checkpoint_dir, start_trim=start_trim, movie_path=args.movie_path)
    
    
def train(model, env, optimizer, num_epoch, num_episodes, shuffle_size, batch_size, train_epochs,
          checkpoint=None, checkpoint_prefix=None, log_dir=None):
    
    global_step = tf.train.get_global_step()
    
    # lets do this thing!
    for i_epoch in range(num_epoch):
        print('epoch', i_epoch)

        # fill replay buffer with experiences
        buffer = []
        entropies = []
        rewards = []
        returns = []
        for i_epi in range(num_episodes):
            episode_rewards = []
            episode_obs = []
            episode_actions = []
            obs = env.reset()
            done = False
            while not done:
                # obs_batch is (1, length, length) shape array of observations
                obs_batch = obs[None].astype(np.float32) # prepend batch dimension to make batch of size 1
                # logits is (1, length, length, num_states) shape array of next state logits
                logits = model.predict(obs_batch)
                entropy = policy_entropy(logits, mean=True)
    #             print('entropy:', entropy)
                entropies.append(entropy)
                # print('logits:', logits)
                action = sample_board(logits)[0].numpy() # batch size of 1
                # print('action:', action)
                next_obs, reward, done, info = env.step(action)
                episode_rewards.append(reward)
                rewards.append(reward)
                episode_obs.append(obs)
                episode_actions.append(action)
                obs = next_obs
                if done:
                    # turn rewards into returns
                    episode_returns = np.cumsum(episode_rewards)
                    # normalize/baseline returns 
                    norm_epi_returns = (episode_returns - np.mean(episode_returns)) / np.std(episode_returns)
                    centered_epi_rewards = np.array(episode_rewards) - np.mean(episode_rewards)
                    buffer.extend(zip(episode_obs, episode_actions, centered_epi_rewards))
                    returns.extend(norm_epi_returns)


        print('last obs:', obs)
        mean_entropy = np.mean(entropies)
        print('mean entropy:', mean_entropy)
        tf.contrib.summary.scalar('mean_entropy', mean_entropy)

        mean_reward = np.mean(rewards)
        print('mean reward:', mean_reward)
        tf.contrib.summary.scalar('mean_reward', mean_reward)

        # make dataset from experience replay buffer
        gen = lambda: (exp for exp in buffer)
        ds = (tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32, tf.float32))
              .shuffle(shuffle_size)
              .batch(batch_size))

        print('training model...', end='')
        losses = []
        for i_te in range(train_epochs):
            print(f'{i_te}...', end='')
            for i_batch, batch in enumerate(ds):
                global_step.assign_add(1) # increment global step
                observations, actions, rewards = batch
                
                with tf.GradientTape() as tape:
                    logits = model(observations)
                    loss = policy_gradient_loss(logits, actions, rewards)
                    tf.contrib.summary.scalar('loss', loss)
                    losses.append(loss)
                    tf.contrib.summary.scalar('batch_mean_entropy', policy_entropy(logits, mean=True))
                    tf.contrib.summary.scalar('batch_mean_reward', tf.reduce_mean(rewards))

                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables),
                                          global_step=global_step)

        print('done')
        print('mean training loss:', np.mean(losses))

        # save (checkpoint) the model every 10 epochs
        if (i_epoch + 1) % 10 == 0:
            if checkpoint:
                checkpoint.save(file_prefix=checkpoint_prefix)



def test(model, env, checkpoint, checkpoint_dir, movie_path=None, start_trim=0):
    
    # restore model and optimizer from latest checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    rewards, populations, entropies = play(model, env, episodes=1, movie_path=movie_path)
    plot_rewards_rule_deltas_and_populations(rewards=rewards, 
                                             state_props_list=populations, 
                                             prop_trajectory_states=(1,2),
                                             entropies=entropies,
                                             start_trim=start_trim)



def main():
    print('tf version:', tf.VERSION)
    print('tf keras version:', tf.keras.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-summary', default=False, action='store_true')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--movie-path', default=None)
    args = parser.parse_args()
    run(args)
        


if __name__ == '__main__':
    main()
