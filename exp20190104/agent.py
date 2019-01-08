'''
Deep Q-Learning, MountainCar-v0 and CartPole-v0


DONE
====

- Experience Replay
- Target Network, called outside gradient tape block
- 2D Embedding for ConvNet

TODO
====

l2 regularization

'''


from pathlib import Path
import argparse
import datetime
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.colors
import numbers
import numpy as np

import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

import gym



class ReplayBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.idx = 0
        self.buffer = []
        
    def add(self, data):
        '''
        Start by filling the buffer with data. When full, overwrite oldest data.
        '''
        if len(self.buffer) < self.max_size:
            self.buffer.append(data)
            self.idx += 1
        else:
            self.idx = (self.idx + 1) % self.max_size
            self.buffer[self.idx] = data
            
    def __len__(self):
        return len(self.buffer)
    
    def sample(self, batch_size):
        '''
        return a tuple of nparrays, (obs, actions, rewards, next_obs, dones), suitable for making 
        tf.data.Dataset
        '''
        sample_idxs = np.random.choice(len(self.buffer), size=batch_size, replace=True)
        obs = np.array([self.buffer[i][0] for i in sample_idxs], dtype=np.float32)
        actions = np.array([self.buffer[i][1] for i in sample_idxs], dtype=np.int32)
        rewards = np.array([self.buffer[i][2] for i in sample_idxs], dtype=np.float32)
        next_obs = np.array([self.buffer[i][3] for i in sample_idxs], dtype=np.float32)
        dones = np.array([self.buffer[i][4] for i in sample_idxs], dtype=bool)
        
        return obs, actions, rewards, next_obs, dones

    
class ConvAgentModel(tf.keras.Model):
    '''
    Input: state
    Output: Q(s,a) for each action
    '''
    def __init__(self, num_actions, size=(16, 16), n_h=128, activation='relu'):
        '''
        n_h: number of hidden units or filters per layer.
        size: an integer or pair of integers, height and width.
        
        Project the input into a 2d image then use a convolutional net.
        '''
        super().__init__()
        self.num_actions = num_actions
        self.n_h = n_h
        self.size = (size, size) if isinstance(size, numbers.Number) else size
        print('ConvAgentModel .size=', self.size)
        self.activation = activation
        
        # embed input into a (height, width, channel) tensor space, for Conv2D
        self.dense1 = tf.keras.layers.Dense(self.size[0] * self.size[1] * self.n_h, activation=activation) 
        
        # residual blocks
        self.conv2a = tf.keras.layers.Conv2D(self.n_h, kernel_size=(3, 3), padding='same', activation=activation)
        self.bn2a = tf.keras.layers.BatchNormalization()
        
        self.conv2b = tf.keras.layers.Conv2D(self.n_h, kernel_size=(3, 3), padding='same', activation=activation)
        self.bn2b = tf.keras.layers.BatchNormalization()
        
        self.conv2c = tf.keras.layers.Conv2D(self.n_h, kernel_size=(3, 3), padding='same', activation=activation)
        self.bn2c = tf.keras.layers.BatchNormalization()
        
        # output q-values
        self.dense3 = tf.keras.layers.Dense(num_actions) # Q(s,a)

    def call(self, x, n=20, training=False):
        x = tf.convert_to_tensor(x) # convert numpy arrays
        
        # input embedding
        x = self.dense1(x)
        x = tf.reshape(x, shape=(-1, self.size[0], self.size[1], self.n_h)) # shape (batch_size, height, width, n_h)

        # residual layer
        x_in = x
        x = self.conv2a(x)
        x = self.bn2a(x, training=training)
        x += x_in

        x_in = x
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x += x_in

        x_in = x
        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        x += x_in

        x = tf.reshape(x, shape=(-1, self.size[0] * self.size[1] * self.n_h)) # shape (batch_size, height * width * n_h)

        x = self.dense3(x) # shape (batch_size, num_actions)

        return x



# Helpful example or residual block: https://www.tensorflow.org/tutorials/eager/custom_layers
class AgentModel(tf.keras.Model):
    '''
    Input: state
    Output: Q(s,a) for each action
    '''
    def __init__(self, num_actions, n_h=128, activation='relu'):
        '''
        num_states: 
        '''
        super().__init__()
        self.num_actions = num_actions
        self.n_h = n_h
        self.activation = activation
        
        # project input into n_h dimensional vector space
        self.dense1 = tf.keras.layers.Dense(n_h, activation=activation) 
        
        # residual blocks
        self.dense2a = tf.keras.layers.Dense(n_h, activation=activation)
        self.bn2a = tf.keras.layers.BatchNormalization()
        self.dense2b = tf.keras.layers.Dense(n_h, activation=activation)
        self.bn2b = tf.keras.layers.BatchNormalization()
        self.dense2c = tf.keras.layers.Dense(n_h, activation=activation)
        self.bn2c = tf.keras.layers.BatchNormalization()
        
        # output q-values
        self.dense3 = tf.keras.layers.Dense(num_actions) # Q(s,a)

    def layer_summary(self, x, layer, name, n=1):
        # histogram summaries are heavy; avoid recording all of them.
        with tf.contrib.summary.record_summaries_every_n_global_steps(n=n):
#             tf.contrib.summary.histogram(f'{name}_out', x)
#             tf.contrib.summary.histogram(f'{name}_weights', layer.weights[0])
#             tf.contrib.summary.histogram(f'{name}_bias', layer.weights[1])
            pass
        
    def call(self, x, n=20, training=False):
        x = tf.convert_to_tensor(x) # convert numpy arrays
        
        x = self.dense1(x)
        if training:
            self.layer_summary(x, self.dense1, 'dense1', n=n)

        # residual layer
        x_in = x
        x = self.dense2a(x)
        if training:
            self.layer_summary(x, self.dense2a, 'dense2a', n=n)
        x = self.bn2a(x, training=training)
        x += x_in

        x_in = x
        x = self.dense2b(x)
        if training:
            self.layer_summary(x, self.dense2b, 'dense2b', n=n)
        x = self.bn2b(x, training=training)
        x += x_in

        x_in = x
        x = self.dense2c(x)
        if training:
            self.layer_summary(x, self.dense2c, 'dense2c', n=n)
        x = self.bn2c(x, training=training)
        x += x_in

        x = self.dense3(x)
        if training:
            self.layer_summary(x, self.dense3, 'dense3', n=n)

        return x


def value_loss(qvalues, actions, rewards, next_qvalues, dones, gamma=1.0):
    '''
    Loss is MSE between q-learning target and qvalues of selected actions, 
    that is, between (r + gamma * max_a' q(s',a')) and q(s,a)
    
    qvalues: batch of action values for state, shape (batch_size, num_actions)
    actions: batch of sampled actions, shape (batch_size,)
    rewards: batch of rewards, shape (batch_size,)
    qvalues: batch of action values for next state, shape (batch_size, num_actions) 
    dones: batch of done flags indicating if the episode is over.
    '''
    max_next_qvalues = tf.reduce_max(next_qvalues, axis=-1)
    targets = rewards + gamma * tf.where(dones, tf.zeros_like(rewards), max_next_qvalues)
    actuals = tf.reduce_sum(tf.one_hot(actions, depth=qvalues.shape[-1]) * qvalues, axis=-1)
    loss = tf.reduce_mean((targets - actuals)**2) # openai deepq uses huber loss instead of MSE
    return loss    


def sample_action(qvalues, epsilon=0.0):
    '''
    Epsilon-greedy action sampling.
    
    epsilon: probability of sampling a random action.
    qvalues: shape (batch_size, num_actions)
    returns: sampled action index
    '''
    ps = tf.random_uniform(shape=qvalues.shape[:-1]) # (batch_size,)
    random_actions = tf.squeeze(tf.multinomial(tf.ones_like(qvalues), num_samples=1), axis=-1) # (batch_size,)
    max_actions = tf.argmax(qvalues, axis=-1)
    selected_actions = tf.where(tf.greater_equal(ps, epsilon), max_actions, random_actions)
    return selected_actions


def display(model):
    print(model.summary())
#     from tf.keras.utils.vis_utils import model_to_dot # visualize model
#     return SVG(model_to_dot(model).create(prog='dot', format='svg'))


def visualize_model(model):
    '''
    Visualize qvalues for mountain car model, to compare to plots in Sutton & Barto.
    Points are colored according to the action selected by the greedy policy.
    '''
    positions = np.linspace(-1.2, 0.6, 21)
    speeds = np.linspace(-0.07, 0.07, 21)
    results = []
    observations = []
    for position in positions:
        for speed in speeds:
            observations.append((position, speed))
    
    qvalues = model(np.array(observations, dtype=np.float32)).numpy()
    
    print('some qvalues:', qvalues[np.random.choice(len(qvalues), size=10)])
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
    plt.title('red=left,grey=null,green=right')
    plt.show()


'''

RESULTS
=======

Model03:
1000s of time steps: description
65: undergoing a landscape transition
75: developing a strategy in a new landscape
85: strategy collapse? qvalue collapse? Learned representation collapse?
88: numerical instability? diverging qvalues? Use huber loss? Use l2 weight decay?
95: nan qvalues.

Model04: 8x8x32 ConvAgentModel is too slow on laptop. Much bigger than AgentModel with n_h=64

Model05: 8x8x8 ConvAgentModel
slow. enough channels to learn diverse features? enough height and width? A bigger computer would be useful.

Model06: 4x4x32 ConvAgentModel
tiny image, more channels.
/Users/tfd/data/2018/learning_reinforcement_learning/exp20190104/model06_checkpoints/ckpt-60
60: some spiraling of policy and q-values.
85: a little more spiraling? the car is almost reaching the flag when the model was tested.
100: policy visualization looks like it is struggling to learn a spiral. getting closer.
Continue training for another 100k steps.  Restore latest model, rebuild buffer. 
104: Crazy huge changes in q-values policy visualization. Flatter, more linear. Bad policy.
137: bowl shaped qvalues. working policy.
'''

def run(args):


    exp_id = 'exp20190104'
    model_id = 'model06'
    data_dir = Path('/Users/tfd/data/2018/learning_reinforcement_learning') / exp_id
    batch_size = 32
    target_update_freq = 1000 # update target network every n episodes
    train_freq = 1 # update network every n episodes
    learning_starts = 1000
    checkpoint_freq=1000
    buffer_size = 10000 # number of tuples to keep in the buffer
    num_steps = 100000 #
    learning_rate=1e-5
    epsilon = 0.1 # exploration rate
    gamma = 1.0 # discount rate
    checkpoint_dir = data_dir / f'{model_id}_checkpoints'
    checkpoint_prefix = checkpoint_dir / 'ckpt'
    # different tensorboard log dir for each run
    log_dir = data_dir / f'log/{model_id}/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    model_type = 'ConvAgentModel' # 'AgentModel'
    activation = 'relu'
    n_h = 32 # number of hidden units
    conv_img_size = (4, 4) # (8, 8)
    
    # make environment
    env = gym.make('MountainCar-v0')
#     env = gym.make('CartPole-v0')
    num_actions = env.action_space.n
    
    print('model_type:', model_type)
    if model_type == 'ConvAgentModel':
        model = ConvAgentModel(num_actions, size=conv_img_size, activation=activation, n_h=n_h)
        target_model = ConvAgentModel(num_actions, size=conv_img_size, activation=activation, n_h=n_h)
    elif model_type == 'AgentModel':
        model = AgentModel(num_actions, activation=activation, n_h=n_h)    
        target_model = AgentModel(num_actions, activation=activation, n_h=n_h)  
        
#     optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
#     optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
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
        train(model, target_model, env, optimizer, global_step, 
              num_epoch=1, num_episodes=1, batch_size=batch_size, train_epochs=1)
        model.summary() # print summary and graph of model
    elif args.train:
        # use tensorboard to log training progress
        summary_writer = tf.contrib.summary.create_file_writer(str(log_dir), flush_millis=5000)
        with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            train(model, target_model, env, optimizer, global_step,
                  batch_size, buffer_size, target_update_freq, train_freq, num_steps,
                  checkpoint, checkpoint_prefix, epsilon=epsilon, gamma=gamma,
                  checkpoint_freq=checkpoint_freq, learning_starts=learning_starts)
    elif args.test:
        test(model, env, render=args.render)
    else:
        print('doing nothing.')


def train(model, target_model, env, optimizer, global_step, 
          batch_size, buffer_size, target_update_freq, train_freq, num_steps,
          checkpoint=None, checkpoint_prefix=None, checkpoint_freq=10000, epsilon=0.1, gamma=1.0,
          learning_starts=1000):
    '''
    Loosely based on https://github.com/openai/baselines/blob/master/baselines/deepq/deepq.py
    
    checkpoint_freq: save a checkpoint every checpoint_freq steps.
    learning_starts: time step at which training begins. Before this, just fill buffer.
    '''
        
    print('starting training')
    print('global_step:', global_step.numpy())

    target_model.set_weights(model.get_weights())
    
    # lets do this thing!
    i_step = 0
    buffer = ReplayBuffer(buffer_size)
    done = True
    episode_reward_totals = [] # total reward for each episode
    while i_step < num_steps:
        if (i_step + 1) % 100 == 0:
            print('i_step', i_step)
            
        # start an episode
        if done:
            done = False
            obs = env.reset()
            episode_rewards = []
        
        # take a step
        qvalues = model(obs[None].astype(np.float32)) # (batch_size, num_actions)
#       print('qvalues:', qvalues)
        action = sample_action(qvalues, epsilon)[0].numpy() # batch size of 1
#       print('action:', action)
        next_obs, reward, done, info = env.step(action)
        buffer.add((obs, action, reward, next_obs, done))
        episode_rewards.append(reward)
        obs = next_obs

        # end an episode
        if done:
            total_episode_reward = np.sum(episode_rewards)
            episode_reward_totals.append(total_episode_reward)
            tf.contrib.summary.scalar('episode_reward', total_episode_reward)
            print('last episode obs:', obs)

        # train model
        if i_step >= learning_starts and i_step % train_freq == 0:
            # dataset is a randomly sampled batch from replay buffer
            batch = buffer.sample(batch_size)
            
            observations, actions, rewards, next_obervations, dones = batch
#             print('next_obervations:', next_obervations)
#             print('type(next_obervations):', type(next_obervations))
            next_qvalues = target_model(next_obervations)
            
            with tf.GradientTape() as tape:
                qvalues = model(observations, training=True)
                loss = value_loss(qvalues, actions, rewards, next_qvalues, dones, gamma=gamma)
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('batch_mean_reward', tf.reduce_mean(rewards))

            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables), global_step=global_step)
        
        # update target model from train model
        if i_step >= learning_starts and i_step % target_update_freq == 0:
            print('updating target_model at i_step', i_step)
            target_model.set_weights(model.get_weights())

        
        # save (checkpoint) the model every 
        if (i_step + 1) % checkpoint_freq == 0:
            if checkpoint:
                cp_path = checkpoint.save(file_prefix=checkpoint_prefix)
                print('saved checkpoint to', cp_path)

        i_step += 1

        
def test(model, env, num_episodes=100, epsilon=0.0, render=False):
    '''
    Play game for num_episodes episodes. Use greedy policy (epsilon == 0).
    '''
    episode_rewards = []
    for i_epi in range(num_episodes):
        print('i_epi:', i_epi)
        episode_reward = 0
        done = False
        obs = env.reset() # shape (obs_dims,)
        while not done:
            qvalues = model(obs[None].astype(np.float32), training=False)
            action = sample_action(qvalues, epsilon=epsilon)[0].numpy()
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            obs = next_obs
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        
    print('num_episodes:', num_episodes)
    print('mean episode reward:', np.mean(episode_rewards))


def main():
    print('tf version:', tf.VERSION)
    print('tf keras version:', tf.keras.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-summary', default=False, action='store_true', help='Print a summary of the model')
    parser.add_argument('--train', default=False, action='store_true', help='train a model')
    parser.add_argument('--test', default=False, action='store_true', help='evaluate a model')
    parser.add_argument('--no-render', dest='render', default=True, action='store_false', help='Do not render environment when evaluating model')
    parser.add_argument('--restore-latest', default=False, action='store_true', help='Restore the latest model checkpoint')
    parser.add_argument('--restore-model', metavar='PATH', help='Restore the model checkpoint located at the given path')
    parser.add_argument('--visualize', default=False, action='store_true', help='Visualize the model q-value landscape')
    args = parser.parse_args()
    run(args)    


if __name__ == '__main__':
    main()
