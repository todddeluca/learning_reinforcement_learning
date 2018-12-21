'''
Deep Q-Learning, MountainCar-v0 and CartPole-v0


DONE
====

Visualize Q-values, like in Example 10.1 from the Sutton and Barto book.

  
TODO
====


RESULTS
=======

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





class AgentModel(tf.keras.Model):
    '''
    Input: state
    Output: Q(s,a) for each action
    '''
    def __init__(self, num_actions, n_h=128, activation='elu', n_l=1):
        '''
        num_states: 
        '''
        super().__init__()
        self.num_actions = num_actions
        self.activation = activation
        self.n_l = n_l
        self.n_h = n_h
        self.dense1 = tf.keras.layers.Dense(n_h, activation=activation)
        self.dense2 = tf.keras.layers.Dense(num_actions) # Q(s,a)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


def discounted_returns(rewards, gamma=1.0, normalize=False):
    '''
    Compute the discounted returns (possibly normalized) for 
    the rewards of an episode
    '''
    disc_returns = np.zeros(len(rewards))
    #gamma = 1.0
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
    loss = tf.reduce_mean((targets - actuals)**2)
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
    positions = np.linspace(-1.2, 0.6, 20)
    speeds = np.linspace(-0.7, 0.7, 20)
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
    model_id = 'model02'
    data_dir = Path('/Users/tfd/data/2018/learning_reinforcement_learning') / exp_id
    batch_size = 32
    num_epoch = 1000 # number of buffers filled and then trained on
    num_episodes = 10 # number of episodes per buffer
    train_epochs = 1 # number of times to train model on each buffer
    learning_rate=1e-4
    epsilon = 0.1 # exploration rate
    gamma = 0.95 # discount rate
    checkpoint_dir = data_dir / f'{model_id}_checkpoints'
    checkpoint_prefix = checkpoint_dir / 'ckpt'
    # different tensorboard log dir for each run
    log_dir = data_dir / f'log/{model_id}/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    activation = 'tanh' #'elu'
    
    # make environment
    env = gym.make('MountainCar-v0')
#     env = gym.make('CartPole-v0')
    num_actions = env.action_space.n
    
    # tensorboard setup
    summary_writer = tf.contrib.summary.create_file_writer(str(log_dir), flush_millis=5000)
    
    # create model
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        model = AgentModel(num_actions)
        
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
        return
        
    if args.model_summary:
        num_epoch = 1
        num_episodes = 1
        # dummy training to initialize model (without saving a checkpoint)
        train(model, env, optimizer, global_step, 
              num_epoch=1, num_episodes=1, batch_size=batch_size, train_epochs=1)
        model.summary()
        return # don't train dummy model?

    if args.train:
        with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
            train(model, env, optimizer, global_step, 
                  num_epoch, num_episodes, batch_size, train_epochs,
                  checkpoint, checkpoint_prefix, epsilon=epsilon, gamma=gamma)

    if args.test:
        test(model, env, checkpoint, checkpoint_dir)


def train(model, env, optimizer, global_step, num_epoch, num_episodes, batch_size, train_epochs,
          checkpoint=None, checkpoint_prefix=None, epsilon=0.1, gamma=1.0):
        
    print('start training')
    print('global_step:', global_step.numpy())
    # lets do this thing!
    for i_epoch in range(num_epoch):
        print('epoch', i_epoch)

        # fill replay buffer with experiences
        buffer = []
        entropies = []
        rewards = []
        episode_reward_totals = []
        s0_returns = [] # returns from initial state
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
                qvalues = model.predict(obs_batch)
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
    print(args)
    run(args)
        


if __name__ == '__main__':
    main()
