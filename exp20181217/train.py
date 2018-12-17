'''
Policy Gradients, LVCA, Eager Tensorflow


https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/eager/python/examples/generative_examples/dcgan.ipynb

TODO
====

- checkpointing. saving and loading models
- env/reward in tensorflow. for this game, env just gives reward, since action is next board.
  - reward
- memory
  - model input: memory + board board
  - model output: next memory + next board
- metrics: track model performance over time, model entropy over time
- use reward baseline: (r - mean(r)/std(r)) or (r - mean(r)) or r - v(s)
- replay buffer: get right balance between gathering experiences and training model.
  - use old experiences, like DQN or SAC?
- use return?
- hyperparam tuning
- training loop: get the right balance between gathering experiences and training model
- convnet model architecture for bigger boards?

'''


from pathlib import Path
import lvcaenv

import argparse

import numpy as np
import tensorflow as tf
# import tensorflow.contrib.eager as tfe


class CAModel(tf.keras.Model):
    '''
    To be a cellular automata, this model only looks at neighborhood information once,
    via a Conv2d layer with a (3, 3) kernel
    '''
    def __init__(self, num_states, input_shape):
        '''
        num_states: 
        '''
        super().__init__()
        self.num_states = num_states
        self.reshape1 = tf.keras.layers.Reshape(list(input_shape) + [1], input_shape=input_shape) # add channel dimension
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='elu') # neighborhood
        self.conv2 = tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='elu')
        self.conv3 = tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='elu')
        self.conv4 = tf.keras.layers.Conv2D(num_states, (1, 1), strides=(1, 1), padding='same') # logits

    def call(self, x):
        x = self.reshape1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    
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


def train():
    
    data_dir = Path('/Users/tfd/data/2018/learning_reinforcement_learning/exp20181216')

    shuffle_size = 1024 
    batch_size = 32
    buffer_size = 1024
    buffer_epochs = 10 # number of times to collect a buffer of experiences
    train_epochs = 2 # number of times to train model on each buffer
    n_epoch = 100
    learning_rate=1e-5
    length = 8 # length of cellular automata board
    episode_len = 16 # number of time steps of cellular automata
    
    # make environment
    env_id = 'lvca-l2-v0'
    print('training model on env', env_id)
    env = lvcaenv.LvcaEnv(length=length, episode_len=16, reward_type='cosine', cosine_noise=0.001)
    num_states = env.num_states
    length = env.length
    print('num_states:', num_states)
    print('length:', length)

    # make model
    model = CAModel(num_states, input_shape=[length, length])
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
#     optimizer=tf.train.AdamOptimizer(0.001)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    
    # repeat
    for i_epoch in range(n_epoch):
        print('epoch', i_epoch)
        
        # fill replay buffer with experiences
        buffer = []
        done = False
        obs = env.reset()
        print('first obs:', obs)
        episode_count = 0
        entropies = []
        for i_buf in range(buffer_size):        
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
            obs, reward, done, info = env.step(action)
            buffer.append((obs, action, reward))
            if done:
                episode_count += 1
                # start new episode
                obs = env.reset()
                done = False
        
        print('last obs:', obs)
        print('episode count:', episode_count)
        mean_entropy = np.mean(entropies)
        print('mean entropy:', mean_entropy)
        
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
                observations, actions, rewards = batch

                with tf.GradientTape() as tape:
                    logits = model(observations)
                    loss = policy_gradient_loss(logits, actions, rewards)
                    losses.append(loss)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print('done')
        print('mean training loss:', np.mean(losses))
        
#         play(model, env)
    
    # visualize results of trained model
    play(model, env, episodes=4)


def test():
    # load model
    model = None
    # make env
    env = None
    play(model, env, episodes=1)


def play(model, env, episodes=1):
    for eps_i in range(episodes):
        obs = env.reset()
        rewards = []
        actions = []
        max_timesteps = 10000
        for i in range(max_timesteps):
            env.render()
            # obs_batch is (1, length, length) shape array of observations
            obs_batch = obs[None].astype(np.float64) # prepend batch dimension to make batch of size 1
            # logits is (1, num_actions) shape array of action logits
            logits = model.predict(obs_batch)
#             print('logits:', logits)
            action = sample_board(logits)[0] # batch size of 1
            print('action:', action)
            obs, reward, done, info = env.step(action.numpy())
            print('reward:', reward)
            rewards.append(reward)
            actions.append(action)
            print('pops:', np.bincount(obs.ravel()))
            if done:
                break

        env.render()

        print('rewards:', rewards)
        print('actions:', actions)


def main():
    print('tf version:', tf.VERSION)
    print('tf keras version:', tf.keras.__version__)

    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    train()
    


if __name__ == '__main__':
    main()
