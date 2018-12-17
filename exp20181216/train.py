'''


Some references I found useful while developing this code:
- https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-rl-pong/trainer/task.py
- https://www.youtube.com/watch?v=t1A3NTttvBA
- https://github.com/MadcowD/tensorgym/blob/master/tf_demo/complete.py
- https://www.tensorflow.org/tutorials/eager/

Model
=====
- DCGAN tensorflow eager execution keras example
  - Uses training keyword argument in model.__call__ or model.call override. 
  - Passes it to batchnorm and dropout layers (and anything else that behaves differently between training and inference.
  - Clean example of modern tensorflow 2.0 style:
    - keras model
    - eager execution
    - tf.contrib.eager.defun
    - GradientTape
    - checkpoints
  https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/eager/python/examples/generative_examples/dcgan.ipynb

Datasets
========

https://www.tensorflow.org/guide/datasets
https://www.tensorflow.org/tutorials/eager/eager_basics

Metrics
=======

For visualizing model performance.
https://www.tensorflow.org/guide/eager#object-oriented_metrics
https://www.tensorflow.org/api_docs/python/tf/contrib/eager/metrics/Metric

Loss Functions
==============

The policy gradient loss function is loss = R * -log pi(a|s), where R might be baselined or normalized.

For a simple exposition on policy gradient reinforcement learning in tensorflow, see:
- Video: https://www.youtube.com/watch?v=t1A3NTttvBA
- Code: https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-rl-pong/trainer/task.py

For tutorials on optimizing custom loss function using tensorflow eager execution and keras, see:
- https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough
- https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/eager/python/examples/generative_examples/cvae.ipynb

Checkpoints
===========

For loading and saving models.

TODO
====

- rewrite env in tensorflow. 
  - reward can be tensorflow.  with spark probability distribution. hmm maybe not. computing the connected components to determine the yield of a board. In tensorflow? 
  - next board can be tensorflow. can run batches of games at a time.
  - simpler if each game is exactly L^2 moves per episode.
Stage 1: metrics and persistence
- metrics: track model performance over time, model entropy over time
- saving and loading model
Stage 2: legal actions and baselined rewards
- use reward baseline: (r - mean(r)/std(r)) or (r - mean(r)) or r - v(s)
- only choose legal actions. will this allow training to 
  focus on making good moves instead of just legal ones?
- replay buffer: get right balance between gathering experiences and training model.
  - how does on-policy replay buffer in a2c work?
  - can off-policy replay buffer, like in SAC work here?
Stage 3: 
- use return?
- hyperparam tuning
- training loop: get the right balance between gathering experiences and training model
- convnet model architecture for bigger boards?

'''


from pathlib import Path
import gym
import gym_hotforest

import argparse

import numpy as np
import tensorflow as tf
# import tensorflow.contrib.eager as tfe


def make_pg_model(input_shape, n_h, num_actions):
    '''
    Policy gradient model.
    The model takes a board as input and outputs logits (action preference for a softmax)
    for the actions.
    input_shape: the shape of a hot forest board (excluding batch dimension)
    n_h: number of hidden units
    num_actions: number of output units.
    '''
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(n_h, input_shape=input_shape, activation='elu'),
        tf.keras.layers.Dense(num_actions), # logits
    ])
    return model

    
def loss(logits, actions, rewards):
    '''
    Loss is: -A * log(p), where A is the advantage. A simple form of advantage is normalized reward.
    A more sophisticated form is U_t - V(s_t), where U_t is monte-carlo return
    or Q(a_t,s_t), the action value.

    rewards: batch of rewards, shape (batch_size,)
    logits: batch of action logits, shape (batch_size, num_actions)
    actions: batch of sampled actions, shape (batch_size,), action is index from 0..(num_actions - 1)
    '''
    # log pi(a|s)
    cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=actions # one-hot encoding happens within sparse_softmax_cross_entropy_with_logits
    )
    losses = tf.reduce_sum(rewards * cross_entropies)
    return losses    


def policy_entropy(logits, mean=False):
    '''
    Compute the policy entropy, - \sum_a p(a) log p(a), for each policy in batch
    logits: shape (batch_size, num_actions)
    returns: shape (batch_size,), each element is entropy of corresponding policy in batch
    '''
    # -np.sum(action_probs * np.log2(action_probs), axis=-1)

    pi = tf.nn.softmax(logits, axis=-1)
    entropies = tf.reduce_sum(-1 * pi * tf.log(pi), axis=-1)
    if mean:
        entropies = tf.reduce_mean(entropies)
        
    return entropies
    
    
def make_dataset(gen, shuffle_size, batch_size):
    '''
    buffer: replay buffer of tuples of (obs, action, reward, next_obs, done) or something like that
    Dataset is a sequence of (obs, action, reward) for training policy gradient model
    '''            
    ds = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32, tf.float32))
    ds = ds.shuffle(shuffle_size).repeat().batch(batch_size)
    return ds


def train():
    
    data_dir = Path('/Users/tfd/data/2018/learning_reinforcement_learning/exp20181216')

    shuffle_size = 1024 
    batch_size = 32
    buffer_size = 1024
    buffer_epochs = 10 # number of times to collect a buffer of experiences
    train_epochs = 2 # number of times to train model on each buffer
    n_epoch = 100
    
    # make environment
    env_id = 'hotforest-l4-v0'
    print('training model on env', env_id)
    env = gym.make(env_id)
    num_actions = env.action_space.n
    board_shape = env.observation_space.shape
    print('num_actions:', num_actions)
    print('board_shape:', board_shape)

    # make model
    model = make_pg_model(board_shape, n_h=32, num_actions=num_actions)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)
#     optimizer=tf.train.AdamOptimizer(0.001)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    
    # repeat
    for i_epoch in range(n_epoch):
        print('epoch', i_epoch)
        # fill replay buffer with experiences
        buffer = []
        done = False
        obs = env.reset()
        episode_count = 0
        entropies = []
        for i_buf in range(buffer_size):        
            # obs_batch is (1, length, length) shape array of observations
            obs_batch = obs[None].astype(np.float64) # prepend batch dimension to make batch of size 1
            # logits is (1, num_actions) shape array of action logits
            logits = model.predict(obs_batch)
            entropy = policy_entropy(logits, mean=True)
            entropies.append(entropy)
#             print('logits:', logits)
            action = sample_action(logits)[0] # batch size of 1
#             print('action:', action)
            obs, reward, done, info = env.step(action)
            buffer.append((obs, action, reward))
            if done:
                episode_count += 1
                # start new episode
                obs = env.reset()
                done = False
                
        print('episode count:', episode_count)
        mean_entropy = np.mean(entropies)
        print('mean entropy:', mean_entropy)
        
        # make dataset from experience replay buffer
        gen = lambda: (exp for exp in buffer)
        ds = (tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32, tf.float32))
              .shuffle(shuffle_size)
              .batch(batch_size))
        
        print('training model...', end='')
        for i_te in range(train_epochs):
            print(f'{i_te}...', end='')
            for i_batch, batch in enumerate(ds):
                observations, actions, rewards = batch

                with tf.GradientTape() as tape:
                    logits = model(observations)
                    losses = loss(logits, actions, rewards)
                grads = tape.gradient(losses, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print('done')
    
    # visualize results of trained model
    play(model, env, episodes=4)


def test():
    # load model
    model = None
    # make env
    env = None
    play(model, env, episodes=1)


def sample_action(logits):
    # Using tensorflow eager execution for fun and education
    # logits shape: (batch_size, num_actions)
    # return shape: (batch_size,), a batch of sampled actions or a single sampled action.
    tens = tf.convert_to_tensor(logits)
    sample = tf.multinomial(logits=tens, num_samples=1)
    return tf.reshape(sample, shape=[-1]).numpy() # (batch_size, 1) => (batch_size,)


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
            print('logits:', logits)
            action = sample_action(logits)[0] # batch size of 1
            print('action:', action)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            actions.append(action)
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
