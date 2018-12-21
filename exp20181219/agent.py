'''
Policy Gradients, MountainCar-v0 and CartPole-v0

I think MountainCar problem might be too hard for policy gradients. PG explores by taking random actions and 
to get to the top of the mountain (and get any reward other than -200), requires a long chain of correct actions.
The probability of getting there in 200 moves (episode length) is very small and so no reward signal is 
achieved. Every episode is -200 reward.

CartPole-v0 was solved by the agent after some learning rate tweaks.

See also:
- https://github.com/lguye/openai-exercise/tree/master/MountainCar-v0/PolicyGradient
- https://github.com/leimao/OpenAI_Gym_AI/blob/master/LunarLander-v2/REINFORCE/2017-05-24-v1/OpenAI_REINFORCE_FC_TF.py


DONE
====

- Discounted Monte carlo returns.
- Entropy using tensorflow cross entropy function to avoid NaN results.
  
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
    '''
    def __init__(self, num_actions, n_h=128):
        '''
        num_states: 
        '''
        super().__init__()
        self.num_actions = num_actions
        self.dense1 = tf.keras.layers.Dense(n_h, activation='elu')
#         self.dense2 = tf.keras.layers.Dense(n_h, activation='elu')
        self.dense3 = tf.keras.layers.Dense(num_actions) # logits

    def call(self, x):
        x = self.dense1(x)
#         x = self.dense2(x)
        x = self.dense3(x)
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


def policy_gradient_loss(logits, actions, rewards):
    '''
    Loss is: -A * log(p), where A is the advantage. A simple form of advantage is normalized reward.
    A more sophisticated form is U_t - V(s_t), where U_t is monte-carlo return
    or Q(a_t,s_t), the action value.

    rewards: batch of rewards, shape (batch_size,)
    logits: batch of action logits, shape (batch_size, num_actions)
    actions: batch of sampled boards, shape (batch_size,)
    '''
    # -log p(a|s)
    cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=actions # one-hot encoding happens within sparse_softmax_cross_entropy_with_logits
    )
    # J(w) = R * log(p(a|s,w))
    # L(w) = -R * log(p)
    loss = tf.reduce_sum(rewards * cross_entropies)
    return loss    


def policy_entropy(logits, mean=False):
    '''
    Compute the policy entropy, - \sum_a p(a) log p(a), for each policy in batch
    logits: shape (batch_size, ..., num_actions)
    returns: shape (batch_size,), each element is entropy of corresponding policy in batch
    Note: entropy reported in base e, not base 2.
    '''
    pi = tf.nn.softmax(logits, axis=-1)
    entropies = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=pi)
    if mean:
        entropies = tf.reduce_mean(entropies)
        
    return entropies
    

def sample_action(logits):
    # Using tensorflow eager execution for fun and education
    # logits shape: (batch_size, num_actions)
    # return shape: (batch_size,), a batch of sampled actions or a single sampled action.
    sample = tf.multinomial(logits=logits, num_samples=1)
    return tf.squeeze(sample, axis=-1) # (batch_size, 1) => (batch_size,)


def run(args):


    exp_id = 'exp20181219'
    model_id = 'model01'
    data_dir = Path('/Users/tfd/data/2018/learning_reinforcement_learning') / exp_id
    batch_size = 32
    num_epoch = 1000 # number of buffers filled and then trained on
    num_episodes = 20 # number of episodes per buffer
    train_epochs = 2 # number of times to train model on each buffer
    learning_rate=1e-4
    checkpoint_dir = data_dir / f'{model_id}_checkpoints'
    checkpoint_prefix = checkpoint_dir / 'ckpt'
    # different tensorboard log dir for each run
    log_dir = data_dir / f'log/{model_id}/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    # make environment
#     env = gym.make('MountainCar-v0')
    env = gym.make('CartPole-v0')
    num_actions = env.action_space.n
    
    # tensorboard setup
    summary_writer = tf.contrib.summary.create_file_writer(str(log_dir), flush_millis=10000)
    
    # create model
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        model = AgentModel(num_actions)
        
#     optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)    
    global_step = tf.train.get_or_create_global_step()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, global_step=global_step)

    if args.restore_latest:
        # restore model and optimizer from latest checkpoint
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

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
                  checkpoint, checkpoint_prefix)

    if args.test:
        test(model, env, checkpoint, checkpoint_dir)
    
    
def train(model, env, optimizer, global_step, num_epoch, num_episodes, batch_size, train_epochs,
          checkpoint=None, checkpoint_prefix=None):
        
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
                logits = model.predict(obs_batch)
                entropy = policy_entropy(logits, mean=True)
    #             print('entropy:', entropy)
                entropies.append(entropy)
                # print('logits:', logits)
                action = sample_action(logits)[0].numpy() # batch size of 1
                # print('action:', action)
                next_obs, reward, done, info = env.step(action)
                episode_rewards.append(reward)
                rewards.append(reward)
                episode_obs.append(obs)
                episode_actions.append(action)
                obs = next_obs
                env.render()
                if done:
                    episode_reward_totals.append(np.sum(episode_rewards))
                    # monte carlo returns with no discounting
                    epi_returns = discounted_returns(episode_rewards, gamma=1.0, normalize=True)
                    s0_returns.append(epi_returns[0])
                    buffer.extend(zip(episode_obs, episode_actions, epi_returns))

        print('last obs:', obs)
        mean_entropy = np.mean(entropies)
        print('mean entropy:', mean_entropy)
        tf.contrib.summary.scalar('mean_entropy', mean_entropy)

        mean_step_reward = np.mean(rewards)
        print('mean_step_reward:', mean_step_reward)
        tf.contrib.summary.scalar('mean_step_reward', mean_step_reward)
        
        mean_s0_return = np.mean(s0_returns)
        print('mean_s0_return:', mean_s0_return)
        tf.contrib.summary.scalar('mean_s0_return', mean_s0_return)
        
        mean_episode_reward = np.mean(episode_reward_totals)
        print('mean_episode_reward:', mean_episode_reward)
        tf.contrib.summary.scalar('mean_episode_reward', mean_episode_reward)

        # make dataset from experience buffer
        gen = lambda: (exp for exp in buffer)
        ds = (tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32, tf.float32))
              .shuffle(len(buffer))
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
                cp_path = checkpoint.save(file_prefix=checkpoint_prefix)
                print('saved checkpoint to', cp_path)



def test(model, env, checkpoint, checkpoint_dir, num_episodes=100):
    
    # play game
    for i_epi in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            obs_batch = obs[None].astype(np.float32)
            logits = model.predict(obs_batch)
            action = sample_action(logits)[0].numpy()
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            env.render()


def main():
    print('tf version:', tf.VERSION)
    print('tf keras version:', tf.keras.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-summary', default=False, action='store_true')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--restore-latest', default=False, action='store_true')
    args = parser.parse_args()
    run(args)
        


if __name__ == '__main__':
    main()
