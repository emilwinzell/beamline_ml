#
#  RL model from: https://keras.io/examples/rl/ddpg_pendulum/
# 
# Proximal Policy Optimization
# Emil Winzell April 2022
import os
import sys
#sys.stdout = open('ddpg_output.txt','wt')

import xrt.backends.raycing.run as rr
import xrt.plotter as xrtp
import xrt.runner as xrtr


import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#import gym
import scipy.signal
import time
from scipy import optimize
import pandas as pd

from vsb import VeritasSimpleBeamline

class RaycingEnv():
    """
    ### RaycingEnv

    ### Action Space
    The action is a `ndarray` with shape `(5,)` representing the ways to shift the M4 mirror
    | Num | Action   | Min  | Max |
    |-----|----------|--------|-------|
    | 0   | Pitch    | -0.003 | 0.003 |
    | 1   | Yaw      | -0.001 | 0.001 |
    | 2   | Roll     | -0.001 | 0.001 |
    | 3   | Lateral  | -5.0   | 5.0   |
    | 4   | Vertical | -2.5   | 2.5   |

    ### Observation Space
    The observation is a `ndarray` with shape `(3,)` representing the minumum full-width-half-maximas of the histograms of the x and y axis. And the gap
    between the location of the minimas along the z axis.

    | Num | Observation      | Min   | Max  |
    |-----|------------------|-------|------|
    | 0   | FWHMx            | 0.0   | 50.0 |
    | 1   | FWHMy            | 0.0   | 50.0 |
    | 2   | Gap              | -10.0 | 10.0 |

    ### Rewards
    The reward function is defined as:

    if done:
        r = 300
    else:
        r = -  FWHMx -  FWHMy - abs(Gap)


    """
    
    def __init__(self):
        super(RaycingEnv, self).__init__()
        self.beamline = VeritasSimpleBeamline() # vsb object
        rr.run_process = self.beamline.run_process

        self.bounds = np.array([self.beamline.p_lim, self.beamline.y_lim, self.beamline.r_lim, self.beamline.l_lim, self.beamline.v_lim])
        
        #self.action_space = spaces.Box(-self.bounds, self.bounds, dtype=np.float32)
        #self.observation_space = spaces.Box(np.array([0.0, 0.0, -10.0]),
        #                                    np.array([50.0, 50.0, 10.0]), dtype=np.float32)
        
        self.num_steps = 0
        self.state = None
        self.f_x = []
        self.f_y = []


    def __calculate_fwhm(self,data,lim,N):
        x = np.linspace(lim[0], lim[-1], N)
        n = np.count_nonzero(data==0)//2+1
        est_std = (lim[-n]-lim[n])/5.0
        est_mean = lim[len(lim)//2]
        est_amp = np.max(data)*2.5
        def gaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp(-((x - mean) / np.sqrt(2) / stddev)**2)
        
        [amp,mean,std],_ = optimize.curve_fit(gaussian, lim, data,p0=[est_amp,est_mean,est_std])
        std = abs(std)
        fwhm = 2*np.sqrt(2*np.log(2))*std

        return fwhm

    def __calculate_argmin(self,data,xlim,N):
        x = np.linspace(xlim[0], xlim[-1], N)

        def poly(x, a0,a1,a2):
            return a0 + a1 *x + a2*x**2
    
        [a0,a1,a2],_ = optimize.curve_fit(poly, xlim, data)

        if a2 > 0:
            amin = -a1/(2*a2)
        else:
            amin = 1000.0
        
        return amin

    def __calculate_vad(self,data,xlim):

        ub = np.max(data)
        lb = np.min(data)
        mapk = 28/(ub-lb)
        mapm = 14-mapk*ub

        rescaled = mapk*np.array(data)+mapm

        def dis_cont(x, a1,b1,a2,b2,c):
            return np.where(x <= c, a1*x+b1, a2*x+b2)

        [k1,m1,k2,m2,r],_ = optimize.curve_fit(dis_cont, xlim, rescaled)
        
        

        inc1 = np.arctan(k1)*180/np.pi
        inc2 = np.arctan(k2)*180/np.pi
        value = abs(inc1+inc2)
        return value

    def __get_observation(self):
        f_x = []
        f_y = []
        for plot in self.beamline.plots:
            xt1D = np.append(plot.xaxis.total1D,0.0)
            xBins = plot.xaxis.binEdges
            yt1D = np.append(plot.yaxis.total1D,0.0)
            yBins = plot.yaxis.binEdges

            if plot.dx == 0: 
                f_x.append(50.0)
            else:
                f_x.append(self.__calculate_fwhm(xt1D,xBins,1000))
            if plot.dy == 0:
                f_y.append(50.0)
            else:
                f_y.append(self.__calculate_fwhm(yt1D,yBins,1000))
            
        xmin = self.__calculate_argmin(f_x, self.beamline.scrTgt11.dqs, 1000)
        vad = self.__calculate_vad(f_y, self.beamline.scrTgt11.dqs)
        ymin = self.beamline.scrTgt11.dqs[np.argmin(f_y)]
        gap = (xmin-ymin)/1000.0
        vad = vad/100.0 
        FWHMx = min(f_x)
        FWHMy = min(f_y)
        
        state =  np.array([FWHMx,FWHMy,gap,vad])

        reward = -FWHMx - FWHMy - abs(np.clip(gap,-1.0,1.0)) - vad

        # Done?
        done = False
        if gap < 3.0/1000.0 and FWHMx < 0.01 and FWHMy < 0.005 and vad < 1.0/100.0:
            done = True
            reward = 300 # max possible steps is 290
        
        return state, reward, done, f_x, f_y


    def __take_action(self, sampled_actions):
        for plot in self.beamline.plots:
                plot.xaxis.limits = None
                plot.yaxis.limits = None
                plot.xaxis.binEdges = np.zeros(self.beamline.bins + 1)
                plot.xaxis.total1D = np.zeros(self.beamline.bins)
                plot.yaxis.binEdges = np.zeros(self.beamline.bins + 1)
                plot.yaxis.total1D = np.zeros(self.beamline.bins)

        # change bealine params
        action = np.clip(sampled_actions, -self.bounds, self.bounds)
        self.beamline.update_m4(action)

        sys.stdout = open(os.devnull, 'w')
        xrtr.run_ray_tracing(self.beamline.plots,repeats=self.beamline.repeats, 
                    updateEvery=self.beamline.repeats, beamLine=self.beamline)#,threads=3,processes=8)
        sys.stdout = sys.__stdout__

    def reset(self):
        #self.beamline = VeritasSimpleBeamline()
        self.beamline.reset()
        self.__take_action(np.zeros(5,dtype=np.float32))
        
        self.num_steps = 0
        self.state,_,_,_,_ = self.__get_observation()  # calculate state
        return self.state

    def step(self, action):
        self.__take_action(action)

        self.state, reward, done, self.f_x, self.f_y = self.__get_observation()
        
        self.num_steps += 1

        return self.state, reward, done, {}

    def render(self):
        print('fwhm x: ', self.f_x)
        print('fwhm y: ', self.f_y)
        print('state: ', self.state)



def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, models, optimizers,  num_states, num_actions, size, hyperparams):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, num_states), dtype=np.float32
        )
        self.action_buffer = np.zeros((size, num_actions), dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros((size,num_actions), dtype=np.float32)
        self.gamma, self.lam, self.clip_ratio = hyperparams
        self.pointer, self.trajectory_start_index = 0, 0
        self.num_actions = num_actions
        self.num_states = num_states
        self.actor = models[0]
        self.critic = models[1]
        self.policy_optimizer = optimizers[0]
        self.value_optimizer = optimizers[1]

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )

    def logprobabilities(self,logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        #print(logprobabilities_all)
        #print(a, self.num_actions)
        #print( tf.one_hot(a, self.num_actions))
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.num_actions) * logprobabilities_all, axis=1
        )
        return logprobability

    # Sample action from actor
    @tf.function
    def sample_action(self,observation):
        logits = self.actor(observation)
        action = tf.squeeze(logits)#tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action

    # Train the policy by maxizing the PPO-Clip objective
    @tf.function
    def train_policy(self):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self.logprobabilities(self.actor(self.observation_buffer), self.action_buffer)
                - self.logprobability_buffer
            )
            min_advantage = tf.where(
                self.advantage_buffer > 0,
                (1 + self.clip_ratio) * self.advantage_buffer,
                (1 - self.clip_ratio) * self.advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * self.advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        kl = tf.reduce_mean(
            self.logprobability_buffer
            - self.logprobabilities(self.actor(self.observation_buffer), self.action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl

    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((self.return_buffer - self.critic(self.observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))

    def save_weights(self, model_dir):
        self.actor.save_weights(os.path.join(model_dir,'am_weights.h5'))
        self.critic.save_weights(os.path.join(model_dir,'cr_weights.h5'))
        print('Weights saved at: ', model_dir)



def mlp(x, sizes, activation=tf.tanh, output_activation=None):
        # Build a feedforward neural network
        for size in sizes[:-1]:
            x = layers.Dense(units=size, activation=activation)(x)
        return layers.Dense(units=sizes[-1], activation=output_activation)(x)

def train(env, model_dir, num_states, num_actions):
    # Hyperparameters of the PPO algorithm
    steps_per_epoch = 8#4000
    epochs = 5#30
    gamma = 0.99
    clip_ratio = 0.2
    policy_learning_rate = 3e-4
    value_function_learning_rate = 1e-3
    train_policy_iterations = 80
    train_value_iterations = 80
    lam = 0.97
    target_kl = 0.01
    hidden_sizes = (125, 125)

    # True if you want to render the environment
    render = False


    # Initialize the actor and the critic as keras models
    observation_input = keras.Input(shape=(num_states,), dtype=tf.float32)
    logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None) * env.bounds
    actor = keras.Model(inputs=observation_input, outputs=logits)
    value = tf.squeeze(
        mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
    )
    critic = keras.Model(inputs=observation_input, outputs=value)

    actor.summary()
    critic.summary()

    # Initialize the policy and the value function optimizers
    policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
    value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

    # Initialize the observation, episode return and episode length
    episode_return, episode_length =  0, 0

    # Initialize the buffer
    buffer = Buffer( [actor, critic], [policy_optimizer, value_optimizer], num_states, num_actions, steps_per_epoch, [gamma, lam, clip_ratio])

    # Iterate over the number of epochs
    for epoch in range(epochs):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        observation  = env.reset()
        # Iterate over the steps of each epoch
        for t in range(steps_per_epoch):
            if render:
                env.render()

            # Get the logits, action, and take one step in the environment
            observation = observation.reshape(1, -1)
            #print(observation)
            logits, action = buffer.sample_action(observation)
            legal_action = np.clip(action.numpy(),-env.bounds,env.bounds)
            observation_new, reward, done, _ = env.step(legal_action)
            print('reward: {0:.4f}, action: {1:.6f}, {2:.6f}, {3:.6f}, {4:.2f}, {5:.2f}'.format(reward,
                                                                                                action[0],
                                                                                                action[1],
                                                                                                action[2],
                                                                                                action[3],
                                                                                                action[4]))
            episode_return += reward
            episode_length += 1

            # Get the value and log-probability of the action
            value_t = critic(observation)
            logprobability_t = buffer.logprobabilities(logits, legal_action)

            # Store obs, act, rew, v_t, logp_pi_t
            buffer.store(observation, legal_action, reward, value_t, logprobability_t)

            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal or (t == steps_per_epoch - 1):
                last_value = 0 if done else critic(observation.reshape(1, -1))
                buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                observation, episode_return, episode_length = env.reset(), 0, 0

        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()

        print(action_buffer)
        # Update the policy and implement early stopping using KL divergence
        for _ in range(train_policy_iterations):
            kl = buffer.train_policy()
            if kl > 1.5 * target_kl:
                # Early Stopping
                break

        # Update the value function
        for _ in range(train_value_iterations):
            buffer.train_value_function()

        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )

    buffer.save_weights(model_dir)

   
def main():
    n=1
    model_dir = 'ppo_models_{}'.format(n)
    created_dir = True
    while not created_dir:
            try:
                os.mkdir(model_dir)
                created_dir = True
            except OSError:
                created_dir = False
                n += 1
                model_dir = 'ppo_models_{}'.format(n)


    #beamline = VeritasSimpleBeamline()
    env = RaycingEnv()
    #bounds = np.array([beamline.p_lim, beamline.y_lim, beamline.r_lim, beamline.l_lim, beamline.v_lim])
    train( env, model_dir, num_states=4, num_actions=5)
    
    print('DONE :)')

    
if __name__ == '__main__':
    main()
