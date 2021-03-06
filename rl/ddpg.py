"""
 RL model from: https://keras.io/examples/rl/ddpg_pendulum/

Deep Deterministic Policy Gradient
Emil Winzell, April 2022
"""
import os
import sys
sys.stdout = open('ddpg_output2.txt','wt')

import xrt.backends.raycing.run as rr
import xrt.plotter as xrtp
import xrt.runner as xrtr


import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.regularizers import l2
#import gym
import scipy.signal
import time
from scipy import optimize
import pandas as pd

from vsb import VeritasSimpleBeamline

# Environment class
#
class RaycingEnv():
    """
    ### RaycingEnv

    ### Action Space
    The action is a `ndarray` with shape `(5,)` representing the ways to shift the M4 mirror
    | Num | Action   | Min    | Max   |
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

        #self.bounds = np.array([self.beamline.p_lim, self.beamline.y_lim, self.beamline.r_lim, self.beamline.l_lim, self.beamline.v_lim])
        self.bounds = np.array([3,1,1,1.5,2.5])
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

    def __get_observation(self):
        f_x = []
        f_y = []
        for i,plot in enumerate(self.beamline.plots):
            xt1D = np.append(plot.xaxis.total1D,0.0)
            xBins = plot.xaxis.binEdges
            yt1D = np.append(plot.yaxis.total1D,0.0)
            yBins = plot.yaxis.binEdges

            if i == 4:
                t_dist = np.sqrt(np.square(plot.cx)+np.square(plot.cy))/30

            if plot.dx == 0: 
                f_x.append(50.0)
            else:
                f_x.append(self.__calculate_fwhm(xt1D,xBins,1000))
            if plot.dy == 0:
                f_y.append(50.0)
            else:
                f_y.append(self.__calculate_fwhm(yt1D,yBins,1000))
            
        xmin = self.__calculate_argmin(f_x, self.beamline.scrTgt11.dqs, 1000)
        ymin = self.beamline.scrTgt11.dqs[np.argmin(f_y)]#self.__calculate_argmin(f_y, self.beamline.scrTgt11.dqs, 1000)
        gap = xmin-ymin
        FWHMx = min(f_x)
        FWHMy = min(f_y)*2
        #if gap == 0.0:
        #    gap = 1000.0
        gap = np.clip(gap/1000.0,-1.0,1.0) # normalize
        
        state =  np.array([FWHMx,FWHMy,gap])

        reward = -FWHMx - FWHMy - abs(gap)- t_dist

        # Done?
        done = False
        if abs(gap) < 3/1000.0 and FWHMx < 0.012 and FWHMy < 0.008:
            done = True
            reward = 1000 # max possible steps is 290
        
        return state, reward, done, f_x, f_y


    def __take_action(self, sampled_actions):
        for plot in self.beamline.plots:
                plot.xaxis.limits = None
                plot.yaxis.limits = None
                plot.xaxis.binEdges = np.zeros(self.beamline.bins + 1)
                plot.xaxis.total1D = np.zeros(self.beamline.bins)
                plot.yaxis.binEdges = np.zeros(self.beamline.bins + 1)
                plot.yaxis.total1D = np.zeros(self.beamline.bins)

        # change beamline params
        action = np.clip(sampled_actions, -self.bounds, self.bounds)
        self.beamline.update_m4(action)

        
        xrtr.run_ray_tracing(self.beamline.plots,repeats=self.beamline.repeats, 
                    updateEvery=self.beamline.repeats, beamLine=self.beamline,threads=3,processes=12)
        

    def reset(self):
        #self.beamline = VeritasSimpleBeamline()
        self.beamline.reset()
        print('Set params: ', self.beamline.M4pitch, self.beamline.M4yaw, self.beamline.M4roll, self.beamline.lateral, self.beamline.vertical)
        self.__take_action(np.zeros(5,dtype=np.float32))
        
        self.num_steps = 0
        self.state,_,_,self.f_x,self.f_y = self.__get_observation()  # calculate state
        return np.array(self.f_x + self.f_y)

    def step(self, action):
        action[:3] = action[:3]*0.001
        sys.stdout = open(os.devnull, 'w')
        self.__take_action(action)
        sys.stdout = open('ddpg_output2.txt','a')#sys.__stdout__

        self.state, reward, done, self.f_x, self.f_y = self.__get_observation()
        
        self.num_steps += 1

        return np.array(self.f_x + self.f_y), reward, done, {}

    def render(self):
        print('fwhm x: ', self.f_x)
        print('fwhm y: ', self.f_y)
        print('state: ', self.state)


# To implement better exploration by the Actor network, we use noisy perturbations, 
# specifically an Ornstein-Uhlenbeck process for generating noise, as described in 
# the paper. It samples noise from a correlated normal distribution
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class NoiseProcess:
    def __init__(self, mean, std_deviation):
        self.mean = mean
        self.init_std = std_deviation
        self.steps = 0

    def __call__(self):
        std = np.exp(-0.005*self.steps)*self.init_std
        x = np.random.normal(self.mean, std,size=self.mean.shape)
        self.steps += 1
        return x

    def reset(self):
        self.steps = 0

class Buffer:
    def __init__(self, num_states, num_actions, models, optimizers, buffer_capacity=100000, batch_size=64):
        # Discount factor for future rewards
        self.gamma = 0.99
        # Used to update target networks
        self.tau = 0.005

        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

        self.actor_model = models[0]
        self.critic_model = models[1]
        self.target_actor = models[2]
        self.target_critic = models[3]

        self.actor_optimizer = optimizers[0]
        self.critic_optimizer = optimizers[1]

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)
        self.update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        self.update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self,target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def policy(self, state, noise_object, bounds):
        sampled_actions = tf.squeeze(self.actor_model(state))
        #print('sampled actions: ', sampled_actions)
        noise = noise_object()
        #print(noise)
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, -bounds, bounds)

        return np.squeeze(legal_action)

    def save_weights(self, model_dir):
        self.actor_model.save_weights(os.path.join(model_dir,'am_weights2.h5'))
        self.critic_model.save_weights(os.path.join(model_dir,'cr_weights2.h5'))
        self.target_actor.save_weights(os.path.join(model_dir,'ta_weights2.h5'))
        self.target_critic.save_weights(os.path.join(model_dir,'tc_weights2.h5'))
        print('Weights saved at: ', model_dir)


def get_actor(num_states, num_actions, bounds):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="tanh",  kernel_regularizer=l2(0.01), kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * bounds
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(num_states,num_actions):
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def train(env, model_dir, num_states, num_actions):
    std_dev = np.array([0.001, 0.0005, 0.0005, 1 ,0.5])
    ou_noise = NoiseProcess(mean=np.zeros(5), std_deviation=std_dev)#OUActionNoise(mean=np.zeros(5), std_deviation=std_dev)

    actor_model = get_actor(num_states,num_actions,env.bounds)
    critic_model = get_critic(num_states,num_actions)

    target_actor = get_actor(num_states,num_actions,env.bounds)
    target_critic = get_critic(num_states,num_actions)

    # Making the weights equal initially
    #target_actor.set_weights(actor_model.get_weights())
    #target_critic.set_weights(critic_model.get_weights())
    actor_model.load_weights(os.path.join(model_dir,'am_weights.h5'))
    critic_model.load_weights(os.path.join(model_dir,'cr_weights.h5'))
    target_actor.load_weights(os.path.join(model_dir,'ta_weights.h5'))
    target_critic.load_weights(os.path.join(model_dir,'tc_weights.h5'))

    # Learning rate for actor-critic models
    critic_lr = 0.002
    actor_lr = 0.001

    optimizers = [tf.keras.optimizers.Adam(actor_lr,clipnorm=0.1), tf.keras.optimizers.Adam(critic_lr, clipnorm=0.1)]
    models = [actor_model, critic_model, target_actor, target_critic]

    total_episodes = 20
    max_steps_per_episode = 1000
    
    buffer = Buffer(num_states, num_actions, models, optimizers, buffer_capacity=100000, batch_size=64)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []


    # Takes about 4 min to train
    for ep in range(total_episodes):

        prev_state = env.reset()
        ou_noise.reset()
        episodic_reward = 0
        step_count = 0
        while True:

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = buffer.policy(tf_prev_state, ou_noise, env.bounds)

            # Recieve state and reward from environment.
            state, reward, done,_ = env.step(action)
            step_count += 1
            #print('new state: ', state)
            print('reward: {0:.4f}, action: {1:.6f}, {2:.6f}, {3:.6f}, {4:.2f}, {5:.2f}'.format(reward,
                                                                                                action[0],
                                                                                                action[1],
                                                                                                action[2],
                                                                                                action[3],
                                                                                                action[4]))

            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward

            buffer.learn()
            
            # End this episode when `done` is True
            if done:
                break
            
            if step_count == max_steps_per_episode:
                print('Maximum steps reached wihtout solution, ending...')
                break
            prev_state = state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Reward: {}, Avg Reward is ==> {}".format(ep, episodic_reward, avg_reward))
        avg_reward_list.append(avg_reward)

    #am = os.path.join(models,'actor_model')
    #actor_model.save(am)
    buffer.save_weights(model_dir)
    print(avg_reward_list)
    # Plotting graph
    # Episodes versus Avg. Rewards
    # plt.plot(avg_reward_list)
    # plt.xlabel("Episode")
    # plt.ylabel("Avg. Epsiodic Reward")
    # plt.show()
    
    
def main():
    n=1
    model_dir = 'ddpg_models_1'.format(n)
    created_dir = True
    while not created_dir:
            try:
                os.mkdir(model_dir)
                created_dir = True
            except OSError:
                created_dir = False
                n += 1
                model_dir = 'ddpg_models_{}'.format(n)


    env = RaycingEnv()

    train(env, model_dir, num_states=18, num_actions=5)
    
    print('DONE :)')

    
if __name__ == '__main__':
    main()
