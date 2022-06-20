#
# Inital RL model from: https://keras.io/examples/rl/actor_critic_cartpole/
# Actor critic, will maybe expand to DDPG 
#
#from secrets import choice
import os
import sys
sys.stdout = open('acm_output.txt','wt')

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

from vsb import VeritasSimpleBeamline


# Environment class
#
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
        self.steps=[1e-5, 2e-4, 2e-4, 0.1 ,0.1]
        self.params=np.array([0.0,0.0,0.0,0.0,0.0])
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


    def __take_action(self):
        for plot in self.beamline.plots:
                plot.xaxis.limits = None
                plot.yaxis.limits = None
                plot.xaxis.binEdges = np.zeros(self.beamline.bins + 1)
                plot.xaxis.total1D = np.zeros(self.beamline.bins)
                plot.yaxis.binEdges = np.zeros(self.beamline.bins + 1)
                plot.yaxis.total1D = np.zeros(self.beamline.bins)

        # change bealine params
        action = np.clip(self.params, -self.bounds, self.bounds)
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
        
        self.params[action[0]] += action[1]*self.steps[action[0]]
        self.__take_action()

        self.state, reward, done, self.f_x, self.f_y = self.__get_observation()
        
        self.num_steps += 1

        return self.state,self.params, reward, done, {}

    def render(self):
        print('fwhm x: ', self.f_x)
        print('fwhm y: ', self.f_y)
        print('state: ', self.state)
#


def get_action(choice):
    acs = [0,0,1,1,2,2,3,3,4,4]
    return (acs[choice],2*(choice%2)-1)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = open('acm_output.txt','a')#sys.__stdout__#


def train(beamline,env,model,num_actions):
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    huber_loss = keras.losses.Huber()
    action_probs_history = []
    critic_value_history = []
    rewards_history = []
    running_reward = 0
    episode_count = 0

    total_episodes = 15
    max_steps_per_episode = 700
    gamma = 0.99 # Discount factor for past rewards
    eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


    
    while True:  # Run until solved
        state  = env.reset()
        episode_reward = 0
        no_imp_cnt = 0
        with tf.GradientTape() as tape:
            for timestep in range(1, max_steps_per_episode):
                # env.render(); Adding this line would show the attempts
                # of the agent in a pop up window.

                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)

                # Predict action probabilities and estimated future rewards
                # from environment state
                action_probs, critic_value = model(state)
                critic_value_history.append(critic_value[0, 0])

                # Sample action from action probability distribution
                choice = np.random.choice(num_actions, p=np.squeeze(action_probs))
                action_probs_history.append(tf.math.log(action_probs[0, choice]))
                action = get_action(choice)
                state,action, reward, done, _ = env.step(action)

                print('reward: {0:.4f}, action: {1:.6f}, {2:.6f}, {3:.6f}, {4:.2f}, {5:.2f}'.format(reward,
                                                                                                action[0],
                                                                                                action[1],
                                                                                                action[2],
                                                                                                action[3],
                                                                                                action[4]))
                rewards_history.append(reward)
                episode_reward += reward

                if done:
                    break


            print('\n Episode: ', episode_count)

            episode_reward = (max_steps_per_episode-timestep)*episode_reward
            # Update running reward to check condition for solving
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            print(episode_reward)
            print(running_reward)
            print(timestep)
            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            # Calculating loss values to update our network
            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up recieving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            # Backpropagation
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Clear the loss and reward history
            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()

        # Log details
        episode_count += 1
        if episode_count % 2 == 0:
            template = "running reward: {:.2f} at episode {}"
            print(template.format(running_reward, episode_count))

        #if running_reward > max_steps_per_episode:  # Condition to consider the task solved
        #    print("Solved at episode {}!".format(episode_count))
        #    break

        if episode_count == total_episodes:
            #if input('end?, y-yes') == 'y':
            return model
    
    
def main():
    beamline = VeritasSimpleBeamline(nrays=10000)
    env = RaycingEnv()

    num_inputs = 3
    num_actions = 10
    num_hidden = 256

    inputs = layers.Input(shape=(num_inputs,))
    common = layers.Dense(num_hidden, activation="relu")(inputs)
    action = layers.Dense(num_actions, activation="softmax")(common)
    critic = layers.Dense(1)(common)

    model = keras.Model(inputs=inputs, outputs=[action, critic])
    model.summary()

    n=1
    model_dir = 'acm_models_{}'.format(n)
    created_dir = False
    while not created_dir:
            try:
                os.mkdir(model_dir)
                created_dir = True
            except OSError:
                created_dir = False
                n += 1
                model_dir = 'acm_models_{}'.format(n)


    model = train(beamline, env,model,num_actions)
    model.save_weights(os.path.join(model_dir,'ac_weights.h5'))

    print('DONE :)')

    

if __name__ == '__main__':
    main()
