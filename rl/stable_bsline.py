"""
For running PPO with stable baselines

Emil Winzell, April 2022
"""


import os
import sys
sys.stdout = open('stb_output.txt','wt')

import xrt.backends.raycing.run as rr
import xrt.plotter as xrtp
import xrt.runner as xrtr


import numpy as np
import cv2 as cv
import gym
from gym import spaces
import scipy.signal
import time
from scipy import optimize

from stable_baselines3 import PPO
from stable_baselines3 import DQN

from vsb import VeritasSimpleBeamline

class RaycingEnv(gym.Env):
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
    
    def __init__(self,nrays):
        super(RaycingEnv, self).__init__()
        self.beamline = VeritasSimpleBeamline(nrays=nrays) # vsb object
        rr.run_process = self.beamline.run_process

        #self.bounds = np.array([self.beamline.p_lim, self.beamline.y_lim, self.beamline.r_lim, self.beamline.l_lim, self.beamline.v_lim])
        self.bounds = np.array([1,1,1,1,1])
        
        self.action_space = spaces.Box(-self.bounds, self.bounds, dtype=np.float32)
        self.observation_space = spaces.Box(np.zeros(18),
                                            np.ones(18), dtype=np.float32)
        
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

        if a2 < 0:
            amin = -a1/(2*a2)
        else:
            amin = 1000.0
        
        return amin
    
    def __calculate_vad(self,data,xlim):
        ub = np.max(data)
        lb = np.min(data)
        if ub-lb > 0:
            mapk = 28/(ub-lb)
        else:
            mapk = 1
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
        for n,plot in enumerate(self.beamline.plots):
            xt1D = np.append(plot.xaxis.total1D,0.0)
            xBins = plot.xaxis.binEdges
            yt1D = np.append(plot.yaxis.total1D,0.0)
            yBins = plot.yaxis.binEdges
            if n == 4:
                t_dist = np.sqrt(np.square(plot.cx)+np.square(plot.cy))/5

            if plot.dx == 0: 
                f_x.append(1.0)
            else:
                f_x.append(self.__calculate_fwhm(xt1D,xBins,1000))
            if plot.dy == 0:
                f_y.append(1.0)
            else:
                f_y.append(self.__calculate_fwhm(yt1D,yBins,1000))
        
        xmin = self.__calculate_argmin(f_x, self.beamline.scrTgt11.dqs, 1000)
        #vad = self.__calculate_vad(f_y, self.beamline.scrTgt11.dqs)
        ymin = self.beamline.scrTgt11.dqs[np.argmin(f_y)]
        gap = (xmin-ymin)/1000.0
        #vad = vad/100.0 
        FWHMx = min(f_x)/0.12
        FWHMy = min(f_y)/0.05
        
        state =  np.array([f_x+f_y])#np.array([FWHMx,FWHMy,gap,vad])

        reward = -FWHMx - FWHMy -t_dist #abs(np.clip(gap,-1.0,1.0)) #- vad

        # Done?
        done = False
        if gap < 3.0/1000.0 and FWHMx < 0.012 and FWHMy < 0.006: #and vad < 1.0/100.0:
            done = True
            reward = 300 # max possible steps is 290
        
        return state, reward, done#, f_x, f_y


    def __take_action(self, sampled_actions):
        for plot in self.beamline.plots:
                plot.xaxis.limits = None
                plot.yaxis.limits = None
                plot.xaxis.binEdges = np.zeros(self.beamline.bins + 1)
                plot.xaxis.total1D = np.zeros(self.beamline.bins)
                plot.yaxis.binEdges = np.zeros(self.beamline.bins + 1)
                plot.yaxis.total1D = np.zeros(self.beamline.bins)

        # change bealine params
        
        self.beamline.update_m4(sampled_actions)

        sys.stdout = open(os.devnull, 'w')
        xrtr.run_ray_tracing(self.beamline.plots,repeats=self.beamline.repeats, 
                    updateEvery=self.beamline.repeats, beamLine=self.beamline,threads=3,processes=9)
        sys.stdout = open('stb_output.txt','a')

    def reset(self):
        #self.beamline = VeritasSimpleBeamline()
        self.beamline.reset()
        print('Set params: ', self.beamline.M4pitch, self.beamline.M4yaw, self.beamline.M4roll, self.beamline.lateral, self.beamline.vertical)
        self.__take_action(np.zeros(5,dtype=np.float32))
        
        self.num_steps = 0
        self.state,_,_= self.__get_observation()  # calculate state
        return self.state#np.array(self.f_x + self.f_y)#

    def step(self, action):
        action = np.clip(action, -self.bounds, self.bounds)
        action = action*np.array([0.003,0.001,0.001,1.5,2.5]) #switch to correct unit
        #action[:3] = action[:3]*0.001
        self.__take_action(action)

        self.state, reward, done = self.__get_observation()
        print('reward: {0:.4f}, action: {1:.6f}, {2:.6f}, {3:.6f}, {4:.2f}, {5:.2f}'.format(reward,
                                                                                                action[0],
                                                                                                action[1],
                                                                                                action[2],
                                                                                                action[3],
                                                                                                action[4]))
        
        self.num_steps += 1

        return self.state, reward, done, {} #np.array(self.f_x + self.f_y)

    def render(self):
        #print('fwhm x: ', self.f_x)
        #print('fwhm y: ', self.f_y)
        #print('state: ', self.state)
        return



def main():
    env = RaycingEnv(nrays=10000)

    print('initalize')
    #model = DQN.load('/home/emiwin/exjobb/beamline/ppo_model_1/ppo_model1',env=env)
    model = PPO("MlpPolicy", env, n_steps=600, verbose=1)
    model.learn(total_timesteps=600*20) # what does this do?
    print('TRAINING DONE, testing now')
    obs = env.reset()
    for i in range(300):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done,info  = env.step(action)
        #env.render()
        if done:
            obs = env.reset()

    model.save("ppo_model2")

    env.close()

if __name__ == '__main__':
    main()
