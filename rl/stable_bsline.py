import os
import sys
#sys.stdout = open('ddpg_output.txt','wt')

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
    
    def __init__(self):
        super(RaycingEnv, self).__init__()
        self.beamline = VeritasSimpleBeamline() # vsb object
        rr.run_process = self.beamline.run_process

        self.bounds = np.array([self.beamline.p_lim, self.beamline.y_lim, self.beamline.r_lim, self.beamline.l_lim, self.beamline.v_lim])
        
        self.action_space = spaces.Box(-self.bounds, self.bounds, dtype=np.float32)
        self.observation_space = spaces.Box(np.array([0.0, 0.0, -10.0]),
                                            np.array([50.0, 50.0, 10.0]), dtype=np.float32)
        
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
        ymin = self.__calculate_argmin(f_y, self.beamline.scrTgt11.dqs, 1000)
        gap = xmin-ymin
        FWHMx = min(f_x)
        FWHMy = min(f_y)
        if gap == 0.0:
            gap = 1000.0
        gap = np.clip(gap/1000.0,-10.0,10.0) # normalize
        
        state =  [FWHMx,FWHMy,gap]

        reward = -FWHMx - FWHMy - abs(gap)

        # Done?
        done = False
        if gap < 1.5/1000.0 and FWHMx < 0.02 and FWHMy < 0.02:
            done = True
            reward = 300 # max possible steps is 290
        
        return state, reward, done, f_x, f_y


    def __take_action(self, sampled_actions):
        for plot in self.beamline.plots:
                plot.xaxis.limits = None
                plot.yaxis.limits = None

        # change bealine params
        action = np.clip(sampled_actions, -self.bounds, self.bounds)
        self.beamline.update_m4(action)

        print('taking action..')
        sys.stdout = open(os.devnull, 'w')
        xrtr.run_ray_tracing(self.beamline.plots,repeats=self.beamline.repeats, 
                    updateEvery=self.beamline.repeats, beamLine=self.beamline)#,threads=3,processes=8)
        sys.stdout = sys.__stdout__

    def reset(self):
        #self.beamline = VeritasSimpleBeamline()
        self.beamline = VeritasSimpleBeamline() # vsb object
        rr.run_process = self.beamline.run_process
        
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



def main():
    env = RaycingEnv()


    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10) # what does this do?


    obs = env.reset()
    for i in range(5):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done,info  = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()

if __name__ == '__main__':
    main()