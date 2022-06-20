"""
Plot rewards and actions from RL model run,
input txt file with following for all actions
reward: -1.1922, action: -0.000011, -0.000013, 0.000009, -0.32, 0.40
....

calcualtes average reward

ex:
python .\rl\plot_rew_act.py -p 'C:\Users\emiwin\exjobb\beamline\rl'

Emil Winzell, May 2022
"""


import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--base", help="path to base directory (timestamp)")
    args = parser.parse_args()
    filename = os.path.join(args.base,'ddpg_output3.txt')
    rew = []
    act = []
    avg_reward = []
    tot_rews = []
    no_eps = 0
    with open(filename) as file:
        for line in file:
            line = line.rstrip()
            l_splt = line.split(':')
            if l_splt[0] == 'Set params':
                org = np.array(line.split(' ')[-5:], dtype=np.float32)
            if not l_splt[0] == 'reward':
                no_steps = len(rew)
                if not no_steps < 10:
                    tot_rews.append(sum(rew))
                    avg_reward.append(np.mean(tot_rews))#print(sum(rew))
                    no_eps += 1
                    act = np.array(act)
                    #print('Best action: {}, with reward={}'.format(act[np.argmax(rew)], np.max(rew)))
                    print(list( org + act[np.argmax(rew)]))
                    if no_eps == 32:
                        plt.figure()
                        plt.title('reward')
                        plt.plot(range(no_steps),rew)
                        plt.figure()
                        plt.title('rot')
                        plt.plot(range(no_steps),act[:, :3])
                        plt.legend(['pitch','yaw','roll'])
                        plt.figure()
                        plt.title('transl')
                        plt.plot(range(no_steps),act[:, 3:])
                        plt.legend(['lateral','vertical'])
                        plt.show()
                    rew = []
                    act = []
                continue
            rew.append( float(l_splt[1].split(',')[0]))
            act.append(np.array( l_splt[-1].split(','), dtype=np.float32 ))
    
    #plt.figure()
    ax = plt.figure().gca()
    plt.plot(range(no_eps), avg_reward)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Average rewards')
    plt.show()

    

                




if __name__ == '__main__':
    main()