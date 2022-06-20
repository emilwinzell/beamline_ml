"""
Input text file from hypermapper run
plots the actions taken

Emil Winzell, May 2022
"""


import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--base", help="path to base directory (timestamp)")
    args = parser.parse_args()
    filename = os.path.join(args.base,'hypermapper_run6.txt')
    rew1 = []
    rew2 = []
    act = []
    #pitch,yaw,roll,lateral,vertical,value,Timestamp
    with open(filename) as file:
        for line in file:
            line = line.rstrip()
            action = []

            if line.split(',')[-1].isnumeric():
                for setting in line.split(',')[:5]:
                    action.append(float(setting))
                rew1.append(float(line.split(',')[5]))
                rew2.append(float(line.split(',')[6]))
                act.append(action)
    
    print('best action: ', act[np.argmin(rew1)], ' at: ', np.argmin(rew1))
    print('best action: ', act[np.argmin(rew2)], ' at: ', np.argmin(rew2))
    plt.plot(range(len(rew1)),rew1)
    plt.plot(range(len(rew2)),rew2)
    plt.show()


    

                




if __name__ == '__main__':
    main()