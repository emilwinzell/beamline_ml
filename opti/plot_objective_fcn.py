"""
Plotting the objective function for some parameters

python .\plot_objective_fcn.py -p 'C:\Users\emiwin\exjobb\05100916'


"""



import os
import sys
#sys.stdout = open('hypermapper_output.txt','wt')

import xrt.backends.raycing.run as rr
import xrt.plotter as xrtp
import xrt.runner as xrtr

import numpy as np
import argparse
import glob
import pandas as pd
from scipy import optimize
from matplotlib import pyplot as plt

def calculate_fwhm(data,lim,N):
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

def fit_poly(data,xlim,N):
    x = np.linspace(xlim[0], xlim[-1], N)

    def poly(x, a0,a1,a2):
        return a0 + a1 *x + a2*x**2

    [a0,a1,a2],_ = optimize.curve_fit(poly, xlim, data)
    
    if a2 > 0:
        amin = -a1/(2*a2)
    else:
        amin = 1000.0
        #amin = x[np.argmin(poly(x, a0,a1,a2))]
    
    return amin

def calculate_vad(data,xlim):
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

def plot_histograms(list_of_hist):
    
    n = 1
    xFWHMs = []
    yFWHMs = []
    values = []
    dqs = np.linspace(-14, 14, 9)
    for hist in list_of_hist:
        df = pd.read_csv(hist)
        xt1D = df["xt1D"].to_numpy()
        xBins = df["xbinEdges"].to_numpy()
        yt1D = df["yt1D"].to_numpy()
        yBins = df["ybinEdges"].to_numpy()

        if sum(xt1D) == 0: 
            x_fwhm = 50.0
        else:
            x_fwhm = calculate_fwhm(xt1D,xBins,1000)
        if sum(yt1D) == 0:
            y_fwhm = 50.0
        else:
            y_fwhm = calculate_fwhm(yt1D,yBins,1000)

        if n == 5:
                t_dist = np.sqrt(np.square(xBins[len(xBins)//2])+np.square(yBins[len(yBins)//2]))/5
       
        xFWHMs.append(x_fwhm)
        yFWHMs.append(y_fwhm)
        if n==9:
    
            dx0 = fit_poly(xFWHMs,dqs,1000)
            dy0 = dqs[np.argmin(yFWHMs)]
            vad = calculate_vad(yFWHMs,dqs)/100.0
            
            gap = np.clip((dx0-dy0)/1.0,-1.0,1.0)
            #if gap == 0.0:
            #    gap = 1000.0
            #gap= np.clip(gap/1000.0,-1.0,1.0)
            FWHMx = min(xFWHMs)/0.12

            FWHMy = min(yFWHMs)/0.05
            n = 1
            xFWHMs = []
            yFWHMs = []
            tot = FWHMx + FWHMy + t_dist# +vad
            values.append((FWHMx,FWHMy,t_dist,tot))
        else:
            n+=1
    return values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--base", help="path to base directory (timestamp)")
    args = parser.parse_args()
    #images = os.path.join(args.base,'images')
    histograms = os.path.join(args.base,'histograms')
    names = ['pitch','yaw','roll','lateral transl.','vertical transl.','roll2']
    stats = ['fwhm_x','fwhm_y','target dist','total']
    num_samples = 30
    pitch_lim = 0.003
    yaw_lim = 0.001
    roll_lim = 0.001
    lat_lim = 5.0
    vert_lim = 2.5

    pitches = np.linspace(-pitch_lim,pitch_lim,num_samples)
    yaws = np.linspace(-yaw_lim,yaw_lim,num_samples)
    rolls = np.linspace(-roll_lim,roll_lim,num_samples)
    lat = np.linspace(-lat_lim,lat_lim,num_samples) 
    vert = np.linspace(-vert_lim,vert_lim,num_samples) 
    input_list = [pitches, yaws, rolls, lat, vert, rolls]

    num_samples = 30
    for i in range(6):
        list_of_hist = glob.glob(histograms + '\\*.csv')
        values = plot_histograms(list_of_hist[(i*30)*9:((i+1)*30)*9])
        #print(values)
        plt.figure()
        plt.rcParams.update({'font.size': 15})
        plt.plot(input_list[i],values)
        plt.title(names[i])
        plt.legend(stats)
        plt.xlabel('Range')
        

        
    plt.show()



if __name__ == '__main__':
    main()
