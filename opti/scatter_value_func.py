"""
Make scatter plot of euc. distance and objective fcn

ex:
python .\scatter_value_func.py -p 'C:\Users\emiwin\exjobb\05121844'

Emil Winzell May 2022
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import xml.etree.ElementTree as ET
import argparse
import glob
import pandas as pd
import sys

from scipy import optimize

def calculate_fwhm(data,lim,N):
    x = np.linspace(lim[0], lim[-1], N)
    n = np.count_nonzero(data==0)//2+1
    est_std = (lim[-n]-lim[n])/5.0
    est_mean = lim[len(lim)//2]
    est_amp = np.max(data)*2.5
    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) / np.sqrt(2) / stddev)**2)
    
    sys.stdout = open(os.devnull, 'w')
    [amp,mean,std],_ = optimize.curve_fit(gaussian, lim, data,p0=[est_amp,est_mean,est_std])
    sys.stdout = sys.__stdout__
    std = abs(std)
    fwhm = 2*np.sqrt(2*np.log(2))*std

    return fwhm

def calculate_argmin(data,xlim,N):
    x = np.linspace(xlim[0], xlim[-1], N)

    def poly(x, a0,a1,a2):
        return a0 + a1 *x + a2*x**2

    sys.stdout = open(os.devnull, 'w')
    [a0,a1,a2],_ = optimize.curve_fit(poly, xlim, data)
    sys.stdout = sys.__stdout__

    if a2 > 0:
        amin = -a1/(2*a2)
    else:
        amin = 1000.0
    
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


def scratnum(path,root):
    list_of_hist = glob.glob(path + '\\*.csv')
    num_samples = len(list_of_hist)//9
    n = 1
    values = []
    dists = []
    dqs = np.linspace(-14, 14, 9)
    best_value = [100,0]
    theor_best = [100,0]
    for sample_nr in range(num_samples):
        f_x = []
        f_y = []
        for i in range(9):
            hist = list_of_hist[9*sample_nr+i]

            #sanity check
            name = os.path.split(hist)[1]
            if not sample_nr == int(name.split('_')[1]):
                print('Unmatched samples! ending...')
                return

            df = pd.read_csv(hist)
            xt1D = df["xt1D"].to_numpy()
            xBins = df["xbinEdges"].to_numpy()
            yt1D = df["zt1D"].to_numpy()
            yBins = df["zbinEdges"].to_numpy()

            f_x.append(calculate_fwhm(xt1D,xBins,1000))
            f_y.append(calculate_fwhm(yt1D,yBins,1000))

            if i == 4:
                t_dist = np.sqrt(np.square(xBins[len(xBins)//2])+np.square(yBins[len(yBins)//2]))/5

        xmin = abs(dqs[np.argmin(f_x)])/14#calculate_argmin(f_x, dqs, 1000)
        vad = calculate_vad(f_y, dqs)/100.0
        ymin = abs(dqs[np.argmin(f_y)])/14
        #gap = np.clip(abs(xmin-ymin),0.0,20.0)
        FWHMx = min(f_x)/0.12
        FWHMy = min(f_y)/0.05
        #value = FWHMx + FWHMy + xmin + ymin + t_dist
        value = FWHMx + FWHMy + t_dist
        #if sample_nr == 98 or sample_nr == 266:
        #    print(FWHMx,FWHMy,xmin,ymin,t_dist)
        #    input('')
        if value < best_value[0]:
            best_value = [value,sample_nr]
        values.append(value)

        #specs = 
        #print(specs)
        p = root.findall('sample_{}/specifications/pitch'.format(sample_nr))[0]
        y = root.findall('sample_{}/specifications/yaw'.format(sample_nr))[0]
        r = root.findall('sample_{}/specifications/roll'.format(sample_nr))[0]
        l = root.findall('sample_{}/specifications/x_transl'.format(sample_nr))[0]
        v = root.findall('sample_{}/specifications/z_transl'.format(sample_nr))[0]

        specs = np.array([float(p.text)/0.003,float(y.text)/0.001,float(r.text)/0.001,float(l.text)/1.5,float(v.text)/2.5])#
        #specs[:2] = specs[:2]*1000,float(r.text)/0.1,float(l.text)/5.0
        d = np.linalg.norm(specs)
        dists.append(d)
        if d < theor_best[0]:
            theor_best = [d,sample_nr]
        #    print('value:', value)
        #    print(specs)
            #input('')
    
    return np.array(values), np.array(dists), best_value, theor_best
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--base", help="path to base directory (timestamp)")
    args = parser.parse_args()
    images = os.path.join(args.base,'images')
    histograms = os.path.join(args.base,'histograms')
    labels = os.path.join(args.base,'data.xml')

    tree = ET.parse(labels)
    root = tree.getroot()

    values,dists,fbest,tbest = scratnum(histograms,root)
    print('the best value found: ', fbest)
    print('theoretical best: ', tbest)

    plt.figure()
    plt.rcParams.update({'font.size': 15})
    plt.scatter(dists,values)
    plt.scatter(dists[fbest[1]],values[fbest[1]])
    plt.scatter(dists[tbest[1]],values[tbest[1]])
    #for i in range(6):
    #    plt.scatter(dists[30*i:30*(i+1)],values[30*i:30*(i+1)])
    plt.ylabel('objective function')
    plt.xlabel('euc. distance to optimum')
    #plt.legend(['pitch','yaw','roll','lat','vert','roll2'])
    plt.show()



if __name__ == '__main__':
    main()