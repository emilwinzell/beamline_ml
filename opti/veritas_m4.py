"""
Script for running hypermapper (Bayesian optimization)
json file needs to be created first!

the function raycing_fcn is the objective function

Emil Winzell, May 2022

"""

import os
import sys
#sys.stdout = open('hypermapper_output.txt','wt')


import xrt.backends.raycing.run as rr
import xrt.plotter as xrtp
import xrt.runner as xrtr

import numpy as np
from scipy import optimize
from hypermapper import optimizer

from rl.vsb import VeritasSimpleBeamline
stdout = sys.stdout


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

def calculate_argmin(data,xlim,N):
    x = np.linspace(xlim[0], xlim[-1], N)

    def poly(x, a0,a1,a2):
        return a0 + a1 *x + a2*x**2

    [a0,a1,a2],_ = optimize.curve_fit(poly, xlim, data)

    if a2 > 0:
        amin = -a1/(2*a2)
    else:
        amin = 1000.0
    
    return amin

def calculate_vad(data,xlim):
    # vertical angle displacement: checking the V-shape of the veritcal fwhms
    # not used in project

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

def raycing_fcn(X):

    p = X['pitch']
    y = X['yaw']
    r = X['roll']
    l = X['lateral']
    v = X['vertical']

    for plot in beamline.plots:
        plot.xaxis.limits = None
        plot.yaxis.limits = None
        plot.xaxis.binEdges = np.zeros(beamline.bins + 1)
        plot.xaxis.total1D = np.zeros(beamline.bins)
        plot.yaxis.binEdges = np.zeros(beamline.bins + 1)
        plot.yaxis.total1D = np.zeros(beamline.bins)

    params = np.array([p,y,r,l,v])
    beamline.update_m4(params)

    #print('Running xrt...')
    #sys.stdout = open(os.devnull, 'w')
    xrtr.run_ray_tracing(beamline.plots,repeats=beamline.repeats, 
                updateEvery=beamline.repeats, beamLine=beamline)#,threads=3,processes=8)
    #sys.stdout = stdout#sys.__stdout__#open('ddpg_output.txt','a')#

    f_x = []
    f_y = []
    for i,plot in enumerate(beamline.plots):
        xt1D = np.append(plot.xaxis.total1D,0.0)
        xBins = plot.xaxis.binEdges
        yt1D = np.append(plot.yaxis.total1D,0.0)
        yBins = plot.yaxis.binEdges
        if i == 4:
            t_dist = np.sqrt(np.square(plot.cx)+np.square(plot.cy))/5
        if plot.dx == 0: 
            f_x.append(50.0)
        else:
            f_x.append(calculate_fwhm(xt1D,xBins,1000))
        if plot.dy == 0:
            f_y.append(50.0)
        else:
            f_y.append(calculate_fwhm(yt1D,yBins,1000))
        
    xmin = calculate_argmin(f_x, beamline.scrTgt11.dqs, 1000)/14
    #vad = calculate_vad(f_y, beamline.scrTgt11.dqs)
    ymin = beamline.scrTgt11.dqs[np.argmin(f_y)]/14
    #ymin = calculate_argmin(f_y, beamline.scrTgt11.dqs, 1000)
    gap = np.clip((xmin-ymin)/1000.0,-1.0,1.0)
    FWHMx = min(f_x)/0.12
    FWHMy = min(f_y)/0.05
    #if gap == 0.0:
    #    gap = 1000.0
    #gap = gap/1000.0 # normalize
    if abs(gap) < 3/1000.0 and FWHMx < 0.012 and FWHMy < 0.006:
        print("BEAM ALIGNED ")
        print("")
        
    output = {}
    #output['FWHM'] = FWHMx + FWHMy
    #output['t_dist'] = t_dist
    output['value'] = FWHMx + FWHMy + t_dist#vad/100.0
    #output['Gap'] = gap
    #output["value"] = FWHMx + FWHMy + gap

    return output



    
def main():
    global beamline, filename
    beamline = VeritasSimpleBeamline(nrays=10000)
    filename = "opti\\veritas_raycing_m4_scenario.json"

    rr.run_process = beamline.run_process

    org_params = [beamline.M4pitch, 
                    beamline.M4yaw,
                    beamline.M4roll,
                    beamline.lateral,
                    beamline.vertical]

    optimizer.optimize(filename, raycing_fcn)
    sys.stdout = stdout

    print('original: ',org_params)

    best_X = {}
    best_X['pitch'] = -org_params[0]
    best_X['yaw'] = -org_params[1]
    best_X['roll'] = -org_params[2]
    best_X['lateral'] = -org_params[3]
    best_X['vertical'] = -org_params[4]

    output = raycing_fcn(best_X)
    print("best possible value:  ", output['value'])#output['FWHM'], output['t_dist'])
    
    print("difference: []")

    print('DONE :)')
    
    
if __name__ == '__main__':
    main()
