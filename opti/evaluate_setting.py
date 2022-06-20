import os
import sys

import xrt.backends.raycing.run as rr
import xrt.plotter as xrtp
import xrt.runner as xrtr
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

from hypermapper import optimizer
from rl.vsb import VeritasSimpleBeamline



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

def raycing_fcn(params):

    for plot in beamline.plots:
        plot.xaxis.limits = None
        plot.yaxis.limits = None
        plot.xaxis.binEdges = np.zeros(beamline.bins + 1)
        plot.xaxis.total1D = np.zeros(beamline.bins)
        plot.yaxis.binEdges = np.zeros(beamline.bins + 1)
        plot.yaxis.total1D = np.zeros(beamline.bins)

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
            t_dist = np.sqrt(np.square(plot.cx)+np.square(plot.cy))
        if plot.dx == 0: 
            f_x.append(50.0)
        else:
            f_x.append(calculate_fwhm(xt1D,xBins,1000))
        if plot.dy == 0:
            f_y.append(50.0)
        else:
            f_y.append(calculate_fwhm(yt1D,yBins,1000))
        
    xmin = calculate_argmin(f_x, beamline.scrTgt11.dqs, 1000)
    vad = calculate_vad(f_y, beamline.scrTgt11.dqs)
    ymin = beamline.scrTgt11.dqs[np.argmin(f_y)]

    gap = abs(xmin-ymin)
    FWHMx = min(f_x)
    FWHMy = min(f_y)

    # plt.figure()
    # plt.scatter( beamline.scrTgt11.dqs,f_x)
    # plt.scatter(xmin,min(f_x))
    # plt.xlabel('z-axis (mm)')
    # plt.ylabel('FWHM (mm)')
    # plt.savefig('fx.png')

    # plt.figure()
    # plt.scatter( beamline.scrTgt11.dqs,f_y)
    # plt.scatter(ymin,min(f_y))
    # plt.xlabel('z-axis (mm)')
    # plt.ylabel('FWHM (mm)')
    # plt.savefig('fy.png')

    
    
    return FWHMx, FWHMy, gap, t_dist

# last six ddpg
# [[0.0037196635, -0.00034589646, -8.462707e-05, 0.89619255, 0.35088646],
#                     [0.005455827, -0.0017622586, 0.0014699681, 1.9081135, 1.156127],
#                     [0.0019169421, 0.00066247955, 0.0015433638, -0.2969372, -4.138736],
#                     [0.00027783634, 0.00014325976, -0.0004545544, -0.0068998933, 1.8906169],
#                     [-0.000837406, -0.00088172435, -0.0007594691, 1.0847819, -1.4287995],
#                     [0.00016913668, -0.00016953601, -0.001050105, -0.19333994, 3.1250937]]
#
# last six ddpg
# [[0.0023397177, -0.003055119937, 0.0016976409, 1.7059243, 4.293073],
# [-0.0005809651, -0.00021862908, 0.00033732352, -0.32539773, 1.4883322],
# [-0.00027387228, -0.0001668298, 0.0004715209, -0.98243386, 3.2985408],
# [0.0008721494, -0.0011158145, 0.00032014627, -0.092494935, 3.9696164],
# [0.0052288333, 0.00071609335, 8.7423716e-05, -0.9328451, 1.1787839],
# [0.00529212, 0.0018583582, 0.0005097508, -0.640535, 1.5332556]]
#     
# hypermapper
# setup 1
    # setting_list = [[-1.42237337e-04, -6.11207359e-06, -2.07993875e-04, -2.57236776e-01,2.33330280e-02],
    #                 [-2.97392991e-04,  8.83705631e-05,  1.30799623e-03, -3.03753079e-01,-3.62302806e-02],
    #                 [ 2.82404352e-04,  1.82926549e-04,  1.67707426e-03,  3.26236547e-01,-1.15008439e-01]]
#
# setup 2
# [[ 2.09790563e-04,  1.36073123e-05, -1.18229980e-03,  3.75313230e-01, 6.05866094e-02],
# [-1.74955449e-04, -1.88883454e-05,  9.84845433e-04, -1.97055024e-01, 1.41386219e-01],
# [-2.71978987e-04, -3.24367983e-05,  1.80543920e-03, -3.83937338e-01, -6.47162069e-02],
# [-4.49448414e-04,  5.32777824e-07,  1.65673408e-04, -5.38152291e-01,-3.42387094e-03]]
def main():
    global beamline
    beamline = VeritasSimpleBeamline(nrays=100000, m4_params=[0.0,0.0,0.0,0.0,0.0])
    rr.run_process = beamline.run_process
    lims = np.array([0.003,0.001,0.001,1.5,2.5])
    setting_list = [0.1*lims,
                    0.03*lims,
                    0.02*lims,
                    0.01*lims]
                    

    results = []
    for i,params in enumerate(setting_list):
        fx,fy,g,t = raycing_fcn(params)
        print("Setting {}: pitch:{}, yaw:{}, roll:{}, lat:{}, vert:{}".format(i,params[0],params[1],params[2],params[3],params[4]))
        print("FWHMx:{} ,FWHMy:{}, gap:{}, target dist.:{} ".format(fx,fy,g,t))
        if fx < 0.01 and fy < 0.005 and g < 2 and t < 0.3:
            print("BEAM IS WELL ALIGNED")
        else:
            print("NOT GOOD ENOUGH")
        results.append([fx,fy,g,t])
        # if i == 5:
        #     results = np.array(results)
        #     print("Average results (DDPG): ", np.mean(results,axis=0))
        #     results = []
        # if i == 11:
        #     results = np.array(results)
        #     print("Average results (DDPG 3): ", np.mean(results,axis=0))
        #     results = []
        # if i == 14:
        #     results = np.array(results)
        #     print("Average results (HM 1): ", np.mean(results,axis=0))
        #     results = []
        # if i == 18:
        #     results = np.array(results)
        #     print("Average results (HM 2): ", np.mean(results,axis=0))
        #     results = []
    results = np.array(results)
    print("Average results: ", np.mean(results,axis=0))


    
if __name__ == '__main__':
    main()
