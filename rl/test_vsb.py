import xrt.backends.raycing.run as rr
import xrt.plotter as xrtp
import xrt.runner as xrtr

import numpy as np
import cv2 as cv
from scipy import optimize

from vsb import VeritasSimpleBeamline

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

        return fwhm, gaussian(x,amp,mean,std)

def fit_poly(data,xlim,N):
    x = np.linspace(xlim[0], xlim[-1], N)

    def poly(x, a0,a1,a2):
        return a0 + a1 *x + a2*x**2
    
    [a0,a1,a2],_ = optimize.curve_fit(poly, xlim, data)
    return poly(x, a0,a1,a2),x

def calculate_argmin(data,xlim,N):
        x = np.linspace(xlim[0], xlim[-1], N)

        def poly(x, a0,a1,a2):
            return a0 + a1 *x + a2*x**2
    
        [a0,a1,a2],_ = optimize.curve_fit(poly, xlim, data)

        if a2 < 0:
            amin = -a1/(2*a2)
        else:
            amin = x[np.argmin(poly(x, a0,a1,a2))]
        
        return amin

def test_gen(beamline,plots):
    params = [0.0,0.0,0.0,0.0,0.0] #pitch, yaw, roll, lat(x), vert(y)
    steps = [1e-5, 2e-4, 2e-4, 0.005 ,0.005]
    while True:
        print('Updated params: ')
        print(beamline.M4.extraPitch)
        print(beamline.M4.extraYaw)
        print(beamline.M4.extraRoll)
        print(beamline.M4.center)
        yield
        xFWHMs = []
        yFWHMs = []
        for i,plot in enumerate(beamline.plots):
            print(plot.beam)
            t2D = plot.total2D_RGB
            if t2D.max() > 0:
                t2D = t2D*65535.0/t2D.max()
            t2D = np.uint16(cv.flip(t2D,0))

            x1td = np.append(plot.xaxis.total1D,0.0)
            xbins = plot.xaxis.binEdges
            y1td = np.append(plot.yaxis.total1D,0.0)
            ybins = plot.yaxis.binEdges

            x_fwhm,g_x = calculate_fwhm(x1td,xbins,1000)
            y_fwhm,g_y = calculate_fwhm(y1td,ybins,1000)
            xFWHMs.append(x_fwhm)
            yFWHMs.append(y_fwhm)
            if i == 8:
                xmin = calculate_argmin(xFWHMs, beamline.scrTgt11.dqs, 1000)
                ymin = calculate_argmin(yFWHMs, beamline.scrTgt11.dqs, 1000)
                gap = abs(xmin-ymin)
                FWHMx = min(xFWHMs)
                FWHMy = min(yFWHMs)
                print('gap: ', gap)
                print('fx: ', FWHMx)
                print('fy: ', FWHMy)
                if gap == 0.0:
                    print(FWHMx)
                    print(FWHMy)

            print(plot.dx)
            print(plot.dy)

            #print(x1td)
            #print(y1td)

            cv.imshow('plot', t2D)
            cv.waitKey(0)
            plot.xaxis.limits = None
            plot.yaxis.limits = None
        #pitch = 0.0005*np.random.randn()#np.random.uniform(-pitch_lim,pitch_lim)
        #yaw = 0.01*np.random.randn()#np.random.uniform(-yaw_lim,yaw_lim)
        #roll = 0.01*np.random.randn()#np.random.uniform(-roll_lim,roll_lim)
       

        # Set M4 mirror
        # while True:
        #     act = int(input('enter action: '))
        #     action = get_action(act)
        #     params[action[0]] += action[1]*steps[action[0]]
        print('pitch, yaw, roll, x, y')
        params = input('enter 5 values separeted by ,  ').split(',')
        for i,p in enumerate(params):
            params[i] = float(p)


        beamline.M4.extraPitch =params[0]
        beamline.M4.extraYaw = params[1]
        beamline.M4.extraRoll = params[2]
        beamline.M4.center[0] += params[3]
        beamline.M4.center[2] += params[4]



def get_action(choice):
    acs = [0,0,1,1,2,2,3,3,4,4]
    return (acs[choice],-2*choice%2+1)       


def main():
    beamline = VeritasSimpleBeamline(nrays=100000)
    rr.run_process = beamline.run_process
    xrtr.run_ray_tracing(beamline.plots,repeats=1,updateEvery=1,beamLine=beamline,generator=test_gen,generatorArgs=(beamline,beamline.plots))


if __name__ == '__main__':
    main()