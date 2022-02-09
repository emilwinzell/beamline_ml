import sys
import os
sys.path.append("/Users/peterwinzell/opt/anaconda3/lib/python3.8/site-packages/xrt") ###this you need to change for your system xrt location
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rsrc
import xrt.backends.raycing.screens as rscr
import xrt.backends.raycing.run as rr
import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.materials as rm
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import cv2 as cv
import argparse

from VERITAS_M4_ import EllipticalMirrorParamSE, OESE

# Select branch to be traced
today=today=datetime.now()
branch = 'B_branch' #remove branch switching in this version, just RIXS VERITAS
M4elliptic='yes' #'yes'
model='VERITAS_M4_fishtails_t'+datetime. now(). strftime("%Y_%m_%d-%I-%M-%S_%p") #this is just to hold plots in save and stop overwrite


numrays = 6000

#####ENERGY
E=400 #1612 max for harmonic 1
resolution=20000 #E/dE
sourcebandwidth = 1/resolution # fractional

#Exit slit
slitsizeY_default=0.05 #mm ES vertical
ES_delta=0. #move ES position, note this *is* in isolation,  M4 and focus stay in global position


#multipicitive fractional errors on mirror radii, 1=nominal
M4longerr=1.0
M4shorterr=1.0
#######################################
#OPTIC Properties



# Measured slope errors in rad and m (roughness, nm)
merSEM4APXPS = np.radians(0.48/3600) #B_branch
sagSEM4APXPS = np.radians(2.37/3600) #B_branch
##Roughness
roughM4APXPS = 0.3*1e-9 #B_branch

### Incident angles
pitchM4APXPS = 2.0 #deg
M4yaw=0 #in radians, directly adds yaw to M4
M4roll=0 #in radians, directly adds roll to M4
focus_position=0#adds extra distance onto focus position directly from nominal

#####################
#DISTANCES
##########
distSLM4APXPS = 14500.-ES_delta #
distM4TgtAPXPS = 700. +focus_position #
#M4  - Ellipse, parametricly defined in BL setup


#
### constants
speedoflight = 299792458
h = 4.135667e-15

#material properties of mirror coating
mAu = rm.Material('Au', rho=19.3)

def build_beamLine(pitch,yaw,roll,nrays=raycing.nrays):

    """

    :rtype : object
    """
    beamLine = raycing.BeamLine(azimuth=0, height=0)

    """
   #This is a geometric source representing the distribution at the exitslit for a 400eV operation. in addition to absolute position,
   the vertical (z) opening can be changed impacting energy resolution, and the horizontal distribution changes gradualy with energy
   of operation. This can be tabulated from other raytracing should we wish to incude these effects
   """
    beamLine.EPU = rsrc.GeometricSource(beamLine, name='EPU53', center=[0,0,0],
                                        nrays=nrays,
                                        distx='normal', dx=0.121, disty=None, dy=0, distz='flat', dz=0.1, #units mm's
                                        distxprime='normal', dxprime=0.0033*0.4246609*np.pi/180, distzprime='normal', dzprime=0.036*0.4246609*np.pi/180, #units sigma, mm's
                                        distE='flat',
                                        energies=(E-(sourcebandwidth*E),E+(sourcebandwidth*E) ), #units eV
                                        polarization='horizontal',
                                        pitch=0, yaw=0)





    #M4 elliptic: this is the final focusing optic with an elliptic figure
    beamLine.M4 = EllipticalMirrorParamSE(beamLine, 'M4', surface='Au', material=(mAu,),
                                              p=14500, #entrance arm, mm
                                              q=700, #exit arm, mm
                                              f1=beamLine.EPU.center, #distance to source (in real beam line the secondary source or exit slit) center. should be equal to p for ideal
                                              center=[0,distSLM4APXPS,0],
                                              positionRoll=-np.pi/2, #horizontal deflection
                                              extraRoll=roll, #placeholder to scan effects of yaw, radians
                                              pitch=(pitchM4APXPS)*np.pi/180, #radians
                                              extraPitch=pitch,
                                              extraYaw=yaw, #placeholder to scan effects of yaw, radians
                                              meridionalSE=merSEM4APXPS, #meridonial figure error, from metrology
                                              sagittalSE=sagSEM4APXPS, #sagittal figure error, from metrology
                                              roughness=roughM4APXPS, #surface roughness, from metrology
                                              limPhysX=[-20.0, 20.0], #physical size of optic, x, mm
                                              limOptX=[-5.0, 5.0], #size of optic apature, x (perpendicular to beam)
                                              limPhysY=[-55.0, 55.0], #physical size of optic, y,mm
                                              limOptY=[-45.0, 45.0]  #size of optic apature, y (along beam propagation)
                                                )

    beamLine.scrM4 = rscr.Screen(beamLine, name='screenM4', center=beamLine.M4.center) #screen perpendicular to beam at M4 center

    tmppitchM4 = pitchM4APXPS*np.pi/180

    displF = 0. ## displacement from ideal focus
    #screen at focal position
    beamLine.scrTgt = rscr.Screen(beamLine, name='screen', center=[beamLine.M4.center[0]-(distM4TgtAPXPS+displF)*np.sin(2*(tmppitchM4)),
                                                                    beamLine.M4.center[1]+(distM4TgtAPXPS+displF)*np.cos(2*(tmppitchM4)),
                                                                    beamLine.M4.center[2]])

    return beamLine

def run_process(beamLine, shineOnly1stSource=False):

    beamSource = beamLine.sources[0].shine()
    beamM4g, beamM4l = beamLine.M4.reflect(beamSource)
    beamM4Scr = beamLine.scrM4.expose(beamM4g)
    beamTgtScr = beamLine.scrTgt.expose(beamM4g)
    outDict = {'beamSource': beamSource, 'beamM4Scr': beamM4Scr,
               'beamTgtScr': beamTgtScr}
    return outDict



def define_plots(beamLine,name):

    #Plots are defined here, they can be quite fancy, please see documentation: https://xrt.readthedocs.io/plots.html

    plots = []


    plot = xrtp.XYCPlot('beamSource')
    plot.xaxis.fwhmFormatStr = '%.4f'
    plot.yaxis.fwhmFormatStr = '%.4f'
    plots.append(plot)

    plotsSL = []

    plot = xrtp.XYCPlot('beamM4Scr')
    plot.xaxis.fwhmFormatStr = '%.4f'
    plot.yaxis.fwhmFormatStr = '%.4f'
    plots.append(plot)

    plotsSL = []



    plot = xrtp.XYCPlot('beamTgtScr',title='TgtScr')
    plot.xaxis.fwhmFormatStr = '%.4f'
    plot.yaxis.fwhmFormatStr = '%.4f'
    #plot.saveName = name
    plots.append(plot)


    plotsSL=[]


    return plots, plotsSL

def makedirs(parent):
    dirname = datetime.now().strftime("%m%d%H%M")
    path = os.path.join(parent,dirname)
    os.mkdir(path)

    img_path = os.path.join(path,'images')
    os.mkdir(img_path)
    return path,img_path

def retrieve_total2D(name,plots):
    # self.total2D_RGB in plotter
    for plot in plots:
        if plot.title == 'TgtScr':
            t2D = plot.total2D_RGB

    print('IN AFTER SCRIPT')
    #print(t2D)
    print(name)
    t2D *= (255.0/t2D.max())
    cv.imwrite(name,t2D)
    return


def main():
    """

    :rtype : none
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--base", help="path to base directory")
    args = parser.parse_args()
    repeats = 5 # number of repeats in raytracing
    rr.run_process = run_process

    num_reps = 10 # number of repetitions per setting
    p_list = np.linspace(1,5,5)*1e-5
    y_list = np.array([0, 0.01, 0.05])
    r_list = np.array([0, 0.01])

    total_samples = num_reps * len(p_list) * len(y_list) * len(r_list)
    print('About to create {} samples'.format(total_samples))
    print('Estimated time: ', str(timedelta(seconds=total_samples*repeats*2)))
    input('Continue? press anything...')

    path,img_path = makedirs(args.base)
    data_df = pd.DataFrame(columns=('img', 'pitch','yaw','roll'))

    i = 0
    for p in p_list:
        for y in y_list:
            for r in r_list:
                for n in range(num_reps):
                    beamLine = build_beamLine(p,y,r,nrays=numrays)
                    imgname = datetime.now().strftime("%M%S%f") + '.png'
                    name = os.path.join(img_path,imgname)
                    plots, plotsSL = define_plots(beamLine,name)
                    xrtr.run_ray_tracing(plots,repeats=repeats,updateEvery=repeats,beamLine=beamLine,afterScript=retrieve_total2D,afterScriptArgs=(name,plots,))
                    data_df.loc[i] = [imgname,p,y,r]
                    i += 1
                    print('img {0} out of {1}'.format(i,total_samples), end='\r')

    filename = os.path.join(path, 'labels.csv')
    data_df.to_csv(filename, index=False)
    print('\n DONE')



if __name__ == '__main__':
    main()
