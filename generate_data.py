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
import xml.etree.ElementTree as ET

import matplotlib as mpl
mpl.use('agg')

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
M4pitch=0
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

def build_beamLine(nrays=raycing.nrays):

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
                                              extraRoll=M4roll, #placeholder to scan effects of yaw, radians
                                              pitch=(pitchM4APXPS)*np.pi/180, #radians
                                              extraPitch=M4pitch,
                                              extraYaw=M4yaw, #placeholder to scan effects of yaw, radians
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
    beamLine.scrTgt.dqs = np.linspace(-80, 80, 9)

    return beamLine

def run_process(beamLine, shineOnly1stSource=False):

    beamSource = beamLine.sources[0].shine()
    beamM4g, beamM4l = beamLine.M4.reflect(beamSource)
    beamM4Scr = beamLine.scrM4.expose(beamM4g)
    outDict = {'beamSource': beamSource, 'beamM4Scr': beamM4Scr}

    for i, dq in enumerate(beamLine.scrTgt.dqs):
        beamLine.scrTgt.center[1] = distSLM4APXPS + distM4TgtAPXPS + dq
        outDict['beamscrTgt_{0:02d}'.format(i)] = beamLine.scrTgt.expose(beamM4g)
    return outDict



def define_plots(beamLine):

    #Plots are defined here, they can be quite fancy, please see documentation: https://xrt.readthedocs.io/plots.html

    plots = []


    # plot = xrtp.XYCPlot('beamSource')
    # plot.xaxis.fwhmFormatStr = '%.4f'
    # plot.yaxis.fwhmFormatStr = '%.4f'
    # plots.append(plot)
    #
    # plotsSL = []
    #
    # plot = xrtp.XYCPlot('beamM4Scr')
    # plot.xaxis.fwhmFormatStr = '%.4f'
    # plot.yaxis.fwhmFormatStr = '%.4f'
    # plots.append(plot)
    #
    # plotsSL = []


    for i, dq in enumerate(beamLine.scrTgt.dqs):
        plot = xrtp.XYCPlot('beamscrTgt_{0:02d}'.format(i),
            title='TgtScr')

        plot.xaxis.fwhmFormatStr = '%.4f'
        plot.yaxis.fwhmFormatStr = '%.4f'
        plots.append(plot)


    plotsSL=[]


    return plots, plotsSL


def makedirs(parent):
    dirname = datetime.now().strftime("%m%d%H%M")
    path = os.path.join(parent,dirname)
    os.mkdir(path)

    img_path = os.path.join(path,'images')
    os.mkdir(img_path)
    return dirname,path,img_path


def write_xml(path,root):
    # afterScript in runner
    tree = ET.ElementTree(_indent(root))
    tree.write(os.path.join(path,'data.xml'), xml_declaration=True, encoding='utf-8')
    return

#pretty print method from: https://roytuts.com/building-xml-using-python/
def _indent(elem, level=0):
    i = "\n" + level*"  "
    j = "\n" + (level-1)*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for subelem in elem:
            _indent(subelem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = j
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = j
    return elem


def data_generator(plots,beamLine,name,save_path,xml_root):
    # generator script in runner
    pitches = np.linspace(1,5,5)*1e-5
    yaws = np.array([0, 0.01, 0.05])
    rolls = np.array([0, 0.01])
    transl = np.linspace(-5,5,11) # +- 5mm
    samplenr = 1
    yaw = 0.0
    roll = 0.0
    for pitch in pitches:
        beamLine.M4.extraPitch = pitch
        print('pitch is: ',pitch)
        imgnr=0
        images = []
        for plot in plots:
            if plot.title == 'TgtScr':
                t2D = plot.total2D_RGB
                s_str = str(samplenr).zfill(5)
                i_str = str(imgnr).zfill(2)
                save_name = name + '_'  + s_str + '_' + i_str + '.png'
                plot.saveName = os.path.join(save_path,save_name)
                imgnr += 1
                images.append(save_name)
        sets = (pitch,yaw,roll)
        xml_root = _build_xml(xml_root,samplenr,sets,images)

        samplenr += 1
        yield

def _build_xml(root,nbr,settings,images):
    sample = ET.SubElement(root,'sample_{}'.format(nbr))
    specs = ET.SubElement(sample,'specifications')

    pitch = ET.SubElement(specs,'pitch',{'unit':'rad'})
    pitch.text = str(settings[0])
    yaw = ET.SubElement(specs,'yaw',{'unit':'rad'})
    yaw.text = str(settings[1])
    roll = ET.SubElement(specs,'roll',{'unit':'rad'})
    roll.text = str(settings[2])

    imgs = ET.SubElement(sample,'images')
    for i in images:
        img = ET.SubElement(imgs,'image',{'file':i})
    return root


    
def main():
    """

    :rtype : none
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--base", help="path to base directory")
    args = parser.parse_args()
    repeats = 5 # number of repeats in raytracing
    rr.run_process = run_process
    beamLine = build_beamLine(nrays=numrays)
    plots, plotsSL = define_plots(beamLine)


    timestp,path,img_path = makedirs(args.base)
    root = ET.Element('data')

    xrtr.run_ray_tracing(
        plots,repeats=repeats, updateEvery=repeats, beamLine=beamLine,
        generator=data_generator, generatorArgs=(plots,beamLine,timestp,img_path,root),
        afterScript=write_xml, afterScriptArgs=(path,root,))


    print('\n DONE')



if __name__ == '__main__':
    main()
