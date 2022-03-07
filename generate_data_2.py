"""
GENERATE DATA VERSION 2
Using new, updated version of VERITAS beamline.
Random parameters, large number of rays.
Should be run on multiple cores.

Emil Winzell
March 2022
"""

import sys
import os
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rsrc
import xrt.backends.raycing.screens as rscr
import xrt.backends.raycing.run as rr
import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing.apertures as rap
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.materials as rm
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import cv2 as cv
import argparse
import xml.etree.ElementTree as ET

from itertools import count
import matplotlib as mpl
mpl.use('agg')

from VERITAS_full_simple import Grating, OESE ,ToroidMirrorSE ,EllipticalMirrorParamSE


# Select branch to be traced
today=today=datetime.now()
branch = 'B_branch' #remove branch switching in this version, just RIXS VERITAS
M4elliptic='yes' #'yes'
model='VERITAS_M4_fishtails_t'+datetime. now(). strftime("%Y_%m_%d-%I-%M-%S_%p") #this is just to hold plots in save and stop overwrite


numrays = 6000

#####ENERGY
E=400 #1612 max for harmonic 1
sourcebandwidth = 0.001 # fractions
photonenergyoffset = 0.#-5.5 #-5.5 #-30!!
photonenergy=E-photonenergyoffset
E = photonenergy + photonenergyoffset
harmonic = 1 #undulator harmonic
ES_delta=0. #move ES position, note this *is* in isolation, it moves M4 and focus

slitsizeY_default=0.05 #mm ES vertical

M4yaw=0. #in radians, directly adds yaw to M4
M4roll=0.
M4pitch=0.0
focus_position=0#adds extra distance onto focus position directly from nominal
lims=5#1/2 size of plot at target screen


delta_Y1=[0] #hangover from prior work
delta_Y2=[0] #hangover from prior work
#multipicitive fractional errors on mirror radii, 1=nominal
M3longerr=1.
M3shorterr=1.

M4longerr=1.0
M4shorterr=1.0
#######################################
#OPTIC Properties

#####################
#DISTANCES
##########

distM1BDA_ = 19000. ## MMA positions
distSM1 = 24000. ## from source not BDA MMA
#the position for M2 is calculated in the PGM sub-routine
distM1BDA = 750.
distBDAPG = 1560.-distM1BDA#####update check, PG position to M1-BDA

distPGM3 = 2440. #from PG

distM3SL_B = 13500.+ES_delta ## Exit slit 
distSLM4APXPS = 14500.-ES_delta #
distM4TgtAPXPS = 700. +focus_position #


# # Measured slope errors in rad and m (roughness, nm)
merSEM1 = np.radians(0.125/3600) 
sagSEM1 = np.radians(0.71/3600)

merSEM2 = np.radians(0.05/3600)
sagSEM2 = np.radians(0.033/3600)

# M3 M4
merSEM3APXPS = np.radians(0.024493/3600) #B_branch
sagSEM3APXPS = np.radians(0.1/3600) #B_branch

merSEM4APXPS = np.radians(0.48/3600) #B_branch
sagSEM4APXPS = np.radians(2.37/3600) #B_branch

##Roughness
set_rough=0.3 #0.3 #nm

roughM1 = set_rough*1e-9
roughM2 = set_rough*1e-9

roughM3APXPS = set_rough*1e-9 #B_branch
roughM4APXPS = set_rough*1e-9 #B_branch

### Incident angles 
pitchM1 = 1.0 #
pitchM2 = 2.#gets calculated
pitchPG = 2.#gets calculated
pitchM3APXPS = 1.5 #
pitchM4APXPS = 2.0 #

#### Raddi for mirrors
## in mm's
#M1 - toroid
m1r= 837.7
m1R=1155969  
#M3 - cylindrical
m3r_b=706.9
m3R_b=1.0E19
#M4  - Ellipse, parametricly defined in BL setup 

## Grating setting
cff = 2.25 ####
m = 1.
n = 1200.0
d = 0.001/n
exitheight = 20


#####EXIT SLIT#########################################
slitsizeX = 5
slitsizeY  = [slitsizeY_default]
#2.5 4.5 typical used at MAX II for high flux experiments### put at MMA values
BDAhor = 2
BDAvert = 2

sampleangle =-90. ## angle to the beam: 0 or 180 is ||el to beam, 90, 270 is right angle/flip

### constants
speedoflight = 299792458
h = 4.135667e-15

mAu = rm.Material('Au', rho=19.3)
mRh = rm.Material('Rh', rho=12.41)
mRh=mAu
mAuGrating = rm.Material('Au', rho=19.3, kind='grating')

def quaternion_product(q1, q2):

    q3 = [0,0,0,0]

    q3[0] = q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3]
    q3[1] = q1[0]*q2[1]+q1[1]*q2[0]+q1[2]*q2[3]-q1[3]*q2[2]
    q3[2] = q1[0]*q2[2]-q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
    q3[3] = q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]

    return q3

def rotate_vector_around_axis(vect, axis, angle):

    rotvect = [0,0,0]

    qaxis = [np.cos(angle/2),axis[0]*np.sin(angle/2),axis[1]*np.sin(angle/2),axis[2]*np.sin(angle/2)]
    qvect = [0, vect[0], vect[1], vect[2]]
    qaxisinv = [qaxis[0], -qaxis[1], -qaxis[2], -qaxis[3]]

    tq1 = quaternion_product(qvect,qaxisinv)
    tq2 = quaternion_product(qaxis,tq1)

    rotvect[0] = tq2[1]
    rotvect[1] = tq2[2]
    rotvect[2] = tq2[3]

    return rotvect


def calculate_cPGM(cff=cff, m=m, E=photonenergy, d=d):

    lam = h*speedoflight/E
    a = 1-cff**2
    b = 2*(m*lam/d)*cff**2
    c = cff**2-1-((m*lam/d)**2)*(cff**2)

    sinbeta = (-b+np.sqrt(b*b-4*a*c))/(2*a)
    beta = np.arcsin(sinbeta)
    cosbeta = np.cos(beta)
    cosalpha = cosbeta/cff
    alpha = np.arccos(cosalpha)
    betagrazing = np.pi/2+beta
    alphagrazing = np.pi/2-alpha
    gammagrazing = 0.5*(alphagrazing+betagrazing)

    return -betagrazing,gammagrazing


def build_beamLine(nrays=raycing.nrays):
    """
    :rtype : object
    """
    beamLine = raycing.BeamLine(azimuth=0, height=0)

    beamLine.EPU = rsrc.GeometricSource(beamLine, name='EPU53', center=[0,0,0], 
                                        nrays=nrays, 
                                        distx='normal', dx=0.05586, disty=None, dy=0, distz='normal', dz=0.02508, #units mm's
                                        distxprime='normal', dxprime=20.6e-6, distzprime='normal', dzprime=20.0e-6, #units sigma, mm's
                                        distE='flat', 
                                        energies=(E-(sourcebandwidth*E),E+(sourcebandwidth*E) ), #units eV
                                        polarization='horizontal', 
                                        pitch=0, yaw=0)
    
    #Beam defining apature BDA
    beamLine.BDA = rap.RectangularAperture(beamLine, 'BDA', center=[0,distM1BDA_, 0],
                                           kind=('top', 'bottom', 'left', 'right'), 
                                           opening=[BDAvert/2,-BDAvert/2,-BDAhor/2,BDAhor/2])    


    beamLine.scrBDA = rscr.Screen(beamLine, 'screenBDA', center=beamLine.BDA.center)
    #print ('BDA/MMA: ', beamLine.scrBDA.center)

    #M1 ohjaa vasemmalle, tällöin roll = -90deg, eli -np.pi/2 R=1057822 ideaali; tuli 1216580
    beamLine.M1 = ToroidMirrorSE(beamLine, name='M1',  r=m1r, R=m1R, positionRoll=-np.pi/2,
                                    center=[0,distSM1,0], pitch=pitchM1*np.pi/180, meridionalSE=merSEM1, 
                                    sagittalSE=sagSEM1, roughness=roughM1)

    beamLine.scrM1 = rscr.Screen(beamLine, name='screenM1', center=beamLine.M1.center)
    beamLine.M1.Y1=delta_Y1 #yaw scan
    print ('M1: ', beamLine.M1.center, np.degrees(beamLine.M1.pitch))

    (pitchPG, pitchM2) = calculate_cPGM(cff, m, E, d)
    distBDAM2 = distBDAPG-exitheight/np.tan(2*pitchM2)
    beamLine.M2 = OESE(beamLine, 'M2', 
                        center=[beamLine.M1.center[0]-(distBDAM2+distM1BDA)*np.sin(2*beamLine.M1.pitch),
                                beamLine.M1.center[1]+(distBDAM2+distM1BDA)*np.cos(2*beamLine.M1.pitch),
                                beamLine.M1.center[2]],
                        pitch=pitchM2, yaw=2*beamLine.M1.pitch, positionRoll=0, roughness = roughM2,
                        sagittalSE = sagSEM2, meridionalSE = merSEM2)
    print ('M2: ', beamLine.M2.center, np.degrees(beamLine.M2.pitch))
    beamLine.scrM2 = rscr.Screen(beamLine, name='screenM2', center=beamLine.M2.center)

    #Grating PGM
    beamLine.PG = Grating(beamLine, 'PG', 
                            center=[beamLine.M2.center[0]-(distBDAPG-distBDAM2)*np.sin(2*beamLine.M1.pitch),
                                    beamLine.M2.center[1]+(distBDAPG-distBDAM2)*np.cos(2*beamLine.M1.pitch),
                                    beamLine.M2.center[2]+exitheight],
                            material=(mAuGrating,), positionRoll=np.pi, pitch=pitchPG, yaw=2*beamLine.M1.pitch)
    beamLine.PG.order = int(m)
    print ('PG: ', beamLine.PG.center, np.degrees(beamLine.PG.pitch))
    beamLine.scrPG = rscr.Screen(beamLine, name='screenPG', center=beamLine.PG.center)

    #M3
    print (branch)
    beamLine.M3 = ToroidMirrorSE(beamLine, 'M3',  r=m3r_b, R=m3R_b,
                                    center=[beamLine.PG.center[0]-distPGM3*np.sin(2*beamLine.M1.pitch),
                                            beamLine.PG.center[1]+distPGM3*np.cos(2*beamLine.M1.pitch),
                                            beamLine.PG.center[2]],
                                    positionRoll=-np.pi/2, pitch=1*(2*pitchM1+pitchM3APXPS)*np.pi/180,
                                    meridionalSE=merSEM3APXPS, sagittalSE=sagSEM3APXPS, roughness=roughM3APXPS)

       
    beamLine.M3.Y2=delta_Y2 #yaw scan
    print ('M3 center: ', beamLine.M3.center,'angle: ', np.degrees(beamLine.M3.pitch))

    tmppitchM3 = pitchM3APXPS*np.pi/180
    beamLine.scrM3 = rscr.Screen(beamLine, name='screenM3', center=beamLine.M3.center)
    print ('M3 pitch', np.degrees(tmppitchM3))
    #print('np.degrees(2*beamLine.M1.pitch+2*tmppitchM3): ', np.degrees(2*beamLine.M1.pitch+2*tmppitchM3))
    ##EXIT SLIT
    beamLine.SL = rap.RectangularAperture(beamLine, 'Slit', 
                                            center=[beamLine.M3.center[0]-distM3SL_B*np.sin(2*beamLine.M1.pitch+2*tmppitchM3),
                                                    beamLine.M3.center[1]+distM3SL_B*np.cos(2*beamLine.M1.pitch+2*tmppitchM3),
                                                    beamLine.M3.center[2]],
                                            kind=['left', 'right', 'bottom', 'top'],
                                            opening=[-slitsizeX/2, slitsizeX/2, -slitsizeY_default/2, slitsizeY_default/2])
        
    beamLine.SL.slit = slitsizeY 
    print ('SL: ', beamLine.SL.center)
    beamLine.scrSL = rscr.Screen(beamLine, name='screenSL', center=beamLine.SL.center)
    #M4 elliptic
    beamLine.M4 = EllipticalMirrorParamSE(beamLine, 'M4', 
                                            p=14500, 
                                            q=700,
                                            f1=beamLine.SL.center,
                                            center=[beamLine.SL.center[0]-distSLM4APXPS*np.sin(2*(beamLine.M1.pitch+tmppitchM3)),
                                                    beamLine.SL.center[1]+distSLM4APXPS*np.cos(2*(beamLine.M1.pitch+tmppitchM3)),
                                                    beamLine.SL.center[2]], positionRoll=-np.pi/2, 
                                            pitch=(2*pitchM1 + 2*pitchM3APXPS + pitchM4APXPS)*np.pi/180,
                                            extraYaw=M4yaw,
                                            extraRoll=M4roll,
                                            extraPitch=M4pitch,
                                            meridionalSE=merSEM4APXPS, 
                                            sagittalSE=sagSEM4APXPS, 
                                            roughness=roughM4APXPS, )
    global m4Center
    m4Center = beamLine.M4.center

    print('beamLine.M1.pitch: ', np.degrees(beamLine.M1.pitch))
    print('beamLine.M3.pitch: ', np.degrees(beamLine.M3.pitch))
    print('beamline M4 ellipse pitch: ', np.degrees(beamLine.M4.pitch))
    print('beamline M4 ellipse p: ', beamLine.M4.p)
    print('beamline M4 ellipse q: ', beamLine.M4.q)
     
    beamLine.scrM4 = rscr.Screen(beamLine, name='screenM4', center=beamLine.M4.center)

    tmppitchM4 = pitchM4APXPS*np.pi/180
        
    displF = 0.
    beamLine.scrTgt11 = rscr.Screen(beamLine, name='screenTgt-0', 
                                    center=[beamLine.M4.center[0]-(distM4TgtAPXPS+displF)*np.sin(2*(beamLine.M1.pitch + tmppitchM3+ tmppitchM4)),
                                            beamLine.M4.center[1]+(distM4TgtAPXPS+displF)*np.cos(2*(beamLine.M1.pitch + tmppitchM3+ tmppitchM4)),
                                            beamLine.M4.center[2]])
    beamLine.scrTgt11.dqs = np.linspace(-14, 14, 9)
    global tgtCenter
    tgtCenter = beamLine.M4.center[1]+(distM4TgtAPXPS+displF)*np.cos(2*(beamLine.M1.pitch + tmppitchM3+ tmppitchM4))
    print ('Tgt11, target 11 center: ', beamLine.scrTgt11.center)

    return beamLine


def run_process(beamLine, shineOnly1stSource=False):
    
    beamSource = beamLine.sources[0].shine()
    beamLine.BDA.propagate(beamSource)
    beamBDAScr = beamLine.scrBDA.expose(beamSource)
    beamM1g, beamM1l = beamLine.M1.reflect(beamSource)
    beamM1Scr = beamLine.scrM1.expose(beamM1g)
    beamM2g, beamM2l = beamLine.M2.reflect(beamM1g)
    beamM2Scr = beamLine.scrM2.expose(beamM2g)
    beamPGg, beamPGl = beamLine.PG.reflect(beamM2g)
    beamPGScr = beamLine.scrPG.expose(beamPGg)
    beamM3g, beamM3l = beamLine.M3.reflect(beamPGg)
    beamM3Scr = beamLine.scrM3.expose(beamM3g)
    beamSLDispScr = beamLine.scrSL.expose(beamM3g)
    beamLine.SL.propagate(beamM3g)
    beamSLScr = beamLine.scrSL.expose(beamM3g)
    beamM4g, beamM4l = beamLine.M4.reflect(beamM3g)
    beamM4Scr = beamLine.scrM4.expose(beamM4g)
    beamTgtScr11 = beamLine.scrTgt11.expose(beamM4g)
    outDict = {'beamSource': beamSource, 'beamM4Scr': beamM4Scr}
    

    for i, dq in enumerate(beamLine.scrTgt11.dqs):
       beamLine.scrTgt11.center[1] = tgtCenter + dq
       outDict['beamscrTgt_{0:02d}'.format(i)] = beamLine.scrTgt11.expose(beamM4g)
    return outDict



def define_plots(beamLine,bins,limit):

    #Plots are defined here, they can be quite fancy, please see documentation: https://xrt.readthedocs.io/plots.html

    plots = []

    for i, dq in enumerate(beamLine.scrTgt11.dqs):
        plot = xrtp.XYCPlot('beamscrTgt_{0:02d}'.format(i),
                                xaxis=xrtp.XYCAxis('$x$', 'mm',limits=[-limit,limit],bins=bins, ppb=2),
                                yaxis=xrtp.XYCAxis( '$z$', 'mm',limits=[-limit,limit],bins=bins, ppb=2))

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

    hist_path = os.path.join(path,'histograms')
    os.mkdir(hist_path)
    return dirname,path


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

def _stack_size2a(size=2):
    """Get stack size for caller's frame.
    """
    try:
        frame = sys._getframe(size)
    except ValueError:
        print('0')
        return 0

    for size in count(size):
        frame = frame.f_back
        if not frame:
            return size


def data_rand_generator(num_samples,plots,beamLine,name,save_path,xml_root):
    img_path = os.path.join(save_path,'images')
    hist_path = os.path.join(save_path,'histograms')

    last_sample = xml_root[-1].tag.split('_')[-1]
    if last_sample.isnumeric():
        start = int(last_sample) + 1
    else:
        start =  0

    # Define param limits, all in radians
    pitch_lim = 0.001
    yaw_lim = 0.02
    roll_lim = 0.02
    # pick one random setting:
    for i in range(num_samples):
        pitch = 0.0005*np.random.randn()#np.random.uniform(-pitch_lim,pitch_lim)
        yaw = 0.01*np.random.randn()#np.random.uniform(-yaw_lim,yaw_lim)
        roll = 0.01*np.random.randn()#np.random.uniform(-roll_lim,roll_lim)
        exX = 0.2*np.random.randn()
        exZ = 0.2*np.random.randn()

        if _stack_size2a(i) > sys.getrecursionlimit()-200:
            print('Getting close to recursion limit, ending')
            return

        # Set M4 mirror
        beamLine.M4.extraPitch = pitch
        beamLine.M4.extraYaw = yaw
        beamLine.M4.extraRoll = roll
        beamLine.M4.center = [m4Center[0]+exX,m4Center[1],m4Center[2]+exZ]

        yield # Return to raytracing, will start here again when done

        imgnr=0
        images = []
        axes = []
        for plot in plots:
            # save 2D intensity image
            s_str = str(i+start).zfill(5)
            i_str = str(imgnr).zfill(2)
            save_name = name + '_'  + s_str + '_' + i_str + '.png'
            t2D = plot.total2D_RGB
            t2D = np.log(t2D+1)
            if t2D.max() > 0:
                t2D = t2D*65535.0/t2D.max()
            t2D = np.uint16(cv.flip(t2D,0))
            t2D = cv.cvtColor(t2D,cv.COLOR_RGB2BGR)
            cv.imwrite(os.path.join(img_path,save_name),t2D)
            imgnr += 1
            images.append(save_name)
            axes.append((plot.cx,plot.dx,plot.cy,plot.dy))

            # save 1D histograms
            xt1D = plot.xaxis.total1D
            xbinEdges = plot.xaxis.binEdges
            zt1D = plot.yaxis.total1D
            zbinEdges = plot.yaxis.binEdges
            data_df = pd.DataFrame({'xt1D':np.append(xt1D,0.0),
                                    'zt1D':np.append(zt1D,0.0),
                                    'xbinEdges':xbinEdges,
                                    'zbinEdges':zbinEdges})
            save_name = name + '_'  + s_str + '_' + i_str + '.csv'
            filename = os.path.join(hist_path, save_name)
            data_df.to_csv(filename, index=False)

            # reset axis limits for plot
            #plot.xaxis.limits = None
            #plot.yaxis.limits = None

        # save label data to xml
        sets = (pitch,yaw,roll,exX,exZ)
        xml_root = _build_xml(xml_root,i+start,sets,images,axes)


def _build_xml(root,nbr,settings,images,axes):
    sample = ET.SubElement(root,'sample_{}'.format(nbr))
    specs = ET.SubElement(sample,'specifications')

    pitch = ET.SubElement(specs,'pitch',{'unit':'rad'})
    pitch.text = str(settings[0])
    yaw = ET.SubElement(specs,'yaw',{'unit':'rad'})
    yaw.text = str(settings[1])
    roll = ET.SubElement(specs,'roll',{'unit':'rad'})
    roll.text = str(settings[2])
    center = ET.SubElement(specs,'center_transl',{'unit':'mm'})
    center.text = 'horizontal:{0}, vertical:{1}'.format(settings[3],settings[4])

    imgs = ET.SubElement(sample,'images')
    for i,img_str in enumerate(images):
        img = ET.SubElement(imgs,'image',{'file':img_str, 
                            'centerx':'{:.4f}'.format(axes[i][0]),
                            'dx':'{:.4f}'.format(axes[i][1]),
                            'centerz':'{:.4f}'.format(axes[i][2]),
                            'dz':'{:.4f}'.format(axes[i][3])})
    return root


    
def main():
    """

    :rtype : none
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--base", help="path to base directory")
    parser.add_argument("-t", "--timestp", help="(optional) path to timestamp if desire to continue on dataset")
    args = parser.parse_args()
    repeats = 10 # number of repeats in raytracing
    rr.run_process = run_process
    beamLine = build_beamLine(nrays=numrays)

    bins = 512
    xz_lim = 2
    num_samples = 50
    plots, plotsSL = define_plots(beamLine,bins,xz_lim)

    if args.timestp is None:
        timestp,path = makedirs(args.base)
        
        root = ET.Element('data', {'numrays':str(numrays), 
                                    'energy':str(E), 
                                    'source_bandwidth':str(sourcebandwidth)})
        setup = ET.SubElement(root, 'setup') #TODO: put all info about beamline in here...
        source = ET.SubElement(setup,'source')
        m4 = ET.SubElement(setup,'M4',{'surface':'Au',
                                        'Pitch_deg':str(pitchM4APXPS),
                                        'Yaw':str(M4yaw),
                                        'Roll':str(M4roll),
                                        'Meridional_error':str(merSEM4APXPS),
                                        'Sagittal_error':str(sagSEM4APXPS),
                                        'Roughness':str(roughM4APXPS),
                                        'Dist_to_target':str(distM4TgtAPXPS)})
        target = ET.SubElement(setup,'Target',{'dqs':'9','xz_limits':'+/- {}'.format(xz_lim),'bins':str(bins)})
    else:
        # Continue on created set
        timestp=os.path.split(args.timestp)[1]
        path = args.timestp
        if not timestp.isnumeric():
            print(timestp + ' is not a timestamp, ending...')
            return
        
        xml = os.path.join(args.timestp,'data.xml')
        tree = ET.parse(xml)
        root = tree.getroot()

    xrtr.run_ray_tracing(
        plots,repeats=repeats, updateEvery=repeats, beamLine=beamLine,
        generator=data_rand_generator, generatorArgs=(num_samples,plots,beamLine,timestp,path,root),
        afterScript=write_xml, afterScriptArgs=(path,root,))#, threads=1,processes=4)
    
    print('\n DONE')



if __name__ == '__main__':
    main()
