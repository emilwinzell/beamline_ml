import sys
import os
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

from itertools import count
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
    global tgtCenter
    tgtCenter = beamLine.M4.center[1]+(distM4TgtAPXPS+displF)*np.cos(2*(tmppitchM4))                                                               
    beamLine.scrTgt.dqs = np.linspace(-14, 14, 9)

    return beamLine

def run_process(beamLine, shineOnly1stSource=False):

    beamSource = beamLine.sources[0].shine()
    beamM4g, beamM4l = beamLine.M4.reflect(beamSource)
    beamM4Scr = beamLine.scrM4.expose(beamM4g)
    outDict = {'beamSource': beamSource, 'beamM4Scr': beamM4Scr}


    for i, dq in enumerate(beamLine.scrTgt.dqs):
        beamLine.scrTgt.center[1] = tgtCenter + dq
        #print('Center is:{0}, i is: {1} '.format(beamLine.scrTgt.center[1],i))
        outDict['beamscrTgt_{0:02d}'.format(i)] = beamLine.scrTgt.expose(beamM4g)
    return outDict



def define_plots(beamLine,bins):

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
    xlims = np.linspace(5,-5,9) # placing the plots
    pm = 2

    for i, dq in enumerate(beamLine.scrTgt.dqs):
        plot = xrtp.XYCPlot('beamscrTgt_{0:02d}'.format(i),
                                xaxis=xrtp.XYCAxis('$x$', 'mm',limits=[-pm,pm],bins=bins, ppb=2),
                                yaxis=xrtp.XYCAxis( '$z$', 'mm',limits=[-pm,pm],bins=bins, ppb=2))

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

def data_generator(input_list,plots,beamLine,name,save_path,xml_root):
    img_path = os.path.join(save_path,'images')
    hist_path = os.path.join(save_path,'histograms')
    # generator script in runner
    print('input list is {} long'.format(len(input_list)))
    seconds = len(input_list)*6 # assuming 10 repeats, 9 images per sample and 9000 rays
    print('estimated time taken: ', str(timedelta(seconds=seconds)))
    if len(input_list) > 250:
        input('input list too big, max recusion depth might be exceeded (press enter to continue)')

    last_sample = xml_root[-1].tag.split('_')[-1]
    if last_sample.isnumeric():
        start = int(last_sample) + 1
    else:
        start =  0

    
    for i,(pitch,yaw,roll,x,z) in enumerate(input_list):
        if i < start:
            continue # fwd to start

        if _stack_size2a(i-start) > sys.getrecursionlimit()-200:
            print('Getting close to recursion limit, ending')
            return
    
        beamLine.M4.extraPitch = pitch
        beamLine.M4.extraYaw = yaw
        beamLine.M4.extraRoll = roll
        beamLine.M4.center = [x,distSLM4APXPS,z]

        yield # Return to raytracing, will start here again when done

        imgnr=0
        images = []
        axes = []
        for plot in plots:
            # save 2D intensity image
            s_str = str(i).zfill(5)
            i_str = str(imgnr).zfill(2)
            save_name = name + '_'  + s_str + '_' + i_str + '.png'
            t2D = plot.total2D_RGB
            if t2D.max() > 0:
                t2D = t2D*255.0/t2D.max()
            t2D = np.uint8(cv.flip(t2D,0))
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
            plot.xaxis.limits = None
            plot.yaxis.limits = None

        # save label data to xml
        sets = (pitch,yaw,roll,x,z)
        xml_root = _build_xml(xml_root,i,sets,images,axes)
        

def _get_input(pitches,yaws,rolls,transl):
    input_list = []
    for pitch in pitches:
        for yaw in yaws:
            for roll in rolls:
                for x in transl:
                    for z in transl:
                        input_list.append((pitch,yaw,roll,x,z))
    return input_list


def data_rand_generator(num_samples,plots,beamLine,name,save_path,xml_root):
    img_path = os.path.join(save_path,'images')
    hist_path = os.path.join(save_path,'histograms')

    last_sample = xml_root[-1].tag.split('_')[-1]
    if last_sample.isnumeric():
        start = int(last_sample) + 1
    else:
        start =  0

    # generator script in runner
    pitch_lim = 0.0001
    yaw_lim = 0.0002
    roll_lim = 0.0002
    # pick one random setting:
    for i in range(num_samples):
        pitch = np.random.uniform(-pitch_lim,pitch_lim)
        yaw = np.random.uniform(-yaw_lim,yaw_lim)
        roll = np.random.uniform(-roll_lim,roll_lim)
        exX = 0#0.2*np.random.randn()
        exZ = 0#0.2*np.random.randn()

        if _stack_size2a(i) > sys.getrecursionlimit()-200:
            print('Getting close to recursion limit, ending')
            return

        # Set M4 mirror
        beamLine.M4.extraPitch = pitch
        beamLine.M4.extraYaw = yaw
        beamLine.M4.extraRoll = roll
        beamLine.M4.center = [0+exX,distSLM4APXPS,0+exZ]

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
    plots, plotsSL = define_plots(beamLine,bins)

    if args.timestp is None:
        # From scratch
        # pitches = np.linspace(-10,10,5)*1e-4
        # yaws = np.linspace(-0.02,0.02,5)
        # rolls = np.linspace(-0.02,0.02,5)
        # transl = np.linspace(-0.3,0.3,3) # +- 1mm
        # input_list = _get_input(pitches,yaws,rolls,transl)
        # np.savetxt(os.path.join(path,'input_list.txt'),input_list)

        timestp,path = makedirs(args.base)
        
        root = ET.Element('data', {'numrays':str(numrays), 
                                    'energy':str(E), 
                                    'resolution':str(resolution)})
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
        target = ET.SubElement(setup,'Target',{'dqs':'9','xz_limits':'auto','bins':str(bins)})
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

        #input = os.path.join(path,'input_list.txt')
        #input_list = np.loadtxt(input)
        

    num_samples = 200

    xrtr.run_ray_tracing(
        plots,repeats=repeats, updateEvery=repeats, beamLine=beamLine,
        generator=data_rand_generator, generatorArgs=(num_samples,plots,beamLine,timestp,path,root),
        afterScript=write_xml, afterScriptArgs=(path,root,))
    
    



    print('\n DONE')



if __name__ == '__main__':
    main()
