__author__ = "Louisa Pickworth" #read XRT documentation: https://xrt.readthedocs.io/index.html
__date__ = "January 2022"

import sys
#sys.path.append("/Users/peterwinzell/opt/anaconda3/lib/python3.8/site-packages/xrt") ###this you need to change for your system xrt location
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rsrc
import xrt.backends.raycing.screens as rscr
import xrt.backends.raycing.run as rr
import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing.oes as roes
import xrt.backends.raycing.materials as rm
import numpy as np
from datetime import datetime
import os
import cv2 as cv

#import matplotlib as mpl
#mpl.use('agg')

# Select branch to be traced
today=today=datetime.now()
branch = 'B_branch' #remove branch switching in this version, just RIXS VERITAS
M4elliptic='yes' #'yes'
model='VERITAS_M4_fishtails_t'+datetime. now(). strftime("%Y_%m_%d-%I-%M-%S_%p") #this is just to hold plots in save and stop overwrite


numrays = 9000

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
M4pitch=0.0 #in radians
M4yaw=0.0 #in radians, directly adds yaw to M4
M4roll=0.0 #in radians, directly adds roll to M4
exX = -0.0
exY = 0.0
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


xyLimits = -5, 5

class OESE(roes.OE):
    def __init__(self, *args, **kwargs):
        kwargs = self.__pop_kwargs(**kwargs)
        super(OESE, self).__init__(*args,**kwargs)

    def __pop_kwargs(self, **kwargs):
        self.sagittalSE = kwargs.pop('sagittalSE')
        self.meridionalSE = kwargs.pop('meridionalSE')
        self.roughness = kwargs.pop('roughness')
        return kwargs

    def local_n_distorted(self, x, y):
        if (self.sagittalSE==0):
            b = 0
        else:
            b = np.random.normal(0.0, self.sagittalSE, len(x))
            print (np.mean(b), np.std(b))
        if (self.meridionalSE==0):
            a = 0
        else:
            a = np.random.normal(0.0, self.meridionalSE, len(y))
            print (np.mean(a),np.std(a))
        return a,b

    def local_z_distorted(self, x, y):
        if (self.roughness==0):
            z = 0
        else:
            z = np.random.normal(0.0, self.roughness, len(x))
        return z


class EllipticalMirrorParamSE(roes.EllipticalMirrorParam):
    def __init__(self, *args, **kwargs):
        kwargs = self.__pop_kwargs(**kwargs)
        super(EllipticalMirrorParamSE, self).__init__(*args, **kwargs)

    def __pop_kwargs(self, **kwargs):
        self.sagittalSE = kwargs.pop('sagittalSE')
        self.meridionalSE = kwargs.pop('meridionalSE')
        self.roughness = kwargs.pop('roughness')
        return kwargs

    def local_n_distorted(self, s, phi):
        r = self.local_r(s, phi)
        x, y, z = self.param_to_xyz(s, phi, r)

        if (self.sagittalSE==0):
            b = 0
        else:
            b = np.random.normal(0.0, self.sagittalSE, len(x))
            print (np.mean(b),np.std(b))
        if (self.meridionalSE==0):
            a = 0
        else:
            a = np.random.normal(0.0, self.meridionalSE, len(y))
            print (np.mean(a),np.std(a))

        return a,b

    def local_r_distorted(self, s, phi):
        r = self.local_r(s, phi)
        x, y, z = self.param_to_xyz(s, phi, r)
        if (self.roughness == 0):
            r1 = r
        else:
            z += np.random.normal(0.0, self.roughness, len(x))
            s1, phi1, r1 = self.xyz_to_param(x, y, z)
        return r1 - r



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
                                              center=[exX,distSLM4APXPS,exY],
                                              positionRoll=-np.pi/2, #horizontal deflection
                                              extraRoll=M4roll, #placeholder to scan effects of yaw, radians
                                              pitch=(pitchM4APXPS)*np.pi/180, #radians
                                              extraPitch=M4pitch, # scan effects of pitch
                                              extraYaw=M4yaw, #placeholder to scan effects of yaw, radians
                                              meridionalSE=merSEM4APXPS, #meridonial figure error, from metrology
                                              sagittalSE=sagSEM4APXPS, #sagittal figure error, from metrology
                                              roughness=roughM4APXPS, #surface roughness, from metrology
                                              limPhysX=[-20.0, 20.0], #physical size of optic, x, mm
                                              limOptX=[-5.0, 5.0], #size of optic apature, x (perpendicular to beam)
                                              limPhysY=[-55.0, 55.0], #physical size of optic, y,mm
                                              limOptY=[-45.0, 45.0]  #size of optic apature, y (along beam propagation)
                                                )



   #Parameters of the mirror can be pulled, printed, stored...
    print('beamline M4 ellipse pitch: ', np.degrees(beamLine.M4.pitch))
    print('beamline M4 ellipse yaw: ', np.degrees(beamLine.M4.yaw))
    print('beamline M4 ellipse roll: ', np.degrees(beamLine.M4.roll))
    print('beamline M4 ellipse definition p: ', beamLine.M4.p)
    print('beamline M4 ellipse definition q: ', beamLine.M4.q)
    print('beamline M4 ellipse global x: ', beamLine.M4.center[0])
    print('beamline M4 ellipse global y: ', beamLine.M4.center[1])
    print('beamline M4 ellipse global z: ', beamLine.M4.center[2])


    beamLine.scrM4 = rscr.Screen(beamLine, name='screenM4', center=beamLine.M4.center) #screen perpendicular to beam at M4 center

    tmppitchM4 = pitchM4APXPS*np.pi/180

    displF = 0. ## displacement from ideal focus
    dispEx1 = 500 ## displacement extra screen 1, unit??
    dispEx2 = -500 ## displacement extra screen 1
    #screen at focal position
    beamLine.scrTgt = rscr.Screen(beamLine, name='screen', center=[beamLine.M4.center[0]-(distM4TgtAPXPS+displF)*np.sin(2*(tmppitchM4)),
                                                                    beamLine.M4.center[1]+(distM4TgtAPXPS+displF)*np.cos(2*(tmppitchM4)),
                                                                    beamLine.M4.center[2]])

    beamLine.scrTgt.dqs = np.linspace(-14, 14, 9)
 

    print ('target enter: ', beamLine.scrTgt.center)


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



def define_plots(beamLine,path):


    #Plots are defined here, they can be quite fancy, please see documentation: https://xrt.readthedocs.io/plots.html

    plots = []


    plotsSL=[]
    xlims = np.linspace(5,-5,9)
    pm = 2.5
    for i, dq in enumerate(beamLine.scrTgt.dqs):
        plot = xrtp.XYCPlot('beamscrTgt_{0:02d}'.format(i), aspect = 'equal',
            xaxis=xrtp.XYCAxis('$x$', 'mm',limits=None,bins=256, ppb=2),
            yaxis=xrtp.XYCAxis( '$z$', 'mm',limits=None,bins=256, ppb=2))
            # ePos=0, title=beamLine.scrTgt.name+'-{0:02d}'.format(i))
        plot.xaxis.fwhmFormatStr = '%.4f'
        plot.yaxis.fwhmFormatStr = '%.4f'
        # plot.textPanel = plot.fig.text(
        #     0.2, 0.75, '', transform=plot.fig.transFigure, size=14, color='r',
        #     ha='left')
        # plot.textPanelTemplate = '{0}: d$q=${1:+.0f} mm'.format('{0}', dq)
        plots.append(plot)
        save_name = 'test_img_{}.txt'.format(i)
        plot.saveName = os.path.join(path,save_name)


    return plots, plotsSL


def data_generator(plots,beamLine,name,save_path):
    # generator script in runner
    pitches = np.linspace(-10,10,5)*1e-4
    yaws = np.linspace(-0.02,0.02,5)
    rolls = np.linspace(-0.02,0.02,5)
    transl = np.linspace(-3,3,3) # +- 1mm
    input_list = _get_input(pitches,yaws,rolls,transl)
    
    for i,(pitch,yaw,roll,x,z) in enumerate(input_list[110:120]):
        beamLine.M4.extraPitch = pitch
        beamLine.M4.extraYaw = yaw
        beamLine.M4.extraRoll = roll
        beamLine.M4.center = [x,distSLM4APXPS,z]

        yield # Return to raytracing, will start here again when done

        imgnr=0
        images = []
        axes = []
        sets = (pitch,yaw,roll,x,z)
        print('sample: ',i)
        print(sets)
        for plot in plots:
            
            s_str = str(i).zfill(5)
            i_str = str(imgnr).zfill(2)
            save_name = name + '_'  + s_str + '_' + i_str + '.png'

            t2D = plot.total2D_RGB
            if t2D.max() > 0:
                t2D = t2D*255.0/t2D.max()
            t2D = np.uint8(cv.flip(t2D,0))
            t2D = cv.cvtColor(t2D,cv.COLOR_RGB2BGR)
            cv.imwrite(os.path.join(save_path,save_name),t2D)

            imgnr += 1
            images.append(save_name)
            axes.append((plot.cx,plot.dx,plot.cy,plot.dy))
            plot.xaxis.limits = None
            plot.yaxis.limits = None
        print(axes)
        input('cont...')
        #xml_root = _build_xml(xml_root,i,sets,images,axes)
        

def _get_input(pitches,yaws,rolls,transl):
    input_list = []
    for pitch in pitches:
        for yaw in yaws:
            for roll in rolls:
                for x in transl:
                    for z in transl:
                        input_list.append((pitch,yaw,roll,x,z))
    return input_list



def main():
    """

    :rtype : none
    """
    print ('main')

    path = 'C:\\Users\\emiwin\\exjobb\\test_imgs'
    name = 'test_img'
    rr.run_process = run_process
    beamLine = build_beamLine(nrays=numrays)
    plots, plotsSL = define_plots(beamLine,path)

    # this runs the number of rays for the time in repeats, updating the defined plots.
    # you can also define parameters to scan (e.g mirror pitch), store or plot those variables.
    # see documentation or i can send example script
    xrtr.run_ray_tracing(plots,repeats=10,updateEvery=10,beamLine=beamLine,
                        generator=data_generator, generatorArgs=(plots,beamLine,name,path))


if __name__ == '__main__':
    main()
