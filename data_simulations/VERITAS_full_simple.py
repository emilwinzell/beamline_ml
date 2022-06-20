__author__ = "Louisa Pickworth" #Samulin SPECIES pohjasta! ::: much inspired by R. Sankari
__date__ = "June 2021"

import sys
sys.path.append("C:/XRT/xrt-1.3.3")
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
import matplotlib.pyplot as plt
from datetime import datetime

# Select branch to be traced
today=today=datetime.now()
branch = 'B_branch' #remove branch switching in this version, just RIXS VERITAS
M4elliptic='yes' #'yes'
model='VERITAS_M4_fishtails_t'+datetime. now(). strftime("%Y_%m_%d-%I-%M-%S_%p") #this is just to hold plots in save and stop overwrite


numrays = 10000

#####ENERGY

E=400 #1612 max for harmonic 1
sourcebandwidth = 0.001 # fractions
photonenergyoffset = 0.#-5.5 #-5.5 #-30!!
photonenergy=E-photonenergyoffset
E = photonenergy + photonenergyoffset
harmonic = 1 #undulator harmonic
ES_delta=0. #move ES position, note this *is* in isolation, it moves M4 and focus

slitsizeY_default=0.05 #mm ES vertical



#-6.85553077e-04,  1.09588839e-04,  1.91748061e-03, -1.06451612e+00,-9.34061316e-02
M4pitch=0#2.82404352e-04 #in radians, directly adds yaw to M4
M4yaw=0#1.82926549e-04
M4roll=0#1.67707426e-03
M4lat =0# 3.26236547e-01
M4vert =0# -1.15008439e-01
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
###################################
####################
##############

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

class Grating(roes.OE):
    def local_g(self, x,y, rho=n):
        return 0,-rho,0 # constant line spacing

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

class ToroidMirrorSE(roes.ToroidMirror):
    def __init__(self, *args, **kwargs):
        kwargs = self.__pop_kwargs(**kwargs)
        super(ToroidMirrorSE, self).__init__(*args,**kwargs)

    def __pop_kwargs(self, **kwargs):
        self.sagittalSE = kwargs.pop('sagittalSE')
        self.meridionalSE = kwargs.pop('meridionalSE')
        self.roughness = kwargs.pop('roughness')
        return kwargs

    def local_n_distorted(self, x, y):
        # Draw slope errors from Gaussian distribution with given rms values
        # If rms is zero, then errors are always zero

        if (self.sagittalSE==0):
            b = 0
        else :
            b = np.random.normal(0.0, self.sagittalSE, len(x))
            print (np.mean(b),np.std(b))
        if (self.meridionalSE==0):
            a = 0
        else:
            a = np.random.normal(0.0, self.meridionalSE, len(y))
            print (np.mean(a),np.std(a))

 
        return a,b

    def local_z_distorted(self, x, y):
        if (self.roughness == 0):
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
                                           kind=('top', 'bottom', 'left', 'right'), opening=[BDAvert/2,-BDAvert/2,-BDAhor/2,
                                                                                         BDAhor/2])    


    beamLine.scrBDA = rscr.Screen(beamLine, 'screenBDA', center=beamLine.BDA.center)
    
    print ('BDA/MMA: ', beamLine.scrBDA.center)



#M1 ohjaa vasemmalle, tällöin roll = -90deg, eli -np.pi/2 R=1057822 ideaali; tuli 1216580
    beamLine.M1 = ToroidMirrorSE(beamLine, name='M1',  r=m1r, R=m1R, positionRoll=-np.pi/2,
                                    center=[0,distSM1,0], pitch=pitchM1*np.pi/180, meridionalSE=merSEM1, sagittalSE=sagSEM1,
                                    roughness=roughM1)

    beamLine.scrM1 = rscr.Screen(beamLine, name='screenM1', center=beamLine.M1.center)
    beamLine.M1.Y1=delta_Y1 #yaw scan

    print ('M1: ', beamLine.M1.center, np.degrees(beamLine.M1.pitch))

    (pitchPG, pitchM2) = calculate_cPGM(cff, m, E, d)

    distBDAM2 = distBDAPG-exitheight/np.tan(2*pitchM2)


    beamLine.M2 = OESE(beamLine, 'M2', center=[beamLine.M1.center[0]-(distBDAM2+distM1BDA)*np.sin(2*beamLine.M1.pitch),
                                                  beamLine.M1.center[1]+(distBDAM2+distM1BDA)*np.cos(2*beamLine.M1.pitch),
                                                  beamLine.M1.center[2]], pitch=pitchM2,
                          yaw=2*beamLine.M1.pitch, positionRoll=0, roughness = roughM2, sagittalSE = sagSEM2, meridionalSE = merSEM2)

    print ('M2: ', beamLine.M2.center, np.degrees(beamLine.M2.pitch))

    beamLine.scrM2 = rscr.Screen(beamLine, name='screenM2', center=beamLine.M2.center)


#Grating PGM

    beamLine.PG = Grating(beamLine, 'PG', center=[beamLine.M2.center[0]-(distBDAPG-distBDAM2)*np.sin(2*beamLine.M1.pitch),
                                                  beamLine.M2.center[1]+(distBDAPG-distBDAM2)*np.cos(2*beamLine.M1.pitch),
                                                  beamLine.M2.center[2]+exitheight], material=(mAuGrating,), positionRoll=np.pi, pitch=pitchPG,
                          yaw=2*beamLine.M1.pitch)
    beamLine.PG.order = int(m)

    print ('PG: ', beamLine.PG.center, np.degrees(beamLine.PG.pitch))

    beamLine.scrPG = rscr.Screen(beamLine, name='screenPG', center=beamLine.PG.center)

#M3

    print (branch)
    beamLine.M3 = ToroidMirrorSE(beamLine, 'M3',  r=m3r_b, R=m3R_b,
                                        center=[beamLine.PG.center[0]-distPGM3*np.sin(2*beamLine.M1.pitch),
                                                                beamLine.PG.center[1]+distPGM3*np.cos(2*beamLine.M1.pitch),
                                                                beamLine.PG.center[2]], positionRoll=-np.pi/2, 
                                                                pitch=1*(2*pitchM1+pitchM3APXPS)*np.pi/180,
                                    meridionalSE=merSEM3APXPS, sagittalSE=sagSEM3APXPS, roughness=roughM3APXPS)

       
    beamLine.M3.Y2=delta_Y2 #yaw scan
    print ('M3 center: ', beamLine.M3.center,'angle: ', np.degrees(beamLine.M3.pitch))

    tmppitchM3 = pitchM3APXPS*np.pi/180
    beamLine.scrM3 = rscr.Screen(beamLine, name='screenM3', center=beamLine.M3.center)
    print ('M3 pitch', np.degrees(tmppitchM3))
    #print('np.degrees(2*beamLine.M1.pitch+2*tmppitchM3): ', np.degrees(2*beamLine.M1.pitch+2*tmppitchM3))



##EXIT SLIT

    beamLine.SL = rap.RectangularAperture(beamLine, 'Slit', center=[beamLine.M3.center[0]-distM3SL_B*np.sin(2*beamLine.M1.pitch+2*tmppitchM3),
                                                                    beamLine.M3.center[1]+distM3SL_B*np.cos(2*beamLine.M1.pitch+2*tmppitchM3),
                                                                    beamLine.M3.center[2]],kind=['left', 'right', 'bottom', 'top'],
                                                                    opening=[-slitsizeX/2, slitsizeX/2, -slitsizeY_default/2, slitsizeY_default/2]
                                                                    )
        
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
                                                roughness=roughM4APXPS, 
                                                )

        

    print('beamLine.M1.pitch: ', np.degrees(beamLine.M1.pitch))
    print('beamLine.M3.pitch: ', np.degrees(beamLine.M3.pitch))
    print('beamline M4 ellipse pitch: ', np.degrees(beamLine.M4.pitch))
    print('beamline M4 ellipse p: ', beamLine.M4.p)
    print('beamline M4 ellipse q: ', beamLine.M4.q)
     
        

    beamLine.scrM4 = rscr.Screen(beamLine, name='screenM4', center=beamLine.M4.center)

    tmppitchM4 = pitchM4APXPS*np.pi/180
        
    displF = 0.
    beamLine.scrTgt11 = rscr.Screen(beamLine, name='screenTgt-0', center=[beamLine.M4.center[0]-(distM4TgtAPXPS+displF)*np.sin(2*(beamLine.M1.pitch + tmppitchM3+ tmppitchM4)),
                                                                    beamLine.M4.center[1]+(distM4TgtAPXPS+displF)*np.cos(2*(beamLine.M1.pitch + tmppitchM3+ tmppitchM4)),
                                                                    beamLine.M4.center[2]])
    beamLine.scrTgt11.dqs = np.linspace(-14, 14, 9)
    global tgtCenter
    tgtCenter = beamLine.M4.center[1]+(distM4TgtAPXPS+displF)*np.cos(2*(beamLine.M1.pitch + tmppitchM3+ tmppitchM4))

    print ('Tgt11, target 11 center: ', beamLine.scrTgt11.center)

    xaxis = [0.,0.,0.]
    xaxis[0] = beamLine.scrTgt11.center[0] - beamLine.M4.center[0]
    xaxis[1] = beamLine.scrTgt11.center[1] - beamLine.M4.center[1]
    xaxis[2] = beamLine.scrTgt11.center[2] - beamLine.M4.center[2]
    mag = np.sqrt(xaxis[0]**2+xaxis[1]**2+xaxis[2]**2)
    xaxisnorm = xaxis/mag

    axis = rotate_vector_around_axis(xaxisnorm,[0,0,1],sampleangle*np.pi/180)
    magaxis = np.sqrt(axis[0]**2+axis[1]**2+axis[2]**2)
    print ('sample axis, mag',axis,magaxis)

    beamLine.scrSample = rscr.Screen(beamLine, name='screenSample', center=beamLine.scrTgt11.center,x=[axis[0],axis[1],axis[2]],z=[0,0,1])
    beamLine.M4.center = [beamLine.M4.center[0]+M4lat,beamLine.M4.center[1],beamLine.M4.center[2]+M4vert]
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

    beamSampleScr = beamLine.scrSample.expose(beamM4g)
    outDict = {'beamSource': beamSource, 'beamM4Scr': beamM4Scr}

    for i, dq in enumerate(beamLine.scrTgt11.dqs):
       beamLine.scrTgt11.center[1] = tgtCenter + dq
       outDict['beamscrTgt_{0:02d}'.format(i)] = beamLine.scrTgt11.expose(beamM4g)

    #outDict = {'beamSource': beamSource, 'beamBDAScr': beamBDAScr,'beamM1g': beamM1g, 'beamM1Scr': beamM1Scr,
    #           'beamM2g': beamM2g, 'beamM2Scr': beamM2Scr, 'beamPGScr':beamPGScr,
    #           'beamM3Scr': beamM3Scr, 'beamSLDispScr': beamSLDispScr, 'beamSLScr': beamSLScr, 'beamM4Scr': beamM4Scr,
    #           'beamTgtScr11': beamTgtScr11,
    #           'beamSampleScr': beamSampleScr}

    return outDict



title_FWHM=branch+'_'+model+'_ES' # title for plot to pull FWHM values from at sample 

def define_plots(beamLine):
    
    plots = []

    
    # plot = xrtp.XYCPlot('beamSource')
    # plot.xaxis.fwhmFormatStr = '%.4f'
    # plot.yaxis.fwhmFormatStr = '%.4f'
    # plots.append(plot)
    
    # plotsSL = []

    
    # plot = xrtp.XYCPlot('beamBDAScr')
    # plot.xaxis.fwhmFormatStr = '%.4f'
    # plot.yaxis.fwhmFormatStr = '%.4f'
    # plots.append(plot)
    
    
    # plotsSL=[]
    
    
    

    
    # plot = xrtp.XYCPlot('beamSampleScr')
    # plot.xaxis.fwhmFormatStr = '%.4f'
    # plot.yaxis.fwhmFormatStr = '%.4f'
    # plots.append(plot)
    
    
    # plotsSL=[]
    
    
    
    # plot = xrtp.XYCPlot('beamTgtScr11')
    # plot.xaxis.fwhmFormatStr = '%.4f'
    # plot.yaxis.fwhmFormatStr = '%.4f'
    # plots.append(plot)
    
    
    plotsSL=[]
    bins=123
    limit=2
    for i, dq in enumerate(beamLine.scrTgt11.dqs):
        plot = xrtp.XYCPlot('beamscrTgt_{0:02d}'.format(i),
                                xaxis=xrtp.XYCAxis('$x$', 'mm',limits=None,bins=bins, ppb=2),
                                yaxis=xrtp.XYCAxis( '$z$', 'mm',limits=None,bins=bins, ppb=2))

        plot.xaxis.fwhmFormatStr = '%.4f'
        plot.yaxis.fwhmFormatStr = '%.4f'
        plots.append(plot)

    
    
    
    
    return plots, plotsSL






def main():
    """

    :rtype : none
    """
    print ('main')

    rr.run_process = run_process
    beamLine = build_beamLine(nrays=numrays)
    plots, plotsSL = define_plots(beamLine)
    #xrtr.run_ray_tracing(plots, generator=plot_generator, generatorArgs=[plots,plotsSL, beamLine],repeats=1,updateEvery=1,beamLine=beamLine)
    xrtr.run_ray_tracing(plots,repeats=10,updateEvery=10,beamLine=beamLine)
    for plot in plots:
        print(plot.cx,plot.cy, np.sqrt(np.square(plot.cx)+np.square(plot.cy)))


if __name__ == '__main__':
    main()
