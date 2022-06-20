"""
VERITAS beamline object.
For implementing RL environments and running the optimization algorithms

sets mirror randomly if parameters not defined in constructor
update_m4 updates the parameters of the M4 mirror
reset gives the mirror a new, random setting

Emil Winzell
April 2022
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
import cv2 as cv
#import gym
from scipy import optimize

import matplotlib as mpl
mpl.use('agg')

# Mirror classes...
class Grating(roes.OE):
    def local_g(self, x,y, rho=1200.0):
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
            #print (np.mean(b), np.std(b))
        if (self.meridionalSE==0):
            a = 0
        else:
            a = np.random.normal(0.0, self.meridionalSE, len(y))
            #print (np.mean(a),np.std(a))
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
            #print (np.mean(b),np.std(b))
        if (self.meridionalSE==0):
            a = 0
        else:
            a = np.random.normal(0.0, self.meridionalSE, len(y))
            #print (np.mean(a),np.std(a))

 
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
            #print (np.mean(b),np.std(b))
        if (self.meridionalSE==0):
            a = 0
        else:
            a = np.random.normal(0.0, self.meridionalSE, len(y))
            #print (np.mean(a),np.std(a))

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


# Element class
class VeritasSimpleBeamline(raycing.BeamLine):
    def __init__(self,nrays=6000, azimuth=0., height=0., alignE='auto', m4_params=None):
        super().__init__(azimuth, height, alignE)

        ### constants
        self.speedoflight = 299792458
        self.h = 4.135667e-15

        self.nrays = nrays
        self.E=400 #1612 max for harmonic 1
        self.sourcebandwidth = 0.001 # fractions
        self.harmonic = 1 #undulator harmonic
        self.ES_delta=0. #move ES position, note this *is* in isolation, it moves M4 and focus
        self.set_rough=0.3

        self.focus_position=0#adds extra distance onto focus position directly from nominal

        # Distances
        self.distM1BDA_ = 19000. ## MMA positions
        self.BDAhor = 2
        self.BDAvert = 2

        ## Grating setting
        self.cff = 2.25 ####
        self.m = 1.
        self.n = 1200.0
        self.d = 0.001/self.n
        self.exitheight = 20

        mAu = rm.Material('Au', rho=19.3)
        mRh = rm.Material('Rh', rho=12.41)
        mRh=mAu
        mAuGrating = rm.Material('Au', rho=19.3, kind='grating')

        # Build beamline
        self.EPU = rsrc.GeometricSource(self, name='EPU53', center=[0,0,0], 
                                        nrays=self.nrays, 
                                        distx='normal', dx=0.05586, disty=None, dy=0, distz='normal', dz=0.02508, #units mm's
                                        distxprime='normal', dxprime=20.6e-6, distzprime='normal', dzprime=20.0e-6, #units sigma, mm's
                                        distE='flat', 
                                        energies=(self.E-(self.sourcebandwidth*self.E),self.E+(self.sourcebandwidth*self.E) ), #units eV
                                        polarization='horizontal', 
                                        pitch=0, yaw=0)
        self.BDA = rap.RectangularAperture(self, 'BDA', center=[0,self.distM1BDA_, 0],
                                           kind=('top', 'bottom', 'left', 'right'), 
                                           opening=[self.BDAvert/2,-self.BDAvert/2,-self.BDAhor/2,self.BDAhor/2])    
        self.scrBDA = rscr.Screen(self, 'screenBDA', center=self.BDA.center)

        #M1 - toroid
        self.distSM1 = 24000. ## from source not BDA MMA
        m1r= 837.7
        m1R=1155969
        pitchM1 = 1.0
        roughM1 = self.set_rough*1e-9
        merSEM1 = np.radians(0.125/3600) 
        sagSEM1 = np.radians(0.71/3600)

        self.M1 = ToroidMirrorSE(self, name='M1',  r=m1r, R=m1R, positionRoll=-np.pi/2,
                                    center=[0,self.distSM1,0], pitch=pitchM1*np.pi/180, meridionalSE=merSEM1, 
                                    sagittalSE=sagSEM1, roughness=roughM1)

        self.scrM1 = rscr.Screen(self, name='screenM1', center=self.M1.center)
        #beamLine.M1.Y1=delta_Y1 #yaw scan

        #M2
        self.distM1BDA = 750.
        self.distBDAPG = 1560.-self.distM1BDA
        (pitchPG, pitchM2) = self.__calculate_cPGM()
        self.distBDAM2 = self.distBDAPG-self.exitheight/np.tan(2*pitchM2)
        merSEM2 = np.radians(0.05/3600)
        sagSEM2 = np.radians(0.033/3600)
        roughM2 = self.set_rough*1e-9
        self.M2 = OESE(self, 'M2', 
                        center=[self.M1.center[0]-(self.distBDAM2+self.distM1BDA)*np.sin(2*self.M1.pitch),
                                self.M1.center[1]+(self.distBDAM2+self.distM1BDA)*np.cos(2*self.M1.pitch),
                                self.M1.center[2]],
                        pitch=pitchM2, yaw=2*self.M1.pitch, positionRoll=0, roughness = roughM2,
                        sagittalSE = sagSEM2, meridionalSE = merSEM2)
        self.scrM2 = rscr.Screen(self, name='screenM2', center=self.M2.center)
        #Grating PGM
        self.PG = Grating(self, 'PG', 
                            center=[self.M2.center[0]-(self.distBDAPG-self.distBDAM2)*np.sin(2*self.M1.pitch),
                                    self.M2.center[1]+(self.distBDAPG-self.distBDAM2)*np.cos(2*self.M1.pitch),
                                    self.M2.center[2]+self.exitheight],
                            material=(mAuGrating,), positionRoll=np.pi, pitch=pitchPG, yaw=2*self.M1.pitch)
        self.PG.order = int(self.m)
        self.scrPG = rscr.Screen(self, name='screenPG', center=self.PG.center)

        #M3 - cylindrical
        m3r_b=706.9
        m3R_b=1.0E19
        self.distPGM3 = 2440. #from PG
        merSEM3APXPS = np.radians(0.024493/3600) #B_branch
        sagSEM3APXPS = np.radians(0.1/3600) #B_branch
        roughM3APXPS = self.set_rough*1e-9 #B_branch
        pitchM3APXPS = 1.5
        self.M3 = ToroidMirrorSE(self, 'M3',  r=m3r_b, R=m3R_b,
                                    center=[self.PG.center[0]-self.distPGM3*np.sin(2*self.M1.pitch),
                                            self.PG.center[1]+self.distPGM3*np.cos(2*self.M1.pitch),
                                            self.PG.center[2]],
                                    positionRoll=-np.pi/2, pitch=1*(2*pitchM1+pitchM3APXPS)*np.pi/180,
                                    meridionalSE=merSEM3APXPS, sagittalSE=sagSEM3APXPS, roughness=roughM3APXPS)
        #beamLine.M3.Y2=delta_Y2 #yaw scan
        tmppitchM3 = pitchM3APXPS*np.pi/180
        self.scrM3 = rscr.Screen(self, name='screenM3', center=self.M3.center)

        ##EXIT SLIT
        slitsizeX = 5
        slitsizeY=0.05 #mm ES vertical
        self.distM3SL_B = 13500.+self.ES_delta
        self.SL = rap.RectangularAperture(self, 'Slit', 
                                                center=[self.M3.center[0]-self.distM3SL_B*np.sin(2*self.M1.pitch+2*tmppitchM3),
                                                        self.M3.center[1]+self.distM3SL_B*np.cos(2*self.M1.pitch+2*tmppitchM3),
                                                        self.M3.center[2]],
                                                kind=['left', 'right', 'bottom', 'top'],
                                                opening=[-slitsizeX/2, slitsizeX/2, -slitsizeY/2, slitsizeY/2])
            
        self.SL.slit = slitsizeY 
        self.scrSL = rscr.Screen(self, name='screenSL', center=self.SL.center)

        #M4 elliptic
        self.distSLM4APXPS = 14500.-self.ES_delta #
        self.distM4TgtAPXPS = 700. +self.focus_position #
        pitchM4APXPS = 2.0
        
        merSEM4APXPS = np.radians(0.48/3600) #B_branch
        sagSEM4APXPS = np.radians(2.37/3600) #B_branch
        roughM4APXPS = self.set_rough*1e-9 #B_branch
        self.M4 = EllipticalMirrorParamSE(self, 'M4', 
                                                p=14500, 
                                                q=700,
                                                f1=self.SL.center,
                                                center=[self.SL.center[0]-self.distSLM4APXPS*np.sin(2*(self.M1.pitch+tmppitchM3)),
                                                        self.SL.center[1]+self.distSLM4APXPS*np.cos(2*(self.M1.pitch+tmppitchM3)),
                                                        self.SL.center[2]], positionRoll=-np.pi/2, 
                                                pitch=(2*pitchM1 + 2*pitchM3APXPS + pitchM4APXPS)*np.pi/180,
                                                meridionalSE=merSEM4APXPS, 
                                                sagittalSE=sagSEM4APXPS, 
                                                roughness=roughM4APXPS, )
        self.m4center = self.M4.center
        print('M4 center:' , self.m4center)
        self.scrM4 = rscr.Screen(self, name='screenM4', center=self.M4.center)
        tmppitchM4 = pitchM4APXPS*np.pi/180
        # Target
        displF = 0.
        self.scrTgt11 = rscr.Screen(self, name='screenTgt-0', 
                                        center=[self.m4center[0]-(self.distM4TgtAPXPS+displF)*np.sin(2*(self.M1.pitch + tmppitchM3+ tmppitchM4)),
                                                self.m4center[1]+(self.distM4TgtAPXPS+displF)*np.cos(2*(self.M1.pitch + tmppitchM3+ tmppitchM4)),
                                                self.m4center[2]])
        self.scrTgt11.dqs = np.linspace(-14, 14, 9)
        self.tgtCenter = self.m4center[1]+(self.distM4TgtAPXPS+displF)*np.cos(2*(self.M1.pitch + tmppitchM3+ tmppitchM4))
        print('VERITAS beamline initialized')

        self.bins = 256
        self.xz_lim = None
        self.repeats = 10
        self.plots = self.__define_plots(self.bins,self.xz_lim)

        # initialize M4 randomly
        self.p_lim=0.003
        self.r_lim=0.001
        self.y_lim=0.001
        self.l_lim=1.0
        self.v_lim=2.5

        if m4_params is None:
            self.M4pitch=np.random.uniform(-self.p_lim,self.p_lim)
            self.M4yaw=np.random.uniform(-self.y_lim,self.y_lim) #in radians, directly adds yaw to M4
            self.M4roll=np.random.uniform(-self.r_lim,self.r_lim)
            self.lateral = np.random.uniform(-self.l_lim,self.l_lim)
            self.vertical = np.random.uniform(-self.v_lim,self.v_lim)
        else:
            self.M4pitch = m4_params[0]
            self.M4yaw = m4_params[1]   
            self.M4roll = m4_params[2]
            self.lateral =  m4_params[3]
            self.vertical = m4_params[4]

        self.M4.extraPitch = self.M4pitch
        self.M4.extraYaw = self.M4yaw
        self.M4.extraRoll = self.M4roll
        self.M4.center = [self.m4center[0] + self.lateral,
                            self.m4center[1], 
                            self.m4center[2] + self.vertical]



    def __calculate_cPGM(self):
        lam = self.h*self.speedoflight/self.E
        a = 1-self.cff**2
        b = 2*(self.m*lam/self.d)*self.cff**2
        c = self.cff**2-1-((self.m*lam/self.d)**2)*(self.cff**2)

        sinbeta = (-b+np.sqrt(b*b-4*a*c))/(2*a)
        beta = np.arcsin(sinbeta)
        cosbeta = np.cos(beta)
        cosalpha = cosbeta/self.cff
        alpha = np.arccos(cosalpha)
        betagrazing = np.pi/2+beta
        alphagrazing = np.pi/2-alpha
        gammagrazing = 0.5*(alphagrazing+betagrazing)

        return -betagrazing,gammagrazing

    def __define_plots(self,bins,limit):
        #limits=[-limit+xlims[i],limit+xlims[i]],
        #limits=[-limit,limit],
        plots = []
        xlims = np.linspace(2.2,-2.2,9)
        for i, dq in enumerate(self.scrTgt11.dqs):
            plot = xrtp.XYCPlot('beamscrTgt_{0:02d}'.format(i),
                                    xaxis=xrtp.XYCAxis('$x$', 'mm',bins=bins, ppb=2),
                                    yaxis=xrtp.XYCAxis( '$z$', 'mm',bins=bins, ppb=2))
            

            plot.xaxis.fwhmFormatStr = '%.4f'
            plot.yaxis.fwhmFormatStr = '%.4f'
            plots.append(plot)

        return plots

    def run_process(self, shineOnly1stSource=False):
        beamSource = self.sources[0].shine()
        self.BDA.propagate(beamSource)
        beamBDAScr = self.scrBDA.expose(beamSource)
        beamM1g, beamM1l = self.M1.reflect(beamSource)
        beamM1Scr = self.scrM1.expose(beamM1g)
        beamM2g, beamM2l = self.M2.reflect(beamM1g)
        beamM2Scr = self.scrM2.expose(beamM2g)
        beamPGg, beamPGl = self.PG.reflect(beamM2g)
        beamPGScr = self.scrPG.expose(beamPGg)
        beamM3g, beamM3l = self.M3.reflect(beamPGg)
        beamM3Scr = self.scrM3.expose(beamM3g)
        beamSLDispScr = self.scrSL.expose(beamM3g)
        self.SL.propagate(beamM3g)
        beamSLScr = self.scrSL.expose(beamM3g)
        beamM4g, beamM4l = self.M4.reflect(beamM3g)
        beamM4Scr = self.scrM4.expose(beamM4g)
        beamTgtScr11 = self.scrTgt11.expose(beamM4g)
        outDict = {'beamSource': beamSource, 'beamM4Scr': beamM4Scr}
        
        for i, dq in enumerate(self.scrTgt11.dqs):
            self.scrTgt11.center[1] = self.tgtCenter + dq
            outDict['beamscrTgt_{0:02d}'.format(i)] = self.scrTgt11.expose(beamM4g)

        return outDict
    
    def update_m4(self,params):
        [pitch,yaw,roll,lateral,vertical] = params
        #limit = lambda n, lim: max(min(lim, n), -lim)

        self.M4.extraPitch = self.M4pitch+pitch #limit(self.M4pitch+pitch, self.p_lim)
        self.M4.extraYaw = self.M4yaw+yaw #limit(self.M4yaw+yaw, self.y_lim)  
        self.M4.extraRoll = self.M4roll+roll #limit(self.M4roll+roll, self.r_lim)
        self.M4.center = [self.m4center[0] + self.lateral + lateral,
                            self.m4center[1], 
                            self.m4center[2] + self.vertical + vertical]
        #print('Updated parameters')
        return
 
    def reset(self):
        self.M4yaw=np.random.uniform(-self.y_lim,self.y_lim) #in radians, directly adds yaw to M4
        self.M4roll=np.random.uniform(-self.r_lim,self.r_lim)
        self.M4pitch=np.random.uniform(-self.p_lim,self.p_lim)
        self.lateral= np.random.uniform(-self.l_lim,self.l_lim)
        self.vertical = np.random.uniform(-self.v_lim,self.v_lim)
        
        self.M4.extraPitch = self.M4pitch
        self.M4.extraRoll = self.M4roll
        self.M4.extraYaw = self.M4yaw
        self.M4.center = [self.m4center[0] + self.lateral,
                            self.m4center[1], 
                            self.m4center[2] + self.vertical]



