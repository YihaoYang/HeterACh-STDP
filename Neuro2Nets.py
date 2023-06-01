#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:26:24 2021

@author: yihaoy
"""


import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import signal
import numpy as np
import math
from numpy.fft import fft, ifft

    
# Constants
DEFAULT_NEURONUM = 1000
DEFAULT_NEURONUM_1 = 500
DEFAULT_NEURONUM_2 = 500
DEFAULT_TEND = 7000
DEFAULT_IDRIVE = 3

DEFAULT_XNUME = 20
DEFAULT_YNUME = 20
DEFAULT_XNUMI = 10
DEFAULT_YNUMI = 10
DEFAULT_DEGREE_EE = 40
DEFAULT_DEGREE_EI = 10
DEFAULT_DEGREE_IE = 400
DEFAULT_DEGREE_II = 100
DEFAULT_WEIGHT_EE = 0.01
DEFAULT_WEIGHT_EI = 0.05
DEFAULT_WEIGHT_IE = 0.02 #0.04
DEFAULT_WEIGHT_II = 0.02 #0.04

DEFAULT_TAU_E_SYN = 3 # ms
DEFAULT_TAU_I_SYN = 6 # ms
DEFAULT_GKS_MIN = 0.2
DEFAULT_GKS_MAX = 1.5

# Methods
def butter_bandpass(lowcut, highcut, fs, order = 2):
    sos = signal.butter(order, [lowcut, highcut], 
                        btype = 'bandpass', 
                        output='sos',fs = fs)
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order = 2):
    sos = butter_bandpass(lowcut, highcut, fs, order = order)
    y = signal.sosfilt(sos,data)
    return y



def periodic_corr(x, y):
    """Periodic correlation, implemented using the FFT.

    x and y must be real sequences with the same length.
    """
    return ifft(fft(x) * fft(y).conj()).real


# Class
class Neuro2Nets():
    
    def __init__(self,
                 neuroNum = DEFAULT_NEURONUM,
                 tEnd = DEFAULT_TEND,
                 Idrive = DEFAULT_IDRIVE,
                 tauSyn = DEFAULT_TAU_E_SYN,
                 gKsMin = DEFAULT_GKS_MIN,
                 gKsMax = DEFAULT_GKS_MAX):
        '''
        

        Parameters
        ----------
        neuroNum : TYPE, optional
            DESCRIPTION. The default is DEFAULT_NEURONUM.
        tEnd : TYPE, optional
            DESCRIPTION. The default is DEFAULT_TEND.
        Idrive : TYPE, optional
            DESCRIPTION. The default is DEFAULT_IDRIVE.
        tauSyn : TYPE, optional
            DESCRIPTION. The default is DEFAULT_TAU_SYN.

        Returns
        -------
        None.

        '''
        # simulation properties
        self.tEnd = tEnd # ms
        self.tStep = 0.05 # ms
        self.tPoints = np.arange(0,self.tEnd,self.tStep)
        # ensemble properties
        self.neuroNum = neuroNum
        self.Idrive = Idrive*np.ones(shape=(self.neuroNum,1))
        # neuronal properties
        self.gKsMin = gKsMin
        self.gKsMax = gKsMax
        self.randomInitialStates()
        self.gKs = self.gKsMax * np.ones((self.neuroNum,1))
        # initial adjMat
        self.adjMat = np.zeros(shape=(self.neuroNum,self.neuroNum))
        self.Esyn = np.zeros((self.neuroNum,1)) 
        # 0 mV for excitatory synapses;
        # -75mV for inhibitory synapses
        self.tauSyn = tauSyn*np.ones((self.neuroNum,1)) # ms
        
        self.STDPstep = 1
        
    def randomInitialStates(self):
        self.states = np.random.rand(self.neuroNum,4)
        self.states[:,3] = -70 + 40 * self.states[:,3]
        return self
        
    def mexicanHat(self,
                 xNumE = DEFAULT_XNUME,
                 yNumE = DEFAULT_YNUME,
                 xNumI = DEFAULT_XNUMI,
                 yNumI = DEFAULT_YNUMI,
                 degreeEE = DEFAULT_DEGREE_EE,
                 degreeEI = DEFAULT_DEGREE_EI,
                 degreeIE = DEFAULT_DEGREE_IE,
                 degreeII = DEFAULT_DEGREE_II,
                 weightEE = DEFAULT_WEIGHT_EE,
                 weightEI = DEFAULT_WEIGHT_EI,
                 weightIE = DEFAULT_WEIGHT_IE,
                 weightII = DEFAULT_WEIGHT_II,
                 tauI = DEFAULT_TAU_I_SYN,
                 tauE = DEFAULT_TAU_E_SYN):
        '''
        

        Parameters
        ----------
        xNumE : TYPE, optional
            DESCRIPTION. The default is DEFAULT_XNUME.
        yNumE : TYPE, optional
            DESCRIPTION. The default is DEFAULT_YNUME.
        xNumI : TYPE, optional
            DESCRIPTION. The default is DEFAULT_XNUMI.
        yNumI : TYPE, optional
            DESCRIPTION. The default is DEFAULT_YNUMI.
        degreeEE : TYPE, optional
            DESCRIPTION. The default is DEFAULT_DEGREE_EE.
        degreeEI : TYPE, optional
            DESCRIPTION. The default is DEFAULT_DEGREE_EI.
        weightEE : TYPE, optional
            DESCRIPTION. The default is DEFAULT_WEIGHT_EE.
        weightEI : TYPE, optional
            DESCRIPTION. The default is DEFAULT_WEIGHT_EI.
        weightIE : TYPE, optional
            DESCRIPTION. The default is DEFAULT_WEIGHT_IE.
        weightII : TYPE, optional
            DESCRIPTION. The default is DEFAULT_WEIGHT_II.

        Returns
        -------
        None.

        '''
        
        self.numE = xNumE * yNumE
        self.xNumE,self.yNumE = xNumE,yNumE
        
        self.numI = xNumI * yNumI
        self.xNumI,self.yNumI = xNumI,yNumI
        
        if 2*self.numI != (self.neuroNum - 2 * self.numE):
            print('ERROR!!')
        
        self.tauSyn[-self.numI:,:] = tauI
        self.tauSyn[self.numE:(self.numI+self.numE),:] = tauI
        
        self.Esyn[-self.numI:,:] = -75 # mV for I-cells in 2nd module
        self.Esyn[self.numE:(self.numI+self.numE),:] = -75 # mV for I-cells in 1st module
        
        # assign x, y coordinates
        xLocE = np.arange(xNumE) + 0.5 # + 0.5 for periodic condition
        yLocE = np.arange(yNumE) + 0.5
        xLocE,yLocE = np.meshgrid(xLocE,yLocE)
        
        self.coordsE = np.stack((xLocE.reshape(-1),yLocE.reshape(-1))).T
        
        xLocI = (np.arange(xNumI) + 0.5) * (xNumE / xNumI)
        yLocI = (np.arange(yNumI) + 0.5) * (yNumE / yNumI)
        xLocI,yLocI = np.meshgrid(xLocI,yLocI)
        
        self.coordsI = np.stack((xLocI.reshape(-1),yLocI.reshape(-1))).T
        
        # compute mexican-hat adjacency matrix
        # compute distance matrices       
        distEE = distance.cdist(self.coordsE,self.coordsE,
                                lambda a,b: self.computeDist(a,b))
        distEI = distance.cdist(self.coordsI,self.coordsE,
                                lambda a,b: self.computeDist(a,b))
        self.distEE = distEE
        self.distEI = distEI
        
        # compute adjEE and adjEI 
        if degreeEE >= self.numE:
            adjMatEE = weightEE * np.ones(shape = (self.numE,self.numE))
        else:
            adjMatEE = np.zeros(shape = (self.numE,self.numE))
    
            adjMatEE[
                np.argsort(distEE,axis = 0)[1:degreeEE+1,:].T.reshape(-1),
                np.concatenate(
                    [i*np.ones(degreeEE,dtype=int) for i in np.arange(self.numE)])
                ] = weightEE
            
        if degreeEI >= self.numI:
            adjMatEI = weightEI * np.ones(shape = (self.numI,self.numE))
        else:
            adjMatEI = np.zeros(shape = (self.numI,self.numE))
            adjMatEI[
                np.argsort(distEI,axis = 0)[:degreeEI,:].T.reshape(-1),
                np.concatenate(
                    [i*np.ones(degreeEI,dtype=int) for i in np.arange(self.numE)])
                ] = weightEI
        
        # compute adjIE and adjII: all to all connection if degree < # of cells
        if degreeIE >= self.numE:
            adjMatIE = weightIE * np.ones(shape = (self.numE,self.numI))
        else:
            distIE = distance.cdist(self.coordsE,self.coordsI,
                                    lambda a,b: self.computeDist(a, b))
            adjMatIE = np.zeros(shape = (self.numE,self.numI))
            adjMatIE[
                np.argsort(distIE,axis=0)[:degreeIE,:].T.reshape(-1),
                np.concatenate(
                    [i*np.ones(degreeIE,dtype=int) for i in np.arange(self.numI)])
                ] = weightIE
            
        if degreeII >= self.numI:
            adjMatII = weightII * np.ones(shape = (self.numI,self.numI))
        else:
            distII = distance.cdist(self.coordsI,self.coordsI,
                                    lambda a,b: self.computeDist(a,b))
            adjMatII = np.zeros(shape = (self.numI,self.numI))

            adjMatII[
                np.argsort(distII,axis = 0)[1:degreeII+1,:].T.reshape(-1),
                np.concatenate(
                    [i*np.ones(degreeII,dtype=int) for i in np.arange(self.numI)])
                ] = weightII
        
        # finally get the adjMat
        adjMatModule = np.vstack((np.hstack((adjMatEE,adjMatIE)),
                                  np.hstack((adjMatEI,adjMatII))))
        
        self.adjMat = np.vstack((np.hstack((adjMatModule, np.zeros((adjMatModule.shape)))),
                                 np.hstack((np.zeros((adjMatModule.shape)),adjMatModule))))
        return self
    
    def randG2Nets(self,indegree,w): # assuming no connections between two modules 
        # random connections from module 2 to module 1
        for arr in self.adjMat[:self.numE,DEFAULT_NEURONUM_1:(DEFAULT_NEURONUM_1+self.numE)]:
            arr[np.random.choice(arr.size, indegree,replace=False)] = w
        
        # from 1 to 2
        for arr in self.adjMat[DEFAULT_NEURONUM_1:(DEFAULT_NEURONUM_1+self.numE),:self.numE]:
            arr[np.random.choice(arr.size, indegree,replace=False)] = w
        
        return self
    
    def nn2Nets(self,indegree,w): # assuming no connections between two modules 
        
        adjMatEE = np.zeros(shape = (self.numE,self.numE))
    
        adjMatEE[
                np.argsort(self.distEE,axis = 0)[:indegree,:].T.reshape(-1),
                np.concatenate(
                    [i*np.ones(indegree,dtype=int) for i in np.arange(self.numE)])
                ] = w
        
        #  connections from module 2 to module 1
        self.adjMat[:self.numE,DEFAULT_NEURONUM_1:(DEFAULT_NEURONUM_1+self.numE)] = adjMatEE
        # from 1 to 2
        self.adjMat[DEFAULT_NEURONUM_1:(DEFAULT_NEURONUM_1+self.numE),:self.numE] = adjMatEE
        
        return self
    
    # compute the euclidean distance with periodic boundary conditions
    def computeDist(self,a,b):
        bounds = np.array([self.xNumE,self.yNumE])
        delta = np.abs(a-b)
        delta = np.where(delta > 0.5 * bounds,delta - bounds,delta)
        return np.sqrt((delta ** 2).sum(axis = -1)) 
    
    def mapGks(self, 
               r_1 = 4, gKsMin_1 = DEFAULT_GKS_MIN, gKsMax_1 = DEFAULT_GKS_MAX,
               releaseLocs_1 = np.array([[0.25,0.25],[0.75,0.75]]),
               r_2 = 4, gKsMin_2 = DEFAULT_GKS_MIN, gKsMax_2 = DEFAULT_GKS_MAX,
               releaseLocs_2 = np.array([[0.25,0.25],[0.75,0.75]]),
               sharpness = 2):
        '''
        Parameters
        ----------
        releaseLocs : TYPE, optional
            DESCRIPTION. The default is np.array([]). Normalized by x,y ranges.

        Returns
        -------
        None.

        '''        
        sigmoid = lambda x: 1/(1 + np.exp(-x))
    
        
        if releaseLocs_1.size > 0:
    
            self.releaseR_1 = r_1
            self.coordsRelease_1 = np.array([self.xNumE,self.yNumE]) * releaseLocs_1      
        
            distER = (distance.cdist(self.coordsRelease_1,self.coordsE,
                                         lambda a,b: self.computeDist(a,b))
                                         .min(axis=0).reshape(-1,1))
            distIR = (distance.cdist(self.coordsRelease_1,self.coordsI,
                                         lambda a,b: self.computeDist(a,b))
                                         .min(axis=0).reshape(-1,1))  
            distToR = np.vstack((distER,distIR))
            self.distToR_1 = distToR
            
            # self.sigmoidDistToR = sigmoidDistToR
            # sigmoidDistToR -= sigmoidDistToR.min()
            gKs_1 = gKsMin_1 + sigmoid(sharpness*(distToR - r_1)) * (
                gKsMax_1 - gKsMin_1) 
        else:
            gKs_1 = gKsMax_1 * np.ones((DEFAULT_NEURONUM_1,1))
            
        if releaseLocs_2.size > 0:
    
            self.coordsRelease_2 = np.array([self.xNumE,self.yNumE]) * releaseLocs_2      
        
            distER = (distance.cdist(self.coordsRelease_2,self.coordsE,
                                         lambda a,b: self.computeDist(a,b))
                                         .min(axis=0).reshape(-1,1))
            distIR = (distance.cdist(self.coordsRelease_2,self.coordsI,
                                         lambda a,b: self.computeDist(a,b))
                                         .min(axis=0).reshape(-1,1))  
            distToR = np.vstack((distER,distIR))
            self.distToR_2 = distToR
            
            # self.sigmoidDistToR = sigmoidDistToR
            # sigmoidDistToR -= sigmoidDistToR.min()
            gKs_2 = gKsMin_2 + sigmoid(sharpness*(distToR - r_2)) * (
                gKsMax_2 - gKsMin_2) 
        else:
            gKs_2 = gKsMax_2 * np.ones((DEFAULT_NEURONUM_2,1))
        
        self.gKs = np.vstack((gKs_1, gKs_2))
        return self
    
    def setupSquareWaveDrive(self,
                             ts_m1 = 0, te_m1 = 7000, 
                             r_1 = 4, a_1 = 0.25, b_1 = 0, w_1 = 150,
                             ts_m2 = 50, te_m2 = 7000,
                             r_2 = 4, a_2 = 0.25, b_2 = 0, w_2 = 150,
                             releaseLocs_1 = np.array([[0.25,0.25],[0.75,0.75]]),
                             releaseLocs_2 = np.array([[0.25,0.25],[0.75,0.75]])
                             ):
        
        self.ts_m1 = ts_m1
        self.te_m1 = te_m1
        self.a_1 = a_1
        self.b_1 = b_1
        self.w_1 = w_1
        
        self.ts_m2 = ts_m2
        self.te_m2 = te_m2
        self.a_2 = a_2
        self.b_2 = b_2
        self.w_2 = w_2
        
        self.releaseR_1 = r_1
        self.coordsRelease_1 = np.array([self.xNumE,self.yNumE]) * releaseLocs_1      
    
        distER = distance.cdist(self.coordsRelease_1,self.coordsE,
                                     lambda a,b: self.computeDist(a,b))
        
        distIR = distance.cdist(self.coordsRelease_1,self.coordsI,
                                     lambda a,b: self.computeDist(a,b))
                                     
        distToR = np.hstack((distER,distIR))
        
        self.distToR_1 = distToR
        
        sqwave_idvec_1 = np.zeros((DEFAULT_NEURONUM_1,1),dtype=int)
        sqwave_idvec_1[distToR[0,:].reshape(-1,1) <= r_1] = 11
        sqwave_idvec_1[distToR[1,:].reshape(-1,1) <= r_1] = 12
        
        self.releaseR_2 = r_2
        self.coordsRelease_2 = np.array([self.xNumE,self.yNumE]) * releaseLocs_2      
    
        distER = distance.cdist(self.coordsRelease_2,self.coordsE,
                                     lambda a,b: self.computeDist(a,b))
        
        distIR = distance.cdist(self.coordsRelease_2,self.coordsI,
                                     lambda a,b: self.computeDist(a,b))
                                     
        distToR = np.hstack((distER,distIR))
        
        self.distToR_2 = distToR
        
        sqwave_idvec_2 = np.zeros((DEFAULT_NEURONUM_2,1),dtype=int)
        sqwave_idvec_2[distToR[0,:].reshape(-1,1) <= r_2] = 21
        sqwave_idvec_2[distToR[1,:].reshape(-1,1) <= r_2] = 22
        
        self.sqwave_idvec = np.vstack((sqwave_idvec_1,sqwave_idvec_2))
        
        return self

    def setupSquareWaveDriveOneSpot(self,
                             ts_m1 = 0, te_m1 = 7000, 
                             r_1 = 4, a_1 = 0.25, b_1 = 0, w_1 = 150,
                             ts_m2 = 50, te_m2 = 7000,
                             r_2 = 4, a_2 = 0.25, b_2 = 0, w_2 = 150,
                             releaseLocs_1 = np.array([[0.5,0.5]]),
                             releaseLocs_2 = np.array([[0.5,0.5]])
                             ):
        
        self.ts_m1 = ts_m1
        self.te_m1 = te_m1
        self.a_1 = a_1
        self.b_1 = b_1
        self.w_1 = w_1
        
        self.ts_m2 = ts_m2
        self.te_m2 = te_m2
        self.a_2 = a_2
        self.b_2 = b_2
        self.w_2 = w_2
        
        self.releaseR_1 = r_1
        self.coordsRelease_1 = np.array([self.xNumE,self.yNumE]) * releaseLocs_1      
    
        distER = distance.cdist(self.coordsRelease_1,self.coordsE,
                                     lambda a,b: self.computeDist(a,b))
        
        distIR = distance.cdist(self.coordsRelease_1,self.coordsI,
                                     lambda a,b: self.computeDist(a,b))
                                     
        distToR = np.hstack((distER,distIR))
        
        self.distToR_1 = distToR
        
        sqwave_idvec_1 = 10*np.ones((DEFAULT_NEURONUM_1,1),dtype=int)
        sqwave_idvec_1[distToR[0,:].reshape(-1,1) <= r_1] = 11
        
        
        self.releaseR_2 = r_2
        self.coordsRelease_2 = np.array([self.xNumE,self.yNumE]) * releaseLocs_2      
    
        distER = distance.cdist(self.coordsRelease_2,self.coordsE,
                                     lambda a,b: self.computeDist(a,b))
        
        distIR = distance.cdist(self.coordsRelease_2,self.coordsI,
                                     lambda a,b: self.computeDist(a,b))
                                     
        distToR = np.hstack((distER,distIR))
        
        self.distToR_2 = distToR
        
        sqwave_idvec_2 = 20*np.ones((DEFAULT_NEURONUM_2,1),dtype=int)
        sqwave_idvec_2[distToR[0,:].reshape(-1,1) <= r_2] = 21
        
        
        self.sqwave_idvec = np.vstack((sqwave_idvec_1,sqwave_idvec_2))
        
        return self
    
    def mapIdrive(self, 
               r_1 = 4, dcStim_1 = 3, dcBase_1 = 3,
               releaseLocs_1 = np.array([]),
               r_2 = 4, dcStim_2 = 3, dcBase_2 = 3,
               releaseLocs_2 = np.array([])):
        '''
        Parameters
        ----------
        releaseLocs : TYPE, optional
            DESCRIPTION. The default is np.array([]). Normalized by x,y ranges.

        Returns
        -------
        None.

        '''        
      
        
        if releaseLocs_1.size > 0:
    
            self.releaseR_1 = r_1
            self.coordsRelease_1 = np.array([self.xNumE,self.yNumE]) * releaseLocs_1      
        
            distER = (distance.cdist(self.coordsRelease_1,self.coordsE,
                                         lambda a,b: self.computeDist(a,b))
                                         .min(axis=0).reshape(-1,1))
            distIR = (distance.cdist(self.coordsRelease_1,self.coordsI,
                                         lambda a,b: self.computeDist(a,b))
                                         .min(axis=0).reshape(-1,1))  
            distToR = np.vstack((distER,distIR))
            self.distToR_1 = distToR
            
            # self.sigmoidDistToR = sigmoidDistToR
            # sigmoidDistToR -= sigmoidDistToR.min()
            dc_1 = dcBase_1 * np.ones((DEFAULT_NEURONUM_1,1))
            dc_1[distToR <= r_1] = dcStim_1
        else:
            dc_1 = dcBase_1 * np.ones((DEFAULT_NEURONUM_1,1))
            
        if releaseLocs_2.size > 0:
    
            self.coordsRelease_2 = np.array([self.xNumE,self.yNumE]) * releaseLocs_2      
        
            distER = (distance.cdist(self.coordsRelease_2,self.coordsE,
                                         lambda a,b: self.computeDist(a,b))
                                         .min(axis=0).reshape(-1,1))
            distIR = (distance.cdist(self.coordsRelease_2,self.coordsI,
                                         lambda a,b: self.computeDist(a,b))
                                         .min(axis=0).reshape(-1,1))  
            distToR = np.vstack((distER,distIR))
            self.distToR_2 = distToR
            
            # self.sigmoidDistToR = sigmoidDistToR
            # sigmoidDistToR -= sigmoidDistToR.min()
            dc_2 = dcBase_2 * np.ones((DEFAULT_NEURONUM_2,1))
            dc_2[distToR <= r_2] = dcStim_2
        else:
            dc_2 = dcBase_2 * np.ones((DEFAULT_NEURONUM_2,1))
        
        self.Idrive = np.vstack((dc_1, dc_2))
        return self    
    
    def computeModMPC(self, s=4000, e=5000, sigma = 1): # generate traces from spike times
        self.E1SpikeTimes = self.spikeTimes[0,(self.spikeTimes[0,:]>s) & 
                                    (self.spikeTimes[0,:]<=e) &
                                    (self.spikeTimes[1,:]<self.numE)]
        self.E2SpikeTimes = self.spikeTimes[0,(self.spikeTimes[0,:]>s) & 
                                    (self.spikeTimes[0,:]<=e) &
                                    (self.spikeTimes[1,:]>=((self.numI+self.numE))) &
                                    (self.spikeTimes[1,:]<(self.neuroNum-self.numI))]
        self.timePoints = self.tPoints[(self.tPoints>s) & (self.tPoints<=e)]
        self.E1SpikeTraces = np.exp(-(0.5/sigma**2)*(self.E1SpikeTimes.reshape(-1,1) - self.timePoints)**2).mean(axis=0)
        self.E2SpikeTraces = np.exp(-(0.5/sigma**2)*(self.E2SpikeTimes.reshape(-1,1) - self.timePoints)**2).mean(axis=0)
        self.E1Bursts, _ = signal.find_peaks(self.E1SpikeTraces, prominence=0.003)
        self.E2Bursts, _ = signal.find_peaks(self.E2SpikeTraces, prominence=0.003)
        self.E1BurstsWithInd = np.vstack((self.E1Bursts, 
                                    np.ones(self.E1Bursts.size, dtype=int)))
        self.E2BurstsWithInd = np.vstack((self.E2Bursts, 
                                    2*np.ones(self.E2Bursts.size, dtype=int)))
        self.EBurstsWithInd = np.hstack((self.E1BurstsWithInd,
                                                  self.E2BurstsWithInd))
        sort_ind = np.argsort(self.EBurstsWithInd[0,:])
        self.EBurstsWithInd = self.EBurstsWithInd[:,sort_ind]
        # diff = np.diff(self.EBurstsWithInd)
        mpc12 = []
        mpc21 = []
        for k in range(1,sort_ind.size-1):
            timing, mod_ind = self.EBurstsWithInd[:,k]            
            for j in range(k-1,-1,-1):
                t_other_k, other_k_ind = self.EBurstsWithInd[:,j]
                if other_k_ind != mod_ind: break
            else: 
                continue
            for j in range(k+1, sort_ind.size):
                t_other_kp1, other_kp1_ind = self.EBurstsWithInd[:,j]
                if other_kp1_ind != mod_ind: break
            else:
                break
            if mod_ind == 2:
                mpc12.append(2j*np.pi*(timing-t_other_k)/(t_other_kp1-t_other_k))
            else: 
                mpc21.append(2j*np.pi*(timing-t_other_k)/(t_other_kp1-t_other_k))
        self.MPC12 = np.abs(np.exp(mpc12).mean())
        self.MPC21 = np.abs(np.exp(mpc21).mean())        
        
        
        
        # _, self.C12 = signal.coherence(self.E1SpikeTraces,self.E2SpikeTraces)
        # _, self.C11 = signal.coherence(self.E1SpikeTraces)
        # _, self.C22 = signal.coherence(self.E2SpikeTraces)
        # self.ModMPC = self.C12*self.C12/(self.C11*self.C22)
    
    def runSimulation(self, 
                      isNet = True, 
                      isSTDP = False, 
                      externalInput = False,
                      ex_drive_strength = 0.1,
                      poisson_noise = False,
                      poisson_rate = 1/200,
                      poisson_amp = 6,    # poisson_Eonly=False,
                      logV = False,
                      sqwave = False):        
        
        THRESHOLD_AP = -20 # mV
        C = 1 # uf/cm2
        v_Na = 55.0 # mV
        v_K = -90 # mV
        v_L = -60 # mV
        g_Na = 24 # mS/cm2
        g_Kdr = 3.0 # mS/cm2
        g_L = 0.02 # mS/cm2

        spikeTimes = np.zeros((self.neuroNum,self.tEnd))          
        spikeCounts = np.zeros((self.neuroNum,1),dtype=int)     
        # vPoints = np.zeros(size(tPoints));

        channelZ = self.states[:,[0]]
        channelH = self.states[:,[1]]
        channelN = self.states[:,[2]]
        memV = self.states[:,[3]]
        
        if logV: 
            logCounter = 0
            self.vPoints = np.zeros(shape=(self.neuroNum,self.tPoints.size))
            # temp current logger
            self.iPoints = np.zeros(shape=(self.neuroNum,self.tPoints.size))
        
        colIdx = np.arange(4)
        neuroIdx = np.arange(self.neuroNum).reshape(-1,1)
        Itotal = self.Idrive
        STDPon = False
        STDPoff = False
        windowIsyn = 20 # ms
        
        ### external input ###
        if externalInput:
            distToRs = []
            for releaseId in range(self.num_external_input):
                distER = (distance.cdist(self.coordsRelease[[releaseId],:],self.coordsE,
                                         lambda a,b: self.computeDist(a,b))
                                         .reshape(-1,1))
                
                distIR = (distance.cdist(self.coordsRelease[[releaseId],:],self.coordsI,
                                         lambda a,b: self.computeDist(a,b))
                                         .reshape(-1,1))  
                distToRs.append(np.vstack((distER,
                                           100*np.ones(shape=distIR.shape))))
                # self.Idrive = DEFAULT_IDRIVE*np.ones(shape=(self.neuroNum,1))
                self.Idrive[distToRs[releaseId]<self.releaseR] = (1+ex_drive_strength) * self.Idrive.min()
        
        ### square wave ###
        if sqwave:
            Isqwave = np.zeros(shape = self.Idrive.shape)
            ts_sqwave = min(self.ts_m1, self.ts_m2)
            te_sqwave = max(self.te_m1, self.te_m2)
        
        ### poisson noise ###        
        if poisson_noise:
            poissonRate = poisson_rate #s-1
            poissonKickAmp = poisson_amp
            poissonKickDur = 1    
            Ipoisson = 0
            
        # ### temp current logger
        # self.meanItotal = 0

        for t in self.tPoints:     
            if logV: 
                self.vPoints[:,[logCounter]] = memV
                self.iPoints[:,[logCounter]] = Itotal
                logCounter += 1
            
            # determine synI vector (for sub class NeuroNet) 
            # and record spike times
            isFiring = (memV < THRESHOLD_AP)
            if isNet:
                EsynMat,memVMat = np.meshgrid(self.Esyn,memV)
                expTerm = np.zeros(shape = (self.neuroNum,1))
                ithLatestSpike = 1
                deltaTs = t - spikeTimes[neuroIdx,spikeCounts-ithLatestSpike]
                while ((deltaTs<windowIsyn) & (spikeCounts>ithLatestSpike)).any():                            
                    expTerm += ((deltaTs < windowIsyn) & 
                                (spikeCounts>ithLatestSpike)) * np.exp(
                                -deltaTs /self.tauSyn)
                    ithLatestSpike += 1 
                    deltaTs = t-spikeTimes[neuroIdx,spikeCounts-ithLatestSpike]
                
                Isyn =self.adjMat * (memVMat - EsynMat) @ expTerm
                Itotal = self.Idrive - Isyn
            
        ### square wave signal calculation ###
            if sqwave and ts_sqwave<=t<=te_sqwave:
                if t>self.ts_m1:
                    if t>self.te_m1:
                        Isqwave[self.sqwave_idvec == 12] = 0
                        Isqwave[self.sqwave_idvec == 11] = 0
                    elif (math.floor((t-self.ts_m1)/self.w_1)%2):# odd means 2nd spot up
                        Isqwave[self.sqwave_idvec == 12] = self.b_1 + self.a_1
                        Isqwave[self.sqwave_idvec == 11] = self.b_1 - self.a_1
                    else:
                        Isqwave[self.sqwave_idvec == 11] = self.b_1 + self.a_1
                        Isqwave[self.sqwave_idvec == 12] = self.b_1 - self.a_1
                if t>self.ts_m2:
                    if t>self.te_m2:
                        Isqwave[self.sqwave_idvec == 22] = 0
                        Isqwave[self.sqwave_idvec == 21] = 0
                    elif (math.floor((t-self.ts_m2)/self.w_2)%2):# odd means 2nd spot up
                        Isqwave[self.sqwave_idvec == 22] = self.b_2 + self.a_2
                        Isqwave[self.sqwave_idvec == 21] = self.b_2 - self.a_2
                    else:
                        Isqwave[self.sqwave_idvec == 21] = self.b_2 + self.a_2
                        Isqwave[self.sqwave_idvec == 22] = self.b_2 - self.a_2
                Itotal += Isqwave
                # ### temp current logger
                # self.meanItotal += Itotal
        ### poisson noise ###    
            if poisson_noise:
                if not t%poissonKickDur:
                    Ipoisson = poissonKickAmp * (np.random.rand(self.neuroNum,1)<poissonRate)
                    # if poisson_Eonly:
                    #     Ipoisson[self.numE:(self.numE+self.numI)] = 0
                    #     Ipoisson[(self.neuroNum-self.numI):] = 0
                Itotal += Ipoisson 
                
            # RK4 method
            kV = np.tile(memV,4)
            kZ = np.tile(channelZ,4)
            kH = np.tile(channelH,4)
            kN = np.tile(channelN,4)
            for colInd in colIdx:
                mInf = 1 / (1 + np.exp((-kV[:,[colInd]]-30.0)/9.5))
                hInf = 1 / (1 + np.exp((kV[:,[colInd]]+53.0)/7.0))
                nInf = 1 / (1 + np.exp((-kV[:,[colInd]]-30.0)/10))
                zInf = 1 / (1 + np.exp((-kV[:,[colInd]]-39.0)/5.0))
                hTau = 0.37 + 2.78 / (1 + np.exp((kV[:,[colInd]]+40.5)/6))
                nTau = 0.37 + 1.85 / (1 + np.exp((kV[:,[colInd]]+27.0)/15))
                fh = (hInf - kH[:,[colInd]]) / hTau
                fn = (nInf - kN[:,[colInd]]) / nTau
                fz = (zInf - kZ[:,[colInd]]) / 75.0
                fv = (1/C)*(g_Na*(mInf**3) * kH[:,[colInd]] *  
                    (v_Na-kV[:,[colInd]]) + 
                    g_Kdr*(kN[:,[colInd]]**4) * (v_K - kV[:,[colInd]])+ 
                    self.gKs * kZ[:,[colInd]] * (v_K - kV[:,[colInd]])+ 
                    g_L*(v_L-kV[:,[colInd]]) + Itotal)
                kH[:,[colInd]] = self.tStep*fh
                kN[:,[colInd]] = self.tStep*fn
                kZ[:,[colInd]] = self.tStep*fz
                kV[:,[colInd]] = self.tStep*fv
                if colInd == 0 or colInd == 1:
                    kH[:,[colInd+1]] = kH[:,[colInd+1]] + 0.5*kH[:,[colInd]]
                    kN[:,[colInd+1]] = kN[:,[colInd+1]] + 0.5*kN[:,[colInd]]
                    kZ[:,[colInd+1]] = kZ[:,[colInd+1]] + 0.5*kZ[:,[colInd]]
                    kV[:,[colInd+1]] = kV[:,[colInd+1]] + 0.5*kV[:,[colInd]]
                elif colInd == 2:
                    kH[:,[colInd+1]] = kH[:,[colInd+1]] + kH[:,[colInd]]
                    kN[:,[colInd+1]] = kN[:,[colInd+1]] + kN[:,[colInd]]
                    kZ[:,[colInd+1]] = kZ[:,[colInd+1]] + kZ[:,[colInd]]
                    kV[:,[colInd+1]] = kV[:,[colInd+1]] + kV[:,[colInd]]
            memV =     memV +     (kV[:,[0]] + 2 * kV[:,[1]] + 
                                   2 * kV[:,[2]] + kV[:,[3]])/6.0
            channelH = channelH + (kH[:,[0]] + 2 * kH[:,[1]] + 
                                   2 * kH[:,[2]] + kH[:,[3]])/6.0
            channelN = channelN + (kN[:,[0]] + 2 * kN[:,[1]] + 
                                   2 * kN[:,[2]] + kN[:,[3]])/6.0
            channelZ = channelZ + (kZ[:,[0]] + 2 * kZ[:,[1]] + 
                                   2 * kZ[:,[2]] + kZ[:,[3]])/6.0
            # RK4 ends
            isFiring &= (memV > THRESHOLD_AP)   
        
        ### STDP part ###       
            # when STDP turned on, initialize adjMat_max,A+, A-, tau+,tau- etc.
            if STDPon: # if STDP rule is taking place
                if not STDPoff: 
                # if STDP has already been turned off, nothing should be done
                    # STDP rule taking effect here! 
                    if isFiring.any(): 
                    # only change weights when at least one cell is firing
                    # This if statement can not combine with above one 
                    # to make sure keep track of time to turn off STDP
                        # iteration for get all the terms 
                        # within cutoff STDP time window    
                        
                        ithLatestSpike = 1
                        deltaTs = t-spikeTimes[neuroIdx,spikeCounts-ithLatestSpike]
                        # if spikeCounts is zeros then -1 index leads to time at 0
                        # deltaWeights = 0
                        deltaWeightsPlus,deltaWeightsMinus = 0,0
                        
                        
                        ### nearest spike
                        deltaWeightsPlus += (deltaTs < windowSTDP) * np.exp(
                                                            -deltaTs / tauPlus)
                        deltaWeightsMinus += (deltaTs < windowSTDP) * np.exp(
                                                            -deltaTs / tauMinus) * 0.5
                        
                        
                        # STDPAdjMat[idxPostSyn[(isFiring&depressions)[:numPostSyn]],:] -= 
                        # STDP from module 1 to 2
                        MatIdxPost_12 = np.arange(STDPAdjMat_12.shape[0]).reshape(-1,1)
                        MatIdxPre_12 = np.arange(STDPAdjMat_12.shape[1]).reshape(-1,1)
                        
                        STDPAdjMat_12[MatIdxPost_12[isFiring[idxPostSyn_12.flatten()]],:] += (
                            deltaWeightConst_12[MatIdxPost_12[isFiring[idxPostSyn_12.flatten()]],:]
                            * deltaWeightsPlus[idxPreSyn_12.flatten()].T)
                        
                        STDPAdjMat_12[:,MatIdxPre_12[isFiring[idxPreSyn_12.flatten()]]] -= (
                            deltaWeightConst_12[:,MatIdxPre_12[isFiring[idxPreSyn_12.flatten()]]]
                            * deltaWeightsMinus[idxPostSyn_12.flatten()]) 
                        
                        # STDP from module 2 to 1
                        MatIdxPost_21 = np.arange(STDPAdjMat_21.shape[0]).reshape(-1,1)
                        MatIdxPre_21 = np.arange(STDPAdjMat_21.shape[1]).reshape(-1,1)
                        
                        STDPAdjMat_21[MatIdxPost_21[isFiring[idxPostSyn_21.flatten()]],:] += (
                            deltaWeightConst_21[MatIdxPost_21[isFiring[idxPostSyn_21.flatten()]],:]
                            * deltaWeightsPlus[idxPreSyn_21.flatten()].T)
                        
                        STDPAdjMat_21[:,MatIdxPre_21[isFiring[idxPreSyn_21.flatten()]]] -= (
                            deltaWeightConst_21[:,MatIdxPre_21[isFiring[idxPreSyn_21.flatten()]]]
                            * deltaWeightsMinus[idxPostSyn_21.flatten()]) 

                        

                        # make sure weights in [0,weightmax]
                        STDPAdjMat_12[STDPAdjMat_12>STDPAdjMatMax_12] = STDPAdjMatMax_12[
                            STDPAdjMat_12>STDPAdjMatMax_12]
                        STDPAdjMat_12[STDPAdjMat_12<STDPAdjMatMin_12] = STDPAdjMatMin_12[
                            STDPAdjMat_12<STDPAdjMatMin_12]   
                        
                        STDPAdjMat_21[STDPAdjMat_21>STDPAdjMatMax_21] = STDPAdjMatMax_21[
                            STDPAdjMat_21>STDPAdjMatMax_21]
                        STDPAdjMat_21[STDPAdjMat_21<STDPAdjMatMin_21] = STDPAdjMatMin_21[
                            STDPAdjMat_21<STDPAdjMatMin_21] 
                        # STDP update done!            
                    if t>self.tSTDP_off: # time to turn off STDP rule
                        STDPoff = True
            elif isSTDP and t>self.tSTDP_on: # turn on STDP at the right time
                STDPon = True
                # initialize important STDP parameters
                
                idxPreSyn_21 = np.arange(DEFAULT_NEURONUM_1, (DEFAULT_NEURONUM_1+self.numE)).reshape(-1,1)
                idxPostSyn_21 = np.arange(self.numE).reshape(-1,1)
                
                STDPAdjMat_21 = self.adjMat0[:self.numE,DEFAULT_NEURONUM_1:(DEFAULT_NEURONUM_1+self.numE)].copy()
  
                STDPAdjMatMax_21 = STDPAdjMat_21* (1 + self.STDPlevel)
                STDPAdjMatMin_21 = STDPAdjMat_21 * 0 #(1 - self.STDPlevel)                
                deltaWeightConst_21 = STDPAdjMat_21 * self.STDPstep * self.STDPlevel/20.0       
                STDPAdjMat_21 = self.adjMat[:self.numE,DEFAULT_NEURONUM_1:(DEFAULT_NEURONUM_1+self.numE)]
                
                idxPreSyn_12 = np.arange(self.numE).reshape(-1,1)
                idxPostSyn_12 = np.arange(DEFAULT_NEURONUM_1, (DEFAULT_NEURONUM_1+self.numE)).reshape(-1,1)
                
                
                STDPAdjMat_12 = self.adjMat0[DEFAULT_NEURONUM_1:(DEFAULT_NEURONUM_1+self.numE),:self.numE].copy()
                
                STDPAdjMatMax_12 = STDPAdjMat_12* (1 + self.STDPlevel)
                STDPAdjMatMin_12 = STDPAdjMat_12 * 0 #(1 - self.STDPlevel)                
                deltaWeightConst_12 = STDPAdjMat_12 * self.STDPstep * self.STDPlevel/20.0       
                STDPAdjMat_12 = self.adjMat[DEFAULT_NEURONUM_1:(DEFAULT_NEURONUM_1+self.numE),:self.numE]
                
                tauSTDP = 10 # ms
                # assymetrical STDP learning rule
                tauPlus = 14 # ms
                tauMinus = 34 #ms
                
                windowSTDP = 100 # ms
                
                 
            spikeTimes[neuroIdx[isFiring],spikeCounts[isFiring]] = t
            spikeCounts += isFiring
        # main simulation over

        # compress spikeTimes to a 2D array        
        timingVec = np.concatenate(
            [spikeTimes[i,:spikeCounts[i,0]] for i in neuroIdx.reshape(-1)])
        idVec = np.concatenate(
            [i*np.ones(spikeCounts[i,0]) for i in neuroIdx.reshape(-1)])
        self.spikeTimes = np.stack((timingVec,idVec))
        self.spikeCounts = spikeCounts
        if not isNet: self.states = np.hstack((channelZ,channelH,channelN,memV))
        return self
        # return spikeCounts    

    def runSimulation_symSTDP(self, 
                      isNet = True, 
                      isSTDP = False, 
                      externalInput = False,
                      ex_drive_strength = 0.1,
                      poisson_noise = False,
                      poisson_rate = 1/150,
                      poisson_amp = 6,
                      logV = False,
                      sqwave = False):        
        
        THRESHOLD_AP = -20 # mV
        C = 1 # uf/cm2
        v_Na = 55.0 # mV
        v_K = -90 # mV
        v_L = -60 # mV
        g_Na = 24 # mS/cm2
        g_Kdr = 3.0 # mS/cm2
        g_L = 0.02 # mS/cm2

        spikeTimes = np.zeros((self.neuroNum,self.tEnd))          
        spikeCounts = np.zeros((self.neuroNum,1),dtype=int)     

        channelZ = self.states[:,[0]]
        channelH = self.states[:,[1]]
        channelN = self.states[:,[2]]
        memV = self.states[:,[3]]
        
        if logV: 
            logCounter = 0
            self.vPoints = np.zeros(shape=(self.neuroNum,self.tPoints.size))
            # temp current logger
            self.iPoints = np.zeros(shape=(self.neuroNum,self.tPoints.size))
        
        colIdx = np.arange(4)
        neuroIdx = np.arange(self.neuroNum).reshape(-1,1)
        Itotal = self.Idrive
        STDPon = False
        STDPoff = False
        windowIsyn = 20 # ms
        
        ### external input ###
        if externalInput:
            distToRs = []
            for releaseId in range(self.num_external_input):
                distER = (distance.cdist(self.coordsRelease[[releaseId],:],self.coordsE,
                                         lambda a,b: self.computeDist(a,b))
                                         .reshape(-1,1))
                
                distIR = (distance.cdist(self.coordsRelease[[releaseId],:],self.coordsI,
                                         lambda a,b: self.computeDist(a,b))
                                         .reshape(-1,1))  
                distToRs.append(np.vstack((distER,
                                           100*np.ones(shape=distIR.shape))))
                # self.Idrive = DEFAULT_IDRIVE*np.ones(shape=(self.neuroNum,1))
                self.Idrive[distToRs[releaseId]<self.releaseR] = (1+ex_drive_strength) * self.Idrive.min()
        
        ### square wave ###
        if sqwave:
            Isqwave = np.zeros(shape = self.Idrive.shape)
            ts_sqwave = min(self.ts_m1, self.ts_m2)
            te_sqwave = max(self.te_m1, self.te_m2)
        
        ### poisson noise ###        
        if poisson_noise:
            poissonRate = poisson_rate #s-1
            poissonKickAmp = poisson_amp
            poissonKickDur = 1    
            Ipoisson = 0
            
        # ### temp current logger
        # self.meanItotal = 0

        for t in self.tPoints:     
            if logV: 
                self.vPoints[:,[logCounter]] = memV
                self.iPoints[:,[logCounter]] = Itotal
                logCounter += 1
            
            # determine synI vector (for sub class NeuroNet) 
            # and record spike times
            isFiring = (memV < THRESHOLD_AP)
            if isNet:
                EsynMat,memVMat = np.meshgrid(self.Esyn,memV)
                expTerm = np.zeros(shape = (self.neuroNum,1))
                ithLatestSpike = 1
                deltaTs = t - spikeTimes[neuroIdx,spikeCounts-ithLatestSpike]
                while ((deltaTs<windowIsyn) & (spikeCounts>ithLatestSpike)).any():                            
                    expTerm += ((deltaTs < windowIsyn) & 
                                (spikeCounts>ithLatestSpike)) * np.exp(
                                -deltaTs /self.tauSyn)
                    ithLatestSpike += 1 
                    deltaTs = t-spikeTimes[neuroIdx,spikeCounts-ithLatestSpike]
                
                Isyn =self.adjMat * (memVMat - EsynMat) @ expTerm
                Itotal = self.Idrive - Isyn
            
        ### square wave signal calculation ###
            if sqwave and ts_sqwave<=t<=te_sqwave:
                if t>self.ts_m1:
                    if t>self.te_m1:
                        Isqwave[self.sqwave_idvec == 12] = 0
                        Isqwave[self.sqwave_idvec == 11] = 0
                    elif (math.floor((t-self.ts_m1)/self.w_1)%2):# odd means 2nd spot up
                        Isqwave[self.sqwave_idvec == 12] = self.b_1 + self.a_1
                        Isqwave[self.sqwave_idvec == 11] = self.b_1 - self.a_1
                    else:
                        Isqwave[self.sqwave_idvec == 11] = self.b_1 + self.a_1
                        Isqwave[self.sqwave_idvec == 12] = self.b_1 - self.a_1
                if t>self.ts_m2:
                    if t>self.te_m2:
                        Isqwave[self.sqwave_idvec == 22] = 0
                        Isqwave[self.sqwave_idvec == 21] = 0
                    elif (math.floor((t-self.ts_m2)/self.w_2)%2):# odd means 2nd spot up
                        Isqwave[self.sqwave_idvec == 22] = self.b_2 + self.a_2
                        Isqwave[self.sqwave_idvec == 21] = self.b_2 - self.a_2
                    else:
                        Isqwave[self.sqwave_idvec == 21] = self.b_2 + self.a_2
                        Isqwave[self.sqwave_idvec == 22] = self.b_2 - self.a_2
                Itotal += Isqwave
                # ### temp current logger
                # self.meanItotal += Itotal
        ### poisson noise ###    
            if poisson_noise:
                if not t%poissonKickDur:
                    Ipoisson = poissonKickAmp * (np.random.rand(self.neuroNum,1)<poissonRate)
                Itotal += Ipoisson 
                
            # RK4 method
            kV = np.tile(memV,4)
            kZ = np.tile(channelZ,4)
            kH = np.tile(channelH,4)
            kN = np.tile(channelN,4)
            for colInd in colIdx:
                mInf = 1 / (1 + np.exp((-kV[:,[colInd]]-30.0)/9.5))
                hInf = 1 / (1 + np.exp((kV[:,[colInd]]+53.0)/7.0))
                nInf = 1 / (1 + np.exp((-kV[:,[colInd]]-30.0)/10))
                zInf = 1 / (1 + np.exp((-kV[:,[colInd]]-39.0)/5.0))
                hTau = 0.37 + 2.78 / (1 + np.exp((kV[:,[colInd]]+40.5)/6))
                nTau = 0.37 + 1.85 / (1 + np.exp((kV[:,[colInd]]+27.0)/15))
                fh = (hInf - kH[:,[colInd]]) / hTau
                fn = (nInf - kN[:,[colInd]]) / nTau
                fz = (zInf - kZ[:,[colInd]]) / 75.0
                fv = (1/C)*(g_Na*(mInf**3) * kH[:,[colInd]] *  
                    (v_Na-kV[:,[colInd]]) + 
                    g_Kdr*(kN[:,[colInd]]**4) * (v_K - kV[:,[colInd]])+ 
                    self.gKs * kZ[:,[colInd]] * (v_K - kV[:,[colInd]])+ 
                    g_L*(v_L-kV[:,[colInd]]) + Itotal)
                kH[:,[colInd]] = self.tStep*fh
                kN[:,[colInd]] = self.tStep*fn
                kZ[:,[colInd]] = self.tStep*fz
                kV[:,[colInd]] = self.tStep*fv
                if colInd == 0 or colInd == 1:
                    kH[:,[colInd+1]] = kH[:,[colInd+1]] + 0.5*kH[:,[colInd]]
                    kN[:,[colInd+1]] = kN[:,[colInd+1]] + 0.5*kN[:,[colInd]]
                    kZ[:,[colInd+1]] = kZ[:,[colInd+1]] + 0.5*kZ[:,[colInd]]
                    kV[:,[colInd+1]] = kV[:,[colInd+1]] + 0.5*kV[:,[colInd]]
                elif colInd == 2:
                    kH[:,[colInd+1]] = kH[:,[colInd+1]] + kH[:,[colInd]]
                    kN[:,[colInd+1]] = kN[:,[colInd+1]] + kN[:,[colInd]]
                    kZ[:,[colInd+1]] = kZ[:,[colInd+1]] + kZ[:,[colInd]]
                    kV[:,[colInd+1]] = kV[:,[colInd+1]] + kV[:,[colInd]]
            memV =     memV +     (kV[:,[0]] + 2 * kV[:,[1]] + 
                                   2 * kV[:,[2]] + kV[:,[3]])/6.0
            channelH = channelH + (kH[:,[0]] + 2 * kH[:,[1]] + 
                                   2 * kH[:,[2]] + kH[:,[3]])/6.0
            channelN = channelN + (kN[:,[0]] + 2 * kN[:,[1]] + 
                                   2 * kN[:,[2]] + kN[:,[3]])/6.0
            channelZ = channelZ + (kZ[:,[0]] + 2 * kZ[:,[1]] + 
                                   2 * kZ[:,[2]] + kZ[:,[3]])/6.0
            # RK4 ends
            isFiring &= (memV > THRESHOLD_AP)   
        
        ### STDP part ###       
            # when STDP turned on, initialize adjMat_max,A+, A-, tau+,tau- etc.
            if STDPon: # if STDP rule is taking place
                if not STDPoff: 
                # if STDP has already been turned off, nothing should be done
                    # STDP rule taking effect here! 
                    if isFiring.any(): 
                    # only change weights when at least one cell is firing
                    # This if statement can not combine with above one 
                    # to make sure keep track of time to turn off STDP
                        # iteration for get all the terms 
                        # within cutoff STDP time window    
                        
                        ithLatestSpike = 1
                        deltaTs = t-spikeTimes[neuroIdx,spikeCounts-ithLatestSpike]
                        # if spikeCounts is zeros then -1 index leads to time at 0
                        # deltaWeights = 0
                        deltaWeightsPlus,deltaWeightsMinus = 0,0
                        
                        
                        ### nearest spike
                        deltaWeightsPlus += (deltaTs < windowSTDP) * np.exp(
                                                            -deltaTs / tauPlus)
                        deltaWeightsMinus += (deltaTs < windowSTDP) * np.exp(
                                                            -deltaTs / tauMinus)
                        
                        
                        # STDPAdjMat[idxPostSyn[(isFiring&depressions)[:numPostSyn]],:] -= 
                        # STDP from module 1 to 2
                        MatIdxPost_12 = np.arange(STDPAdjMat_12.shape[0]).reshape(-1,1)
                        MatIdxPre_12 = np.arange(STDPAdjMat_12.shape[1]).reshape(-1,1)
                        
                        STDPAdjMat_12[MatIdxPost_12[isFiring[idxPostSyn_12.flatten()]],:] += (
                            deltaWeightConst_12[MatIdxPost_12[isFiring[idxPostSyn_12.flatten()]],:]
                            * deltaWeightsPlus[idxPreSyn_12.flatten()].T)
                        
                        STDPAdjMat_12[:,MatIdxPre_12[isFiring[idxPreSyn_12.flatten()]]] -= (
                            deltaWeightConst_12[:,MatIdxPre_12[isFiring[idxPreSyn_12.flatten()]]]
                            * deltaWeightsMinus[idxPostSyn_12.flatten()]) 
                        
                        # STDP from module 2 to 1
                        MatIdxPost_21 = np.arange(STDPAdjMat_21.shape[0]).reshape(-1,1)
                        MatIdxPre_21 = np.arange(STDPAdjMat_21.shape[1]).reshape(-1,1)
                        
                        STDPAdjMat_21[MatIdxPost_21[isFiring[idxPostSyn_21.flatten()]],:] += (
                            deltaWeightConst_21[MatIdxPost_21[isFiring[idxPostSyn_21.flatten()]],:]
                            * deltaWeightsPlus[idxPreSyn_21.flatten()].T)
                        
                        STDPAdjMat_21[:,MatIdxPre_21[isFiring[idxPreSyn_21.flatten()]]] -= (
                            deltaWeightConst_21[:,MatIdxPre_21[isFiring[idxPreSyn_21.flatten()]]]
                            * deltaWeightsMinus[idxPostSyn_21.flatten()]) 

                        

                        # make sure weights in [0,weightmax]
                        STDPAdjMat_12[STDPAdjMat_12>STDPAdjMatMax_12] = STDPAdjMatMax_12[
                            STDPAdjMat_12>STDPAdjMatMax_12]
                        STDPAdjMat_12[STDPAdjMat_12<STDPAdjMatMin_12] = STDPAdjMatMin_12[
                            STDPAdjMat_12<STDPAdjMatMin_12]   
                        
                        STDPAdjMat_21[STDPAdjMat_21>STDPAdjMatMax_21] = STDPAdjMatMax_21[
                            STDPAdjMat_21>STDPAdjMatMax_21]
                        STDPAdjMat_21[STDPAdjMat_21<STDPAdjMatMin_21] = STDPAdjMatMin_21[
                            STDPAdjMat_21<STDPAdjMatMin_21] 
                        # STDP update done!            
                    if t>self.tSTDP_off: # time to turn off STDP rule
                        STDPoff = True
            elif isSTDP and t>self.tSTDP_on: # turn on STDP at the right time
                STDPon = True
                # initialize important STDP parameters
                
                idxPreSyn_21 = np.arange(DEFAULT_NEURONUM_1, (DEFAULT_NEURONUM_1+self.numE)).reshape(-1,1)
                idxPostSyn_21 = np.arange(self.numE).reshape(-1,1)
                
                STDPAdjMat_21 = self.adjMat[:self.numE,DEFAULT_NEURONUM_1:(DEFAULT_NEURONUM_1+self.numE)].copy()
  
                STDPAdjMatMax_21 = STDPAdjMat_21* (1 + self.STDPlevel)
                STDPAdjMatMin_21 = STDPAdjMat_21 * 0 #(1 - self.STDPlevel)                
                deltaWeightConst_21 = STDPAdjMat_21 * self.STDPstep * self.STDPlevel/20.0       
                STDPAdjMat_21 = self.adjMat[:self.numE,DEFAULT_NEURONUM_1:(DEFAULT_NEURONUM_1+self.numE)]
                
                idxPreSyn_12 = np.arange(self.numE).reshape(-1,1)
                idxPostSyn_12 = np.arange(DEFAULT_NEURONUM_1, (DEFAULT_NEURONUM_1+self.numE)).reshape(-1,1)
                
                
                STDPAdjMat_12 = self.adjMat[DEFAULT_NEURONUM_1:(DEFAULT_NEURONUM_1+self.numE),:self.numE].copy()
                
                STDPAdjMatMax_12 = STDPAdjMat_12* (1 + self.STDPlevel)
                STDPAdjMatMin_12 = STDPAdjMat_12 * 0 #(1 - self.STDPlevel)                
                deltaWeightConst_12 = STDPAdjMat_12 * self.STDPstep * self.STDPlevel/20.0       
                STDPAdjMat_12 = self.adjMat[DEFAULT_NEURONUM_1:(DEFAULT_NEURONUM_1+self.numE),:self.numE]
                
                tauSTDP = 10 # ms
                # assymetrical STDP learning rule
                tauPlus = 14 # ms
                tauMinus = 14 #ms
                
                windowSTDP = 100 # ms
                
                 
            spikeTimes[neuroIdx[isFiring],spikeCounts[isFiring]] = t
            spikeCounts += isFiring
        # main simulation over

        # compress spikeTimes to a 2D array        
        timingVec = np.concatenate(
            [spikeTimes[i,:spikeCounts[i,0]] for i in neuroIdx.reshape(-1)])
        idVec = np.concatenate(
            [i*np.ones(spikeCounts[i,0]) for i in neuroIdx.reshape(-1)])
        self.spikeTimes = np.stack((timingVec,idVec))
        self.spikeCounts = spikeCounts
        if not isNet: self.states = np.hstack((channelZ,channelH,channelN,memV))
        return self
        # return spikeCounts    

    def runSimulation_revSTDP(self, 
                      isNet = True, 
                      isSTDP = False, 
                      externalInput = False,
                      ex_drive_strength = 0.1,
                      poisson_noise = False,
                      poisson_rate = 1/200,
                      poisson_amp = 6,
                      logV = False,
                      sqwave = False):        
        
        THRESHOLD_AP = -20 # mV
        C = 1 # uf/cm2
        v_Na = 55.0 # mV
        v_K = -90 # mV
        v_L = -60 # mV
        g_Na = 24 # mS/cm2
        g_Kdr = 3.0 # mS/cm2
        g_L = 0.02 # mS/cm2

        spikeTimes = np.zeros((self.neuroNum,self.tEnd))          
        spikeCounts = np.zeros((self.neuroNum,1),dtype=int)     
        # vPoints = np.zeros(size(tPoints));

        channelZ = self.states[:,[0]]
        channelH = self.states[:,[1]]
        channelN = self.states[:,[2]]
        memV = self.states[:,[3]]
        
        if logV: 
            logCounter = 0
            self.vPoints = np.zeros(shape=(self.neuroNum,self.tPoints.size))
            # temp current logger
            self.iPoints = np.zeros(shape=(self.neuroNum,self.tPoints.size))
        
        colIdx = np.arange(4)
        neuroIdx = np.arange(self.neuroNum).reshape(-1,1)
        Itotal = self.Idrive
        STDPon = False
        STDPoff = False
        windowIsyn = 20 # ms
        
        ### external input ###
        if externalInput:
            distToRs = []
            for releaseId in range(self.num_external_input):
                distER = (distance.cdist(self.coordsRelease[[releaseId],:],self.coordsE,
                                         lambda a,b: self.computeDist(a,b))
                                         .reshape(-1,1))
                
                distIR = (distance.cdist(self.coordsRelease[[releaseId],:],self.coordsI,
                                         lambda a,b: self.computeDist(a,b))
                                         .reshape(-1,1))  
                distToRs.append(np.vstack((distER,
                                           100*np.ones(shape=distIR.shape))))
                # self.Idrive = DEFAULT_IDRIVE*np.ones(shape=(self.neuroNum,1))
                self.Idrive[distToRs[releaseId]<self.releaseR] = (1+ex_drive_strength) * self.Idrive.min()
        
        ### square wave ###
        if sqwave:
            Isqwave = np.zeros(shape = self.Idrive.shape)
            ts_sqwave = min(self.ts_m1, self.ts_m2)
            te_sqwave = max(self.te_m1, self.te_m2)
        
        ### poisson noise ###        
        if poisson_noise:
            poissonRate = poisson_rate #s-1
            poissonKickAmp = poisson_amp
            poissonKickDur = 1    
            Ipoisson = 0
            
        # ### temp current logger
        # self.meanItotal = 0

        for t in self.tPoints:     
            if logV: 
                self.vPoints[:,[logCounter]] = memV
                self.iPoints[:,[logCounter]] = Itotal
                logCounter += 1
            
            # determine synI vector (for sub class NeuroNet) 
            # and record spike times
            isFiring = (memV < THRESHOLD_AP)
            if isNet:
                EsynMat,memVMat = np.meshgrid(self.Esyn,memV)
                expTerm = np.zeros(shape = (self.neuroNum,1))
                ithLatestSpike = 1
                deltaTs = t - spikeTimes[neuroIdx,spikeCounts-ithLatestSpike]
                while ((deltaTs<windowIsyn) & (spikeCounts>ithLatestSpike)).any():                            
                    expTerm += ((deltaTs < windowIsyn) & 
                                (spikeCounts>ithLatestSpike)) * np.exp(
                                -deltaTs /self.tauSyn)
                    ithLatestSpike += 1 
                    deltaTs = t-spikeTimes[neuroIdx,spikeCounts-ithLatestSpike]
                
                Isyn =self.adjMat * (memVMat - EsynMat) @ expTerm
                Itotal = self.Idrive - Isyn
            
        ### square wave signal calculation ###
            if sqwave and ts_sqwave<=t<=te_sqwave:
                if t>self.ts_m1:
                    if t>self.te_m1:
                        Isqwave[self.sqwave_idvec == 12] = 0
                        Isqwave[self.sqwave_idvec == 11] = 0
                    elif (math.floor((t-self.ts_m1)/self.w_1)%2):# odd means 2nd spot up
                        Isqwave[self.sqwave_idvec == 12] = self.b_1 + self.a_1
                        Isqwave[self.sqwave_idvec == 11] = self.b_1 - self.a_1
                    else:
                        Isqwave[self.sqwave_idvec == 11] = self.b_1 + self.a_1
                        Isqwave[self.sqwave_idvec == 12] = self.b_1 - self.a_1
                if t>self.ts_m2:
                    if t>self.te_m2:
                        Isqwave[self.sqwave_idvec == 22] = 0
                        Isqwave[self.sqwave_idvec == 21] = 0
                    elif (math.floor((t-self.ts_m2)/self.w_2)%2):# odd means 2nd spot up
                        Isqwave[self.sqwave_idvec == 22] = self.b_2 + self.a_2
                        Isqwave[self.sqwave_idvec == 21] = self.b_2 - self.a_2
                    else:
                        Isqwave[self.sqwave_idvec == 21] = self.b_2 + self.a_2
                        Isqwave[self.sqwave_idvec == 22] = self.b_2 - self.a_2
                Itotal += Isqwave
                # ### temp current logger
                # self.meanItotal += Itotal
        ### poisson noise ###    
            if poisson_noise:
                if not t%poissonKickDur:
                    Ipoisson = poissonKickAmp * (np.random.rand(self.neuroNum,1)<poissonRate)
                Itotal += Ipoisson 
                
            # RK4 method
            kV = np.tile(memV,4)
            kZ = np.tile(channelZ,4)
            kH = np.tile(channelH,4)
            kN = np.tile(channelN,4)
            for colInd in colIdx:
                mInf = 1 / (1 + np.exp((-kV[:,[colInd]]-30.0)/9.5))
                hInf = 1 / (1 + np.exp((kV[:,[colInd]]+53.0)/7.0))
                nInf = 1 / (1 + np.exp((-kV[:,[colInd]]-30.0)/10))
                zInf = 1 / (1 + np.exp((-kV[:,[colInd]]-39.0)/5.0))
                hTau = 0.37 + 2.78 / (1 + np.exp((kV[:,[colInd]]+40.5)/6))
                nTau = 0.37 + 1.85 / (1 + np.exp((kV[:,[colInd]]+27.0)/15))
                fh = (hInf - kH[:,[colInd]]) / hTau
                fn = (nInf - kN[:,[colInd]]) / nTau
                fz = (zInf - kZ[:,[colInd]]) / 75.0
                fv = (1/C)*(g_Na*(mInf**3) * kH[:,[colInd]] *  
                    (v_Na-kV[:,[colInd]]) + 
                    g_Kdr*(kN[:,[colInd]]**4) * (v_K - kV[:,[colInd]])+ 
                    self.gKs * kZ[:,[colInd]] * (v_K - kV[:,[colInd]])+ 
                    g_L*(v_L-kV[:,[colInd]]) + Itotal)
                kH[:,[colInd]] = self.tStep*fh
                kN[:,[colInd]] = self.tStep*fn
                kZ[:,[colInd]] = self.tStep*fz
                kV[:,[colInd]] = self.tStep*fv
                if colInd == 0 or colInd == 1:
                    kH[:,[colInd+1]] = kH[:,[colInd+1]] + 0.5*kH[:,[colInd]]
                    kN[:,[colInd+1]] = kN[:,[colInd+1]] + 0.5*kN[:,[colInd]]
                    kZ[:,[colInd+1]] = kZ[:,[colInd+1]] + 0.5*kZ[:,[colInd]]
                    kV[:,[colInd+1]] = kV[:,[colInd+1]] + 0.5*kV[:,[colInd]]
                elif colInd == 2:
                    kH[:,[colInd+1]] = kH[:,[colInd+1]] + kH[:,[colInd]]
                    kN[:,[colInd+1]] = kN[:,[colInd+1]] + kN[:,[colInd]]
                    kZ[:,[colInd+1]] = kZ[:,[colInd+1]] + kZ[:,[colInd]]
                    kV[:,[colInd+1]] = kV[:,[colInd+1]] + kV[:,[colInd]]
            memV =     memV +     (kV[:,[0]] + 2 * kV[:,[1]] + 
                                   2 * kV[:,[2]] + kV[:,[3]])/6.0
            channelH = channelH + (kH[:,[0]] + 2 * kH[:,[1]] + 
                                   2 * kH[:,[2]] + kH[:,[3]])/6.0
            channelN = channelN + (kN[:,[0]] + 2 * kN[:,[1]] + 
                                   2 * kN[:,[2]] + kN[:,[3]])/6.0
            channelZ = channelZ + (kZ[:,[0]] + 2 * kZ[:,[1]] + 
                                   2 * kZ[:,[2]] + kZ[:,[3]])/6.0
            # RK4 ends
            isFiring &= (memV > THRESHOLD_AP)   
        
        ### STDP part ###       
            # when STDP turned on, initialize adjMat_max,A+, A-, tau+,tau- etc.
            if STDPon: # if STDP rule is taking place
                if not STDPoff: 
                # if STDP has already been turned off, nothing should be done
                    # STDP rule taking effect here! 
                    if isFiring.any(): 
                    # only change weights when at least one cell is firing
                    # This if statement can not combine with above one 
                    # to make sure keep track of time to turn off STDP
                        # iteration for get all the terms 
                        # within cutoff STDP time window    
                        
                        ithLatestSpike = 1
                        deltaTs = t-spikeTimes[neuroIdx,spikeCounts-ithLatestSpike]
                        # if spikeCounts is zeros then -1 index leads to time at 0
                        # deltaWeights = 0
                        deltaWeightsPlus,deltaWeightsMinus = 0,0
                        
                        
                        ### nearest spike
                        deltaWeightsPlus += (deltaTs < windowSTDP) * np.exp(
                                                            -deltaTs / tauPlus) * 0.5
                        deltaWeightsMinus += (deltaTs < windowSTDP) * np.exp(
                                                            -deltaTs / tauMinus) 
                        
                        
                        # STDPAdjMat[idxPostSyn[(isFiring&depressions)[:numPostSyn]],:] -= 
                        # STDP from module 1 to 2
                        MatIdxPost_12 = np.arange(STDPAdjMat_12.shape[0]).reshape(-1,1)
                        MatIdxPre_12 = np.arange(STDPAdjMat_12.shape[1]).reshape(-1,1)
                        
                        STDPAdjMat_12[MatIdxPost_12[isFiring[idxPostSyn_12.flatten()]],:] += (
                            deltaWeightConst_12[MatIdxPost_12[isFiring[idxPostSyn_12.flatten()]],:]
                            * deltaWeightsPlus[idxPreSyn_12.flatten()].T)
                        
                        STDPAdjMat_12[:,MatIdxPre_12[isFiring[idxPreSyn_12.flatten()]]] -= (
                            deltaWeightConst_12[:,MatIdxPre_12[isFiring[idxPreSyn_12.flatten()]]]
                            * deltaWeightsMinus[idxPostSyn_12.flatten()]) 
                        
                        # STDP from module 2 to 1
                        MatIdxPost_21 = np.arange(STDPAdjMat_21.shape[0]).reshape(-1,1)
                        MatIdxPre_21 = np.arange(STDPAdjMat_21.shape[1]).reshape(-1,1)
                        
                        STDPAdjMat_21[MatIdxPost_21[isFiring[idxPostSyn_21.flatten()]],:] += (
                            deltaWeightConst_21[MatIdxPost_21[isFiring[idxPostSyn_21.flatten()]],:]
                            * deltaWeightsPlus[idxPreSyn_21.flatten()].T)
                        
                        STDPAdjMat_21[:,MatIdxPre_21[isFiring[idxPreSyn_21.flatten()]]] -= (
                            deltaWeightConst_21[:,MatIdxPre_21[isFiring[idxPreSyn_21.flatten()]]]
                            * deltaWeightsMinus[idxPostSyn_21.flatten()]) 

                        

                        # make sure weights in [0,weightmax]
                        STDPAdjMat_12[STDPAdjMat_12>STDPAdjMatMax_12] = STDPAdjMatMax_12[
                            STDPAdjMat_12>STDPAdjMatMax_12]
                        STDPAdjMat_12[STDPAdjMat_12<STDPAdjMatMin_12] = STDPAdjMatMin_12[
                            STDPAdjMat_12<STDPAdjMatMin_12]   
                        
                        STDPAdjMat_21[STDPAdjMat_21>STDPAdjMatMax_21] = STDPAdjMatMax_21[
                            STDPAdjMat_21>STDPAdjMatMax_21]
                        STDPAdjMat_21[STDPAdjMat_21<STDPAdjMatMin_21] = STDPAdjMatMin_21[
                            STDPAdjMat_21<STDPAdjMatMin_21] 
                        # STDP update done!            
                    if t>self.tSTDP_off: # time to turn off STDP rule
                        STDPoff = True
            elif isSTDP and t>self.tSTDP_on: # turn on STDP at the right time
                STDPon = True
                # initialize important STDP parameters
                
                idxPreSyn_21 = np.arange(DEFAULT_NEURONUM_1, (DEFAULT_NEURONUM_1+self.numE)).reshape(-1,1)
                idxPostSyn_21 = np.arange(self.numE).reshape(-1,1)
                
                STDPAdjMat_21 = self.adjMat[:self.numE,DEFAULT_NEURONUM_1:(DEFAULT_NEURONUM_1+self.numE)].copy()
  
                STDPAdjMatMax_21 = STDPAdjMat_21* (1 + self.STDPlevel)
                STDPAdjMatMin_21 = STDPAdjMat_21 * 0 #(1 - self.STDPlevel)                
                deltaWeightConst_21 = STDPAdjMat_21 * self.STDPstep * self.STDPlevel/20.0       
                STDPAdjMat_21 = self.adjMat[:self.numE,DEFAULT_NEURONUM_1:(DEFAULT_NEURONUM_1+self.numE)]
                
                idxPreSyn_12 = np.arange(self.numE).reshape(-1,1)
                idxPostSyn_12 = np.arange(DEFAULT_NEURONUM_1, (DEFAULT_NEURONUM_1+self.numE)).reshape(-1,1)
                
                
                STDPAdjMat_12 = self.adjMat[DEFAULT_NEURONUM_1:(DEFAULT_NEURONUM_1+self.numE),:self.numE].copy()
                
                STDPAdjMatMax_12 = STDPAdjMat_12* (1 + self.STDPlevel)
                STDPAdjMatMin_12 = STDPAdjMat_12 * 0 #(1 - self.STDPlevel)                
                deltaWeightConst_12 = STDPAdjMat_12 * self.STDPstep * self.STDPlevel/20.0       
                STDPAdjMat_12 = self.adjMat[DEFAULT_NEURONUM_1:(DEFAULT_NEURONUM_1+self.numE),:self.numE]
                
                tauSTDP = 10 # ms
                # assymetrical STDP learning rule
                tauPlus = 34 # ms
                tauMinus = 14 #ms
                
                windowSTDP = 100 # ms
                
                 
            spikeTimes[neuroIdx[isFiring],spikeCounts[isFiring]] = t
            spikeCounts += isFiring
        # main simulation over

        # compress spikeTimes to a 2D array        
        timingVec = np.concatenate(
            [spikeTimes[i,:spikeCounts[i,0]] for i in neuroIdx.reshape(-1)])
        idVec = np.concatenate(
            [i*np.ones(spikeCounts[i,0]) for i in neuroIdx.reshape(-1)])
        self.spikeTimes = np.stack((timingVec,idVec))
        self.spikeCounts = spikeCounts
        if not isNet: self.states = np.hstack((channelZ,channelH,channelN,memV))
        return self
        # return spikeCounts

    def detectRhythmModLevel(self,tMin=DEFAULT_TEND-4000,tMax=DEFAULT_TEND):
        thresholdTheta = 2
        thresholdGamma = 2
        infThetaBand = 2.5
        supThetaBand = 20
        infGammaBand = 30
        supGammaBand = 200
        freqUpperLimit = 100
        timeWindow = 0.5 * 1000/freqUpperLimit # ms
        
        for modId in [1,2]:
            if modId == 1:
                tempSpikeTimes = self.spikeTimes[:,
                            (self.spikeTimes[0,:]>tMin) & (self.spikeTimes[0,:]<tMax) & (self.spikeTimes[1,:]<DEFAULT_NEURONUM_1)]
            else:
                tempSpikeTimes = self.spikeTimes[:,
                            (self.spikeTimes[0,:]>tMin) & (self.spikeTimes[0,:]<tMax) & (self.spikeTimes[1,:]>=DEFAULT_NEURONUM_1)]
            
            timePoints = np.arange(tMin,tMax,timeWindow)
            iterationTimes = timePoints.shape[0]
            logicalRaster = np.zeros(shape=(iterationTimes,self.neuroNum))
            
            for ithTimeWindow in range(iterationTimes):
                temp1 = np.stack((np.abs(tempSpikeTimes[0,:]-
                                         timePoints[ithTimeWindow]),
                                  tempSpikeTimes[1,:]),axis = 0)
                temp2 = temp1[1,temp1[0,:] <= timeWindow/2]
                logicalRaster[[ithTimeWindow],temp2.astype(int)] = 1
            logicalRaster = logicalRaster.T
            fPoints,PxxDensity = signal.periodogram(logicalRaster,
                                            fs=1000/timeWindow)
            # Network Pxx Density and its normalization
            netPxxDensity = PxxDensity.mean(axis=0)
            netPxxDensity = netPxxDensity/netPxxDensity.mean() 
            
            if modId == 1:                
                self.fPoints_1 = fPoints
                self.netPxxDensity_1 = netPxxDensity
                # log peak power and corresponding freq in theta and gamma band
                # peak power
                self.Ptheta_1 = netPxxDensity[
                    (fPoints>infThetaBand)&(fPoints<supThetaBand)].max()
                self.Pgamma_1 = netPxxDensity[
                    (fPoints>infGammaBand)&(fPoints<supGammaBand)].max()
                # freq part
                # if self.Ptheta_1 > thresholdTheta:
                self.thetaFreq_1 = fPoints[
                    (fPoints>infThetaBand)&(fPoints<supThetaBand)][
                        netPxxDensity[(
                            fPoints>infThetaBand)&(fPoints<supThetaBand)].argmax()] 
                # else:
                #     self.thetaFreq_1 = np.nan
                
                # if self.Pgamma_1 > thresholdGamma:
                self.gammaFreq_1 = fPoints[
                    (fPoints>infGammaBand)&(fPoints<supGammaBand)][
                        netPxxDensity[(
                            fPoints>infGammaBand)&(fPoints<supGammaBand)].argmax()] 
                # else:
                #     self.gammaFreq_1 = np.nan
            else:
                self.fPoints_2 = fPoints
                self.netPxxDensity_2 = netPxxDensity
                # log peak power and corresponding freq in theta and gamma band
                # peak power
                self.Ptheta_2 = netPxxDensity[
                    (fPoints>infThetaBand)&(fPoints<supThetaBand)].max()
                self.Pgamma_2 = netPxxDensity[
                    (fPoints>infGammaBand)&(fPoints<supGammaBand)].max()
                # freq part
                # if self.Ptheta_2 > thresholdTheta:
                self.thetaFreq_2 = fPoints[
                    (fPoints>infThetaBand)&(fPoints<supThetaBand)][
                        netPxxDensity[(
                            fPoints>infThetaBand)&(fPoints<supThetaBand)].argmax()] 
                # else:
                #     self.thetaFreq_2 = np.nan
                
                # if self.Pgamma_2 > thresholdGamma:
                self.gammaFreq_2 = fPoints[
                    (fPoints>infGammaBand)&(fPoints<supGammaBand)][
                        netPxxDensity[(
                            fPoints>infGammaBand)&(fPoints<supGammaBand)].argmax()] 
                # else:
                #     self.gammaFreq_2 = np.nan
        
        return self
    def computeLFP(self,clearVolTraces = True):
        ensemble_names = ["11","12","21","22"]
        lfp = dict()
        for ens in ensemble_names:
            ens_int = int(ens)            
            lfp[ens] = self.vPoints[self.sqwave_idvec.flatten() == ens_int,:].sum(axis=0)
        if clearVolTraces:
            del self.vPoints
        self.lfp = lfp
        return self
    
    
    def computeThetaLFP(self,ts,te, sigma=1):
        tempSpikeTimes = self.spikeTimes[:,(self.spikeTimes[0,:]>ts) & 
                                    (self.spikeTimes[0,:]<=te)]
        timePoints = self.tPoints[(self.tPoints>ts) & (self.tPoints<=te)]
        # maskVec = (ts<=self.tPoints) & (self.tPoints<=te)
        fs = 1000/self.tStep
        self.detectRhythmModLevel(tMin = ts, tMax = te)
        
        ensemble_names = ["11","12","21","22"]
        self.thetaLFP = dict()
        thetaAnalytic = dict()
        self.thetaPhase = dict()
        for ens in ensemble_names:
            ens_int = int(ens)
            mod_int = ens_int//10     
            ensSpikeTimes = tempSpikeTimes[0,self.sqwave_idvec.flatten()[tempSpikeTimes[1,:].astype(int)]==ens_int]
            lfp = np.exp(-(0.5/sigma**2)*(ensSpikeTimes.reshape(-1,1) - timePoints)**2).mean(axis=0)
            
            thetaFreq = self.thetaFreq_1 if mod_int == 1 else self.thetaFreq_2
            
            self.thetaLFP[ens] = butter_bandpass_filter(lfp, 
                                                    thetaFreq-0.5, 
                                                    thetaFreq+0.5, fs)
            
            thetaAnalytic[ens] = signal.hilbert(self.thetaLFP[ens])
            self.thetaPhase[ens] = np.unwrap(np.angle(thetaAnalytic[ens]))
        
        return self
    
    def computeThetaLFP_legacy(self,ts,te):
        maskVec = (ts<=self.tPoints) & (self.tPoints<=te)
        fs = 1000/self.tStep
        self.detectRhythmModLevel(tMin = ts, tMax = te)
        
        ensemble_names = ["11","12","21","22"]
        lfp = dict()
        self.thetaLFP = dict()
        thetaAnalytic = dict()
        self.thetaPhase = dict()
        for ens in ensemble_names:
            ens_int = int(ens)
            mod_int = ens_int//10            
            lfp[ens] = self.lfp[ens][maskVec]
            thetaFreq = self.thetaFreq_1 if mod_int == 1 else self.thetaFreq_2
            
            self.thetaLFP[ens] = butter_bandpass_filter(lfp[ens], 
                                                   thetaFreq-0.5, 
                                                   thetaFreq+0.5, fs)
            
            thetaAnalytic[ens] = signal.hilbert(self.thetaLFP[ens])
            self.thetaPhase[ens] = np.unwrap(np.angle(thetaAnalytic[ens]))
        
        return self
    
    def computeGammaLFP(self,ts,te):
        maskVec = (ts<=self.tPoints) & (self.tPoints<=te)
        fs = 1000/self.tStep
        self.detectRhythmModLevel(tMin = ts, tMax = te)
        
        ensemble_names = ["11","12","21","22"]
        lfp = dict()
        self.gammaLFP = dict()
        gammaAnalytic = dict()
        self.gammaPhase = dict()
        for ens in ensemble_names:
            ens_int = int(ens)
            mod_int = ens_int//10            
            lfp[ens] = self.lfp[ens][maskVec]
            gammaFreq = self.gammaFreq_1 if mod_int == 1 else self.gammaFreq_2
            
            self.gammaLFP[ens] = butter_bandpass_filter(lfp[ens], 
                                                   gammaFreq-2, 
                                                   gammaFreq+2, fs)
            
            gammaAnalytic[ens] = signal.hilbert(self.gammaLFP[ens])
            self.gammaPhase[ens] = np.unwrap(np.angle(gammaAnalytic[ens]))
        
        return self    
    
    def computeXcorrPhaseDiff(self, rhythm = "theta"):
        if rhythm == "theta":
            thetaT = (1000/self.thetaFreq_1 + 1000/self.thetaFreq_2)/2  
            corr_s1 = signal.correlate(self.thetaLFP["21"], self.thetaLFP["11"])
            lags_s1 = signal.correlation_lags(len(self.thetaLFP["21"]), len(self.thetaLFP["11"]))
            
            corr_s2 = signal.correlate(self.thetaLFP["22"], self.thetaLFP["12"])
            lags_s2 = signal.correlation_lags(len(self.thetaLFP["22"]), len(self.thetaLFP["12"]))
    
            phaseDiff_s1 = (self.tStep/thetaT) * lags_s1[corr_s1.argmax()] % 1
            self.xcorrPhaseDiff_s1 = phaseDiff_s1 if phaseDiff_s1<0.5 else phaseDiff_s1-1
            
            phaseDiff_s2 = (self.tStep/thetaT) * lags_s2[corr_s2.argmax()] % 1
            self.xcorrPhaseDiff_s2 = phaseDiff_s2 if phaseDiff_s2<0.5 else phaseDiff_s2-1
        else:
            gammaT = (1000/self.gammaFreq_1 + 1000/self.gammaFreq_2)/2  
            corr_s1 = signal.correlate(self.gammaLFP["21"], self.gammaLFP["11"])
            lags_s1 = signal.correlation_lags(len(self.gammaLFP["21"]), len(self.gammaLFP["11"]))
            
            corr_s2 = signal.correlate(self.gammaLFP["22"], self.gammaLFP["12"])
            lags_s2 = signal.correlation_lags(len(self.gammaLFP["22"]), len(self.gammaLFP["12"]))
    
            phaseDiff_s1 = (self.tStep/gammaT) * lags_s1[corr_s1.argmax()] % 1
            self.xcorrPhaseDiff_s1 = phaseDiff_s1 if phaseDiff_s1<0.5 else phaseDiff_s1-1
            
            phaseDiff_s2 = (self.tStep/gammaT) * lags_s2[corr_s2.argmax()] % 1
            self.xcorrPhaseDiff_s2 = phaseDiff_s2 if phaseDiff_s2<0.5 else phaseDiff_s2-1
        return self
        
    
    
    def detectRhythm(self,tMin=DEFAULT_TEND-4000,tMax=DEFAULT_TEND):
        thresholdTheta = 2
        thresholdGamma = 2
        infThetaBand = 2.5
        supThetaBand = 20
        infGammaBand = 30
        supGammaBand = 200
        freqUpperLimit = 100
        timeWindow = 0.5 * 1000/freqUpperLimit # ms
        
        tempSpikeTimes = self.spikeTimes[:,
                    (self.spikeTimes[0,:]>tMin) & (self.spikeTimes[0,:]<tMax)]
        timePoints = np.arange(tMin,tMax,timeWindow)
        iterationTimes = timePoints.shape[0]
        logicalRaster = np.zeros(shape=(iterationTimes,self.neuroNum))
        
        for ithTimeWindow in range(iterationTimes):
            temp1 = np.stack((np.abs(tempSpikeTimes[0,:]-
                                     timePoints[ithTimeWindow]),
                              tempSpikeTimes[1,:]),axis = 0)
            temp2 = temp1[1,temp1[0,:] <= timeWindow/2]
            logicalRaster[[ithTimeWindow],temp2.astype(int)] = 1
        logicalRaster = logicalRaster.T
        fPoints,PxxDensity = signal.periodogram(logicalRaster,
                                        fs=1000/timeWindow)
        # Network Pxx Density and its normalization
        netPxxDensity = PxxDensity.mean(axis=0)
        netPxxDensity = netPxxDensity/netPxxDensity.mean() 
        self.fPoints = fPoints
        self.netPxxDensity = netPxxDensity
        # log peak power and corresponding freq in theta and gamma band
        # peak power
        self.Ptheta = netPxxDensity[
            (fPoints>infThetaBand)&(fPoints<supThetaBand)].max()
        self.Pgamma = netPxxDensity[
            (fPoints>infGammaBand)&(fPoints<supGammaBand)].max()
        # freq part
        if self.Ptheta > thresholdTheta:
            self.thetaFreq = fPoints[
                (fPoints>infThetaBand)&(fPoints<supThetaBand)][
                    netPxxDensity[(
                        fPoints>infThetaBand)&(fPoints<supThetaBand)].argmax()] 
        else:
            self.thetaFreq = np.nan
        
        if self.Pgamma > thresholdGamma:
            self.gammaFreq = fPoints[
                (fPoints>infGammaBand)&(fPoints<supGammaBand)][
                    netPxxDensity[(
                        fPoints>infGammaBand)&(fPoints<supGammaBand)].argmax()] 
        else:
            self.gammaFreq = np.nan
        
        # Use neuronal Pxx Density to map neuronal type (rhythmic)
        self.neuroRhythmType = np.zeros(self.neuroNum) # 0 for null type
        neuroPtheta = PxxDensity[:,(fPoints>infThetaBand)&(fPoints<supThetaBand)].max(axis=1)
        self.neuroRhythmType[neuroPtheta>thresholdTheta*PxxDensity.mean(axis=1)] = 1 # 1 for theta type
        neuroPgamma = PxxDensity[:,(fPoints>infGammaBand)&(fPoints<supGammaBand)].max(axis=1)
        self.neuroRhythmType[neuroPgamma>thresholdGamma*PxxDensity.mean(axis=1)] = 2 # 2 for gamma type
        self.neuroRhythmType[
            (neuroPtheta>thresholdTheta*PxxDensity.mean(axis=1)) & 
            (neuroPgamma>thresholdGamma*PxxDensity.mean(axis=1))] = 3 # 3 for mixed     
        return self
            
    def computeDotProduct(self,k_max=4):
        tMin = 0
        tMax = self.tEnd
        timeWindow = 5.0 # ms
        tempSpikeTimes = self.spikeTimes[:,
                    ((self.spikeTimes[0,:]>tMin) & 
                     (self.spikeTimes[0,:]<tMax) &
                     (self.spikeTimes[1,:]>=500) &
                     (self.spikeTimes[1,:]< 900))]
        timePoints = np.arange(tMin,tMax,timeWindow)
        iterationTimes = timePoints.shape[0]
        logicalRaster = np.zeros(shape=(iterationTimes,self.neuroNum))
        
        for ithTimeWindow in range(iterationTimes):
            temp1 = np.stack((np.abs(tempSpikeTimes[0,:]-
                                     timePoints[ithTimeWindow]),
                              tempSpikeTimes[1,:]),axis = 0)
            temp2 = temp1[1,temp1[0,:] <= timeWindow/2]
            logicalRaster[[ithTimeWindow],temp2.astype(int)] = 1
        logicalRaster = logicalRaster.T[500:900]
        self.logicalRaster = logicalRaster
        burst_locs = dict()
        burst = 0
        last = False
        isNonEmptyBin = logicalRaster.any(axis = 0)
        for ithTimeWindow in range(iterationTimes):
            if last == False: 
                if isNonEmptyBin[ithTimeWindow]:
                    # new burst
                    burst_locs[burst] = np.array([ithTimeWindow])
                    last = True
            else:# last is true
                if isNonEmptyBin[ithTimeWindow]:
                    burst_locs[burst] = np.array(list(burst_locs[burst])+[ithTimeWindow])
                else:
                    burst += 1
                    last = False
        self.burst_locs = burst_locs
        res = []
        for burst, locs in burst_locs.items():
            # k-th col is for <N,N-k> dot product
            # j-th col is j-th burst result
            if burst<k_max:
                res.append([0]*(k_max+1))
            else:
                temp = []
                for k in range(k_max+1):
                    temp.append(int(logicalRaster[:,locs].any(axis=1,keepdims=True).T.astype(int) @ 
                                logicalRaster[:,burst_locs[burst-k]].any(axis=1,keepdims=True)))
                res.append(temp)
        return np.array(res).T
        
    
    def showRaster(self):
        # preliminary method needs improvement
        # plt.figure()
        plt.plot(self.spikeTimes[0,:],self.spikeTimes[1,:],'o',markersize = 2)
        # plt.xlim(self.tEnd - 500,self.tEnd)
        plt.ylim(0,self.neuroNum)
        # plt.show()
        
        
    def rewireEE(self,rewiringProb=0.2): # assuming const weight
        adjMatEE = self.adjMat[:self.numE,:self.numE]
        tempVec = adjMatEE[adjMatEE!=0]
        synapNum = tempVec.shape[0]
        weightEE = tempVec[0]
        rewiringNum =  round(synapNum*rewiringProb)
        breakId = np.random.choice(synapNum,rewiringNum,replace=False)
        tempVec[breakId] = 0
        adjMatEE[adjMatEE!=0] = tempVec
        # except the diagonal elements
        tempVec = adjMatEE[(adjMatEE+np.eye(self.numE))==0]
        tempVec[np.random.choice(tempVec.shape[0],rewiringNum,replace=False)] = weightEE
        adjMatEE[(adjMatEE+np.eye(self.numE))==0] = tempVec
        return self
    
    def sparsenEE(self,sparsity=0.5): # assuming const weight
        adjMatEE = self.adjMat[:self.numE,:self.numE]
        tempVec = adjMatEE[adjMatEE!=0]
        synapNum = tempVec.shape[0]

        breakId = np.random.choice(synapNum,
                                   round(synapNum*sparsity),
                                   replace=False)
        tempVec[breakId] = 0
        adjMatEE[adjMatEE!=0] = tempVec
        
        return self        
