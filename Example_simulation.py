# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:59:48 2021

@author: yihaoy
"""
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import signal
import numpy as np
import jsonpickle
import json
from Neuro2Nets import Neuro2Nets
#%% test random map 4 sites

net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 3)
net.randomInitialStates().mexicanHat()
net.mapGks(r_1 = 3, releaseLocs_1 = np.random.rand(4,2),
           r_2 = 3, releaseLocs_2 = np.random.rand(4,2)).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True, logV = True)
#%% test random map 8 sites

net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 6)
net.randomInitialStates().mexicanHat()
net.mapGks(r_1 = 2, releaseLocs_1 = np.random.rand(8,2),
           r_2 = 2, releaseLocs_2 = np.random.rand(8,2)).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True, logV = True)
#%% dcbase=0 delay=0 ts_sqwave = 5000 te_sqwave = 10000 no stdp tests - two spots - with sqwave - delay = 50 
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()
net.setupSquareWaveDrive(ts_m1 = 5000, te_m1 = 9950, a_1 = 0,
                         ts_m2 = 5000, te_m2 = 9950, a_2 = 0)
dcbase=0
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation(isSTDP = True, sqwave = True, logV = True)

#%% dcbase=0.5 delay=0 ts_sqwave = 5000 te_sqwave = 10000 no stdp tests - two spots - with sqwave - delay = 50 
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()
net.setupSquareWaveDrive(ts_m1 = 5000, te_m1 = 9950, a_1 = 0,
                         ts_m2 = 5000, te_m2 = 9950, a_2 = 0)
dcbase=0.5
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation(isSTDP = True, sqwave = True, logV = True)

#%% dcbase=1 delay=0 ts_sqwave = 5000 te_sqwave = 10000 no stdp tests - two spots - with sqwave - delay = 50 
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()
net.setupSquareWaveDrive(ts_m1 = 5000, te_m1 = 9950, a_1 = 0,
                         ts_m2 = 5000, te_m2 = 9950, a_2 = 0)
dcbase=1
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation(isSTDP = True, sqwave = True, logV = True)

#%% dcbase2=0 delay=0 ts_sqwave = 5000 te_sqwave = 10000 no stdp tests - two spots - with sqwave - delay = 50 
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()
net.setupSquareWaveDrive(ts_m1 = 5000, te_m1 = 9950, a_1 = 0,
                         ts_m2 = 5000, te_m2 = 9950, a_2 = 0)
dcbase=0
net.mapIdrive(dcBase_2=dcbase)
net.runSimulation(isSTDP = True, sqwave = True, logV = True)

#%% dcbase2=0.5 delay=0 ts_sqwave = 5000 te_sqwave = 10000 no stdp tests - two spots - with sqwave - delay = 50 
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()
net.setupSquareWaveDrive(ts_m1 = 5000, te_m1 = 9950, a_1 = 0,
                         ts_m2 = 5000, te_m2 = 9950, a_2 = 0)
dcbase=0.5
net.mapIdrive(dcBase_2=dcbase)
net.runSimulation(isSTDP = True, sqwave = True, logV = True)

#%% dcbase2=1 delay=0 ts_sqwave = 5000 te_sqwave = 10000 no stdp tests - two spots - with sqwave - delay = 50 
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()
net.setupSquareWaveDrive(ts_m1 = 5000, te_m1 = 9950, a_1 = 0,
                         ts_m2 = 5000, te_m2 = 9950, a_2 = 0)
dcbase=1
net.mapIdrive(dcBase_2=dcbase)
net.runSimulation(isSTDP = True, sqwave = True, logV = True)

#%% dcbase2=1.5 delay=0 ts_sqwave = 5000 te_sqwave = 10000 no stdp tests - two spots - with sqwave - delay = 50 
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()
net.setupSquareWaveDrive(ts_m1 = 5000, te_m1 = 9950, a_1 = 0,
                         ts_m2 = 5000, te_m2 = 9950, a_2 = 0)
dcbase=1.5
net.mapIdrive(dcBase_2=dcbase)
net.runSimulation(isSTDP = True, sqwave = True, logV = True)

#%% dcbase2=2 delay=0 ts_sqwave = 5000 te_sqwave = 10000 no stdp tests - two spots - with sqwave - delay = 50 
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()
net.setupSquareWaveDrive(ts_m1 = 5000, te_m1 = 9950, a_1 = 0,
                         ts_m2 = 5000, te_m2 = 9950, a_2 = 0)
dcbase=2
net.mapIdrive(dcBase_2=dcbase)
net.runSimulation(isSTDP = True, sqwave = True, logV = True)

#%% dcbase2=2.5 delay=0 ts_sqwave = 5000 te_sqwave = 10000 no stdp tests - two spots - with sqwave - delay = 50 
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()
net.setupSquareWaveDrive(ts_m1 = 5000, te_m1 = 9950, a_1 = 0,
                         ts_m2 = 5000, te_m2 = 9950, a_2 = 0)
dcbase=2.5
net.mapIdrive(dcBase_2=dcbase)
net.runSimulation(isSTDP = True, sqwave = True, logV = True)

#%% One spot control  lower dc2 = 0
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_2=0)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)
#%% One spot control  lower dc2 = 0.5
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_2=0.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)
#%% One spot control  lower dc2 = 1
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_2=1)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)
#%% One spot control  lower dc2 = 1.5
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_2=1.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)
#%% One spot control  lower dc2 = 2
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_2=2)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)
#%% One spot control  lower dc2 = 2.5
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_2=2.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)

#%% One spot control  lower dc1 = 0
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=0)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)
#%% One spot control  lower dc1 = 0.5
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=0.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)
#%% One spot control  lower dc1 = 1
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=1)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)
#%% One spot control  lower dc1 = 1.5
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=1.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)
#%% One spot control  lower dc1 = 2
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=2)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)
#%% One spot control  lower dc1 = 2.5
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=2.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)



#%% One spot control  dc = 3 asymSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=3,dcBase_2=3)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)
#%% One spot control  dc = 2.5 asymSTDP with noise
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=2.5,dcBase_2=2.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(poisson_noise = True,poisson_amp = 4,isSTDP = True,logV = True)
#%% One spot control  dc = 1.5 asymSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=1.5,dcBase_2=1.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)
#%% One spot control  dc = 2.5 asymSTDP stdplevel3 with noise
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 3
net.STDPstep = 1/3
net.mapIdrive(dcBase_1=2.5,dcBase_2=2.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(poisson_noise = True,poisson_amp = 4,isSTDP = True)
#%% One spot control  dc = 1.5 asymSTDP stdplevel3
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 3
#net.STDPstep = 1/3
net.mapIdrive(dcBase_1=1.5,dcBase_2=1.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)

#%% One spot control  dc = 2.5 noSTDP nn2Nets with noise
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).nn2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
#net.STDPstep = 1/3
net.mapIdrive(dcBase_1=2.5,dcBase_2=2.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(poisson_noise = True,poisson_amp = 4,isSTDP = False,logV = True)

#%% One spot control  dc = 1.5 noSTDP nn2Nets
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).nn2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 3
#net.STDPstep = 1/3
net.mapIdrive(dcBase_1=1.5,dcBase_2=1.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = False,logV = True)
#%% One spot control  dc = 2.5 asymSTDP nn2Nets with noise
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).nn2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
#net.STDPstep = 1/3
net.mapIdrive(dcBase_1=2.5,dcBase_2=2.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(poisson_noise = True,poisson_amp = 4,isSTDP = True)

#%% One spot control  dc = 1.5 asymSTDP nn2Nets
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).nn2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
#net.STDPstep = 1/3
net.mapIdrive(dcBase_1=1.5,dcBase_2=1.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True)

#%% One spot control  dc = 3 asymSTDP gKsMax_2 = 0.6
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 4, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4, gKsMax_2 = 0.6,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=3,dcBase_2=3)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)
#%% One spot control  dc = 3 asymSTDP gKsMax_2 = 0.8
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 4, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4, gKsMax_2 = 0.8,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=3,dcBase_2=3)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)
#%% One spot control  dc = 3 asymSTDP gKsMax_2 = 1.0
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 4, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,gKsMax_2 = 1,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=3,dcBase_2=3)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)

#%% One spot control  dc = 3 asymSTDP gKsMax_2 = 1.2
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 4, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,gKsMax_2 = 1.2,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=3,dcBase_2=3)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)

#%% One spot control  dc = 3 revSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=3,dcBase_2=3)
net.adjMat0 = net.adjMat.copy()
net.runSimulation_revSTDP(isSTDP = True,logV = True)

#%% One spot control  dc = 2.5 revSTDP
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=2.5,dcBase_2=2.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation_revSTDP(isSTDP = True,logV = True)

#%% One spot control  dc = 2 revSTDP
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=2,dcBase_2=2)
net.adjMat0 = net.adjMat.copy()
net.runSimulation_revSTDP(isSTDP = True,logV = True)
#%% One spot control  dc = 1.5 revSTDP
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=1.5,dcBase_2=1.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation_revSTDP(isSTDP = True,logV = True)

#%% One spot control  dc = 1 revSTDP
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=1,dcBase_2=1)
net.adjMat0 = net.adjMat.copy()
net.runSimulation_revSTDP(isSTDP = True,logV = True)

#%% One spot control  dc = 0.5 revSTDP
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=0.5,dcBase_2=0.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation_revSTDP(isSTDP = True,logV = True)
#%% One spot control  dc = 3 symSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=3,dcBase_2=3)
net.adjMat0 = net.adjMat.copy()
net.runSimulation_symSTDP(isSTDP = True,logV = True)

#%% One spot control  dc = 2.5 symSTDP
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=2.5,dcBase_2=2.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation_symSTDP(isSTDP = True,logV = True)

#%% One spot control  dc = 2 symSTDP
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=2,dcBase_2=2)
net.adjMat0 = net.adjMat.copy()
net.runSimulation_symSTDP(isSTDP = True,logV = True)

#%% One spot control  dc = 1.5 symSTDP
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=1.5,dcBase_2=1.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation_symSTDP(isSTDP = True,logV = True)

#%% One spot control  dc = 1 symSTDP
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=1,dcBase_2=1)
net.adjMat0 = net.adjMat.copy()
net.runSimulation_symSTDP(isSTDP = True,logV = True)

#%% One spot control  dc = 0.5 symSTDP
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=0.5,dcBase_2=0.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation_symSTDP(isSTDP = True,logV = True)

#%% One spot control  dc = 1 asymSTDP
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=1,dcBase_2=1)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)

#%% One spot control  dc = 0.5 asymSTDP
net = Neuro2Nets(tEnd = 5000)
net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, 
               releaseLocs_1 = np.array([[0.5,0.5]]),
               r_2 = 4,
               releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.mapIdrive(dcBase_1=0.5,dcBase_2=0.5)
net.adjMat0 = net.adjMat.copy()
net.runSimulation(isSTDP = True,logV = True)

#%% Two spot control  dc = 3 symSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=3
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation_symSTDP(isSTDP = True, logV = True)
#%% Two spot control  dc = 2.5 symSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=2.5
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation_symSTDP(isSTDP = True, logV = True)

#%% Two spot control  dc = 2 symSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=2
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation_symSTDP(isSTDP = True, logV = True)

#%% Two spot control  dc = 1.5 symSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=1.5
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation_symSTDP(isSTDP = True, logV = True)

#%% Two spot control  dc = 1 symSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=1
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation_symSTDP(isSTDP = True, logV = True)

#%% Two spot control  dc = 0.5 symSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=0.5
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation_symSTDP(isSTDP = True, logV = True)

#%% Two spot control  dc = 3 revSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=3
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation_revSTDP(isSTDP = True, logV = True)
#%% Two spot control  dc = 2.5 revSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=2.5
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation_revSTDP(isSTDP = True, logV = True)

#%% Two spot control  dc = 2 revSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=2
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation_revSTDP(isSTDP = True, logV = True)

#%% Two spot control  dc = 1.5 revSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=1.5
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation_revSTDP(isSTDP = True, logV = True)

#%% Two spot control  dc = 1 revSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=1
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation_revSTDP(isSTDP = True, logV = True)

#%% Two spot control  dc = 0.5 revSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=0.5
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation_revSTDP(isSTDP = True, logV = True)

#%% Two spot control  dc = 3 asymSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=3
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation(isSTDP = True, logV = True)
#%% Two spot control  dc = 2.5 asymSTDP with noise
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=2.5
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation(poisson_noise = True,poisson_amp = 4,isSTDP = True, logV = False)

#%% Two spot control  dc = 2.5 noSTDP nn2Nets
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).nn2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=2.5
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation(isSTDP = False, logV = True)
#%% Two spot control  dc = 1.5 noSTDP nn2Nets
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).nn2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=1.5
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation(isSTDP = False, logV = True)

#%% Two spot control  dc = 2.5 asymSTDP nn2Nets
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).nn2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=2.5
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation(isSTDP = True)
#%% Two spot control  dc = 1.5 asymSTDP nn2Nets
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).nn2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=1.5
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation(isSTDP = True)


#%% Two spot control  dc = 2 asymSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=2
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation(isSTDP = True, logV = True)

#%% Two spot control  dc = 1.5 asymSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=1.5
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation(isSTDP = True, logV = True)

#%% Two spot control  dc = 1 asymSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=1
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation(isSTDP = True, logV = True)

#%% Two spot control  dc = 0.5 asymSTDP
net = Neuro2Nets(tEnd = 5000)
np.random.seed(seed = 1)
net.randomInitialStates().mexicanHat().mapGks(releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
net.tSTDP_on = 500
net.tSTDP_off = 5000
net.STDPlevel = 1
net.STDPstep = 1
net.adjMat0 = net.adjMat.copy()

dcbase=0.5
net.mapIdrive(dcBase_1=dcbase,dcBase_2=dcbase)
net.runSimulation(isSTDP = True, logV = True)