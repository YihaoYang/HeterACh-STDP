#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:33:54 2022

@author: yihaoy
"""
#%% loading libraries
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import signal
import numpy as np
import jsonpickle
import json
from Neuro2Nets import Neuro2Nets
from datetime import datetime
import os.path

#%% simulations: parameter scan (DC,ACh) for two spot 
simulation_start_time = datetime.now()
#file_name_start = ".\\Spatial_data\\" # windows ver.
file_name_start = "./Spatial_data_2spot_NNcon_paramScanFlatmapOnly/" # macos ver.
file_name_end   = ".json"

dc_inputs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
gks_background = [0, 0.1, 0.2, 0.3, 0.6, 0.9, 1.2, 1.5]

for rngseed in [1,2,3,4]:
    for dc in dc_inputs:
        for gksMax in gks_background:        
            exp_name = "dc_{}_gksmax_{}_rngseed_{}".format(round(dc,1),round(gksMax,1),rngseed)
            file_name = file_name_start + exp_name + file_name_end
            if os.path.isfile(file_name): continue            
            
            net = Neuro2Nets(tEnd = 5000)
            np.random.seed(seed = rngseed)
            net.randomInitialStates().mexicanHat().mapGks(r_1 = 4, gKsMax_1 = 1.5,
                           r_2 = 4, gKsMax_2 = gksMax,
                           releaseLocs_2 = np.array([])).nn2Nets(indegree = 40, w = 0.005)
            net.tSTDP_on = 500
            net.tSTDP_off = 5000
            net.STDPlevel = 1
            net.mapIdrive(dcBase_1=3,dcBase_2=dc)
            net.adjMat0 = net.adjMat.copy()
            net.runSimulation(isSTDP = True)
            
            print(file_name)
            print(datetime.now()-simulation_start_time)
    
            with open(file_name,'w') as f:
                json.dump(jsonpickle.encode(net),f)
                
                
#%% simulations: parameter scan (Radius,ACh) for one spot 
simulation_start_time = datetime.now()
#file_name_start = ".\\Spatial_data_gksmin_r2_NNcon\\" # windows ver.
file_name_start = "./Spatial_data_gksmin_r2_mod2only/" # macos ver.
file_name_end   = ".json"

r2s = [0,1,2,3,4,5,6]
gksMins = [0, 0.2, 0.4, 0.6, 0.8]
for rngseed in [1,2,3]:
    for r2 in r2s:
        for gksMin in gksMins:            
            exp_name = "r2_{}_gksmin_{}_rngseed_{}".format(r2,round(gksMin,1),rngseed)
            file_name = file_name_start + exp_name + file_name_end
            if os.path.isfile(file_name): continue            
            
            net = Neuro2Nets(tEnd = 5000)
            np.random.seed(seed = rngseed)
            net.randomInitialStates().mexicanHat().mapGks(r_1 = 4, gKsMin_1 = 0.2,
                           releaseLocs_1 = np.array([[0.5,0.5]]),
                           r_2 = r2, gKsMin_2 = gksMin,
                           releaseLocs_2 = np.array([[0.5,0.5]])).randG2Nets(indegree = 40, w = 0.005)
            net.tSTDP_on = 500
            net.tSTDP_off = 5000
            net.STDPlevel = 1
            net.adjMat0 = net.adjMat.copy()
            net.runSimulation(isSTDP = True)
            
            print(file_name)
            print(datetime.now()-simulation_start_time)
    
            with open(file_name,'w') as f:
                json.dump(jsonpickle.encode(net),f)
#%% simulations: parameter scan (DC,ACh) for both flat  
simulation_start_time = datetime.now()
#file_name_start = ".\\Spatial_data\\" # windows ver.
file_name_start = "./NoisePer100_Spatial_flatBoth_parasMod2Only_gksMin_0d6_dc_2/" # macos ver.
file_name_end   = ".json"

dc_inputs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
dc_inputs = [2.0]
gks_background = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]
for rngseed in [1,2,3,4]:
    for dc in dc_inputs:
        for gksMax in gks_background:        
            exp_name = "dc_{}_gksmax_{}_rngseed_{}".format(round(dc,1),round(gksMax,1),rngseed)
            file_name = file_name_start + exp_name + file_name_end
            if os.path.isfile(file_name): continue            
            
            net = Neuro2Nets(tEnd = 5000)
            np.random.seed(seed = rngseed)
            net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, gKsMax_1 = 0.6,
                           releaseLocs_1 = np.array([]),
                           r_2 = 4, gKsMax_2 = gksMax,
                           releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
            net.tSTDP_on = 500
            net.tSTDP_off = 5000
            net.STDPlevel = 1 
            net.mapIdrive(dcBase_1=2,dcBase_2=dc)
            net.adjMat0 = net.adjMat.copy()
            net.runSimulation(isSTDP = True, poisson_noise = True,
                              poisson_rate=1/100, 
                              logV = False)
            print(file_name)
            print(datetime.now()-simulation_start_time)
    
            with open(file_name,'w') as f:
                json.dump(jsonpickle.encode(net),f)
#%% simulations: parameter scan (DC,ACh) for one spot 
simulation_start_time = datetime.now()
#file_name_start = ".\\Spatial_data\\" # windows ver.
file_name_start = "./Spatial_data_1spot_NNcon_paramScanFlatmapOnly_gksMin_0d6_dc_2/" # macos ver.
file_name_end   = ".json"

dc_inputs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
# dc_inputs = [2.0, 2.5, 3.0, 3.5]
gks_background = [0, 0.1, 0.3, 0.6, 0.9, 1.2, 1.5]
for rngseed in [1,2,3,4]:
    for dc in dc_inputs:
        for gksMax in gks_background:        
            exp_name = "dc_{}_gksmax_{}_rngseed_{}".format(round(dc,1),round(gksMax,1),rngseed)
            file_name = file_name_start + exp_name + file_name_end
            if os.path.isfile(file_name): continue            
            
            net = Neuro2Nets(tEnd = 5000)
            np.random.seed(seed = rngseed)
            net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, gKsMin_1 = 0.6,
                           releaseLocs_1 = np.array([[0.5,0.5]]),
                           r_2 = 4, gKsMax_2 = gksMax,
                           releaseLocs_2 = np.array([])).nn2Nets(indegree = 40, w = 0.005)
            net.tSTDP_on = 500
            net.tSTDP_off = 5000
            net.STDPlevel = 1 
            net.mapIdrive(dcBase_1=2,dcBase_2=dc)
            net.adjMat0 = net.adjMat.copy()
            net.runSimulation(isSTDP = True, poisson_noise = False,
                              poisson_rate=1/200, 
                              logV = False)
            
            print(file_name)
            print(datetime.now()-simulation_start_time)
    
            with open(file_name,'w') as f:
                json.dump(jsonpickle.encode(net),f)
#%% simulations: parameter scan
simulation_start_time = datetime.now()
#file_name_start = ".\\Spatial_data\\" # windows ver.
file_name_start = "./Spatial_data_1spot_paramScanFlatmapOnly_gksMin_0d6_dc_2/" # macos ver.
file_name_end   = ".json"

# dc_inputs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
# dc_inputs = [2.0, 2.5, 3.0, 3.5]
dc_inputs = [2.0]
gks_background = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]
for rngseed in [1,2,3,4]:
    for dc in dc_inputs:
        for gksMax in gks_background:        
            exp_name = "dc_{}_gksmax_{}_rngseed_{}".format(round(dc,1),round(gksMax,1),rngseed)
            file_name = file_name_start + exp_name + file_name_end
            if os.path.isfile(file_name): continue            
            
            net = Neuro2Nets(tEnd = 5000)
            np.random.seed(seed = rngseed)
            net.randomInitialStates().mexicanHat().mapGks(r_1 = 6, gKsMin_1 = 0.6,
                           releaseLocs_1 = np.array([[0.5,0.5]]),
                           r_2 = 4, gKsMax_2 = gksMax,
                           releaseLocs_2 = np.array([])).randG2Nets(indegree = 40, w = 0.005)
            net.tSTDP_on = 500
            net.tSTDP_off = 5000
            net.STDPlevel = 1 
            net.mapIdrive(dcBase_1=2,dcBase_2=dc)
            net.adjMat0 = net.adjMat.copy()
            net.runSimulation(isSTDP = True, poisson_noise = False,
                              poisson_rate=1/200, 
                              logV = False)
            
            print(file_name)
            print(datetime.now()-simulation_start_time)
    
            with open(file_name,'w') as f:
                json.dump(jsonpickle.encode(net),f)
#%% simulations: parameter scan (DC,ACh) for two spot 
simulation_start_time = datetime.now()
#file_name_start = ".\\Spatial_data\\" # windows ver.
file_name_start = "./Spatial_data_2spot_NNcon_paramScanFlatmapOnly/" # macos ver.
file_name_end   = ".json"

dc_inputs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
gks_background = [0, 0.1, 0.2, 0.3, 0.6, 0.9, 1.2, 1.5]

for rngseed in [1,2,3,4]:
    for dc in dc_inputs:
        for gksMax in gks_background:        
            exp_name = "dc_{}_gksmax_{}_rngseed_{}".format(round(dc,1),round(gksMax,1),rngseed)
            file_name = file_name_start + exp_name + file_name_end
            # if os.path.isfile(file_name): continue            
            
            net = Neuro2Nets(tEnd = 5000)
            np.random.seed(seed = rngseed)
            net.randomInitialStates().mexicanHat().mapGks(r_1 = 4, gKsMax_1 = 1.5,
                           r_2 = 4, gKsMax_2 = gksMax,
                           releaseLocs_2 = np.array([])).nn2Nets(indegree = 40, w = 0.005)
            net.tSTDP_on = 500
            net.tSTDP_off = 5000
            net.STDPlevel = 1
            net.mapIdrive(dcBase_1=3,dcBase_2=dc)
            net.adjMat0 = net.adjMat.copy()
            net.runSimulation(isSTDP = True)
            
            print(file_name)
            print(datetime.now()-simulation_start_time)
    
            # with open(file_name,'w') as f:
            #     json.dump(jsonpickle.encode(net),f)
                
                
#%% simulations: parameter scan (Radius,ACh) for one spot 
simulation_start_time = datetime.now()
#file_name_start = ".\\Spatial_data_gksmin_r2_NNcon\\" # windows ver.
file_name_start = "./local_var_gksmin/" # macos ver.
file_name_end   = ".json"

r2s = [0,1,2,3,4,5,6]
gksMins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
for rngseed in [3,4]:

    for gksMin in gksMins:            
        exp_name = "gksmin_{}_rngseed_{}".format(round(gksMin,1),rngseed)
        file_name = file_name_start + exp_name + file_name_end
        if os.path.isfile(file_name): continue            
        
        net = Neuro2Nets(tEnd = 5000)
        np.random.seed(seed = rngseed)
        net.randomInitialStates().mexicanHat().mapGks(r_1 = 5, gKsMin_1 = gksMin,
                       releaseLocs_1 = np.array([[0.5,0.5]]),
                       r_2 = 5, gKsMin_2 = gksMin,
                       releaseLocs_2 = np.array([[0.5,0.5]])).nn2Nets(indegree = 5, w = 0.005)
        net.tSTDP_on = 500
        net.tSTDP_off = 5000
        net.STDPlevel = 1
        net.adjMat0 = net.adjMat.copy()
        net.runSimulation(isSTDP = True,poisson_noise = True)
        
        print(file_name)
        print(datetime.now()-simulation_start_time)

        with open(file_name,'w') as f:
            json.dump(jsonpickle.encode(net),f)
#%% simulations: parameter scan (DC,ACh) for one spot 
simulation_start_time = datetime.now()
#file_name_start = ".\\Spatial_data\\" # windows ver.
file_name_start = "./Spatial_V_diffspot_paramScanFlatmapOnly_mod1_gksMin_0d6_dc_2_deg_20/" # macos ver.
file_name_end   = ".json"

dc_inputs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
gks_background = [0, 0.3, 0.6, 0.9, 1.2, 1.5]
for rngseed in [1]:
    for dc in dc_inputs:
        for gksMin2 in gks_background:        
            exp_name = "dc_{}_gksmin2_{}_rngseed_{}".format(round(dc,1),round(gksMin2,1),rngseed)
            file_name = file_name_start + exp_name + file_name_end
            if os.path.isfile(file_name): continue            
            
            net = Neuro2Nets(tEnd = 5000)
            np.random.seed(seed = rngseed)
            net.randomInitialStates().mexicanHat().mapGks(r_1 = 4, gKsMax_1 = 1.5, gKsMin_1=0.6,
                           releaseLocs_1 = np.array([[0.25,0.25]]),
                           r_2 = 4, gKsMax_2 = 1.5,gKsMin_2 = gksMin2,
                           releaseLocs_2 = np.array([[0.75,0.75]])).nn2Nets(indegree = 20, w = 0.005)
            net.tSTDP_on = 500
            net.tSTDP_off = 5000
            net.STDPlevel = 1
            net.mapIdrive(dcBase_1=2,dcBase_2=dc)
            net.adjMat0 = net.adjMat.copy()
            net.runSimulation(isSTDP = True)
            
            print(file_name)
            print(datetime.now()-simulation_start_time)
    
            with open(file_name,'w') as f:
                json.dump(jsonpickle.encode(net),f)
                
#%% simulations: parameter scan (DC,ACh) for one spot (both mods)
simulation_start_time = datetime.now()
#file_name_start = ".\\Spatial_data\\" # windows ver.
file_name_start = "./NoiseOn_Spatial_V_gksMin_0d2_dc_3/" # macos ver.
file_name_end   = ".json"

dc_inputs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
gks_background = [0, 0.3, 0.6, 0.9, 1.2]
for rngseed in [1]:
    for dc in dc_inputs:
        for gksMin in gks_background:        
            exp_name = "dc_{}_gksmin2_{}_rngseed_{}".format(round(dc,1),round(gksMin,1),rngseed)
            file_name = file_name_start + exp_name + file_name_end
            if os.path.isfile(file_name): continue            
            
            net = Neuro2Nets(tEnd = 5000)
            np.random.seed(seed = rngseed)
            net.randomInitialStates().mexicanHat().mapGks(r_1 = 4, gKsMax_1 = 1.5, gKsMin_1=gksMin,
                           releaseLocs_1 = np.array([[0.25,0.25]]),
                           r_2 = 4, gKsMax_2 = 1.5,gKsMin_2 = gksMin,
                           releaseLocs_2 = np.array([[0.75,0.75]])).nn2Nets(indegree = 5, w = 0.005)
            net.tSTDP_on = 500
            net.tSTDP_off = 5000
            net.STDPlevel = 1
            net.mapIdrive(dcBase_1=dc,dcBase_2=dc)
            net.adjMat0 = net.adjMat.copy()
            net.runSimulation(isSTDP = True,poisson_noise = True)
            
            print(file_name)
            print(datetime.now()-simulation_start_time)
    
            with open(file_name,'w') as f:
                json.dump(jsonpickle.encode(net),f)
                
