#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:33:54 2022

@author: yihaoy
"""
#%% loading libraries
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import gridspec
from scipy.spatial import distance
from scipy import signal
import numpy as np
import jsonpickle
import json
from Neuro2Nets import Neuro2Nets
import pandas as pd
import seaborn as sns
from collections import defaultdict
import os.path
from datetime import datetime
from matplotlib.ticker import PercentFormatter
#%% spike traces
def normalize2(v,a,b):
    return (b-a)*(v-v.min())/(v.max() - v.min()) + a
s,e = 4000,4300 # ms
for front_str in ["","NoiseOn_"]:
# Spatial_data_1spot_paramScanFlatmapOnly_gksMin_0d6_dc_2
# Spatial_data_1spot_NNcon_paramScanFlatmapOnly_gksMin_0d6_dc_2
# Spatial_flatBoth_parasMod2Only_gksMin_0d2_dc_3/
# Spatial_data_1spot_paramScanFlatmapOnly_gksMin_0d2_dc_3
# Spatial_data_1spot_NNcon_paramScanFlatmapOnly_gksMin_0d2_dc_3
# for front_str in ["","NoisePer400_","NoiseOn_","NoisePer100_"]:
    file_name = "./"+front_str+"Spatial_flatBoth_parasMod2Only_gksMin_0d6_dc_2/" +\
          'dc_3.0_gksmax_0.6_rngseed_1.json'
    with open(file_name,'r') as f:
        net = jsonpickle.decode(json.load(f)) 
    ESpikeTimes = net.spikeTimes[:,(net.spikeTimes[0,:]>s) & 
                                    (net.spikeTimes[0,:]<=e) &
                                    ((net.spikeTimes[1,:]<400) | 
                                     ((net.spikeTimes[1,:]<900) & 
                                     (net.spikeTimes[1,:]>=500)))]    
    ISpikeTimes = net.spikeTimes[:,(net.spikeTimes[0,:]>s) & 
                                    (net.spikeTimes[0,:]<=e) &
                                    ((net.spikeTimes[1,:]>=900) | 
                                     ((net.spikeTimes[1,:]<500) & 
                                     (net.spikeTimes[1,:]>=400)))] 
    
    net.computeModMPC(s=s,e=e)
    E2SpikeTraces = normalize2(net.E1SpikeTraces, 200, 700)
    E1SpikeTraces = normalize2(net.E2SpikeTraces, 500, 1000)
    fig=plt.figure(figsize=[9,4],dpi=200)
    ax0 = plt.subplot()
    
    ax0.plot(ESpikeTimes[0,:],ESpikeTimes[1,:],'k.',markersize = 2)
    ax0.plot(ISpikeTimes[0,:],ISpikeTimes[1,:],'g.',markersize = 2)
    ax0.plot(net.timePoints,E1SpikeTraces,label="Mod 1")
    ax0.plot(net.timePoints,E2SpikeTraces,label="Mod 2")
    
    plt.legend()
    plt.ylim(0,net.neuroNum)
    plt.xlim(s,e)
    plt.xticks([s,e])
    plt.ylabel('neuron ID')
    plt.xlabel('t [ms]')
    

#%% stdp effect measure
hist_color = 'tab:blue'
stdp_color = 'tab:orange'
plt.rcParams.update({'font.size': 20})
plt.rc('font', family='Arial')

bin_width = 10
overall_dW = []
for front_str in ["","NoiseOn_"]:#"NoiseOn_"
    file_name = "./"+front_str+"Spatial_flatBoth_parasMod2Only_gksMin_0d6_dc_2/" +\
          'dc_3.5_gksmax_1.2_rngseed_1.json'
          
    with open(file_name,'r') as f:
        net = jsonpickle.decode(json.load(f)) 
    for s,t in [(4000,5000)]:#(1000,2000),(2000,3000),(3000,4000),
        # s,t = 3000, 5000 # ms
        mod2SpikeTimes = net.spikeTimes[0,(net.spikeTimes[0,:]>s) & 
                                        (net.spikeTimes[0,:]<=t) &
                                        (net.spikeTimes[1,:]<net.numE)]
        mod1SpikeTimes = net.spikeTimes[0,(net.spikeTimes[0,:]>s) & 
                                        (net.spikeTimes[0,:]<=t) &
                                        (net.spikeTimes[1,:]>=((net.numI+net.numE))) &
                                        (net.spikeTimes[1,:]<(net.neuroNum-net.numI))]
        pairedSpikeDiff21 = (mod1SpikeTimes - mod2SpikeTimes.reshape(-1,1)).reshape(-1)
        
        plt.figure(figsize=[6,4],dpi=200)
        # plt.title("paired spike diff [{}-{} ms] (mod1 - mod2)".format(s,t))
        plt.xlabel('$\Delta t_{i,j}$')
        ax = sns.histplot(pairedSpikeDiff21[(pairedSpikeDiff21>-100) & (pairedSpikeDiff21<100)]+0.01,
                     bins=np.arange(-100,101,bin_width),
                     stat = 'proportion',
                     edgecolor = None,color = hist_color)
        ax.set_ylabel('Proportion', color=hist_color)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(axis ='y', labelcolor = hist_color)
        
        ax2 = ax.twinx()
        xvals = np.arange(-100,100,bin_width)+bin_width/2
        yvals = np.zeros(xvals.size)
        i = 0
        num_spikepair = ((pairedSpikeDiff21>-100) & (pairedSpikeDiff21<100)).sum()
        for xval in xvals:
            left, right = xval-bin_width/2, xval+bin_width/2
            tempSpDiff = pairedSpikeDiff21[(pairedSpikeDiff21>=left) & (pairedSpikeDiff21<right)]
            if xval < 0:
                yvals[i] = - (np.exp(tempSpDiff/34)/2).sum()/160000*5*4
            else:
                yvals[i] = (np.exp(-tempSpDiff/14)).sum()/160000*5*4
            i += 1
        
        sns.lineplot(x=[-100,101], y=0, color='black', lw=2, ax=ax2, linestyle='--')
        sns.lineplot(x=xvals, y=yvals, marker='o', color=stdp_color, lw=3, ax=ax2)
        ax2.set_ylabel('$\Delta W^*$', color=stdp_color)
    
        ax2.tick_params(axis ='y', labelcolor = stdp_color)
        
        plt.show()
        overall_dW.append(yvals.sum())
#%% bar plot for delta W star
from matplotlib import ticker
plt.figure(figsize=[6,4],dpi=200)
plt.bar(['$f_{noise}=0Hz$','$f_{noise}=5Hz$'],overall_dW,width=0.4,
        color = stdp_color,alpha=0.8)
plt.axhline(0, linestyle='--', color='black')
plt.ylabel('$\Delta W^*$')
plt.yscale('symlog',linthresh=2,linscale=0.5)
plt.yticks([-100,-10,-2,2,10,100],["-100%","-10%","-2%","+2%","+10%","+100%"])
plt.ylim(-100,100)
# plt.locator_params(axis='y',tight=True, nbins=5)

#%% data flat map learning pattern (rand connection) 
simulation_start_time = datetime.now()

file_name_start = "./NoiseOn_Spatial_flatBoth_parasMod2Only_gksMin_0d2_dc_3/" # macos ver.
file_name_start = "./NoiseOn_Spatial_data_1spot_paramScanFlatmapOnly_gksMin_0d6_dc_2/"
file_name_start = "./Spatial_data_1spot_NNcon_paramScanFlatmapOnly_gksMin_0d6_dc_2/"
file_name_start = "./NoiseOn_Spatial_data_1spot_paramScanFlatmapOnly_gksMin_0d2_dc_3/"
file_name_start = "./NoiseOn_Spatial_data_1spot_NNcon_paramScanFlatmapOnly_gksMin_0d2_dc_3/"

file_name_end   = ".json"


dc_inputs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
gks_background = [0, 0.3, 0.6, 0.9, 1.2, 1.5]

syn_change_12 = defaultdict(list)
syn_change_21 = defaultdict(list)

for dc in dc_inputs:
    for gksMax in gks_background:
        for rngseed in [1]:#[1,2,3,4]:
            exp_name = "dc_{}_gksmax_{}_rngseed_{}".format(round(dc,1),round(gksMax,1),rngseed)
            file_name = file_name_start + exp_name + file_name_end
            with open(file_name,'r') as f:
                net = jsonpickle.decode(json.load(f)) 
            diffMat_21 = net.adjMat[:400,500:900] - net.adjMat0[:400,500:900]
            diffMat_12 = net.adjMat[500:900,:400] - net.adjMat0[500:900,:400]
            syn_change_12[dc,gksMax].append(diffMat_12.sum())
            syn_change_21[dc,gksMax].append(diffMat_21.sum())

            print(file_name)
            print(datetime.now()-simulation_start_time)
#%% plotting flat map learning pattern (rand connection) 
df = 1.25*pd.DataFrame(syn_change_21).T.rename_axis(['dc','gks_level']).rename(lambda x:x+1, axis = 'columns')
plt.figure(figsize=[6,4],dpi=200)
sns.heatmap(df.mean(axis=1).unstack(level=0),vmin=-100, vmax=100, cmap = "seismic",
            center=0,norm=colors.SymLogNorm(linthresh=2,linscale=0.5,vmin=-100, vmax=100, base=10))
plt.title("synaptic change from 2 to 1")

df = 1.25*pd.DataFrame(syn_change_12).T.rename_axis(['dc','gks_level']).rename(lambda x:x+1, axis = 'columns')
plt.figure(figsize=[6,4],dpi=200)
sns.heatmap(df.mean(axis=1).unstack(level=0),vmin=-100, vmax=100,cmap = "seismic",
            center=0,norm=colors.SymLogNorm(linthresh=2,linscale=0.5,vmin=-100, vmax=100, base=10))
plt.title("synaptic change from 1 to 2")

#%% Raster+incoming syn diff visual
s,t = 4700,5000 # ms
for front_str in ["","NoiseOn_"]:
# Spatial_data_1spot_paramScanFlatmapOnly_gksMin_0d6_dc_2
# Spatial_data_1spot_NNcon_paramScanFlatmapOnly_gksMin_0d6_dc_2
# Spatial_flatBoth_parasMod2Only_gksMin_0d2_dc_3/
# Spatial_data_1spot_paramScanFlatmapOnly_gksMin_0d2_dc_3
# Spatial_data_1spot_NNcon_paramScanFlatmapOnly_gksMin_0d2_dc_3
# for front_str in ["","NoisePer400_","NoiseOn_","NoisePer100_"]:
    file_name = "./"+front_str+"Spatial_flatBoth_parasMod2Only_gksMin_0d6_dc_2/" +\
          'dc_3.0_gksmax_0.6_rngseed_1.json'
    with open(file_name,'r') as f:
        net = jsonpickle.decode(json.load(f)) 
    ESpikeTimes = net.spikeTimes[:,(net.spikeTimes[0,:]>s) & 
                                    (net.spikeTimes[0,:]<=t) &
                                    ((net.spikeTimes[1,:]<400) | 
                                     ((net.spikeTimes[1,:]<900) & 
                                     (net.spikeTimes[1,:]>=500)))]    
    ISpikeTimes = net.spikeTimes[:,(net.spikeTimes[0,:]>s) & 
                                    (net.spikeTimes[0,:]<=t) &
                                    ((net.spikeTimes[1,:]>=900) | 
                                     ((net.spikeTimes[1,:]<500) & 
                                     (net.spikeTimes[1,:]>=400)))] 
    
    
    fig=plt.figure(figsize=[6,4],dpi=200)
    gs = gridspec.GridSpec(1,2,width_ratios=[2,1])
    ax0 = plt.subplot(gs[0])
    
    ax0.plot(ESpikeTimes[0,:],ESpikeTimes[1,:],'k.',markersize = 2)
    ax0.plot(ISpikeTimes[0,:],ISpikeTimes[1,:],'g.',markersize = 2)
    plt.ylim(0,net.neuroNum)
    plt.xlim(s,t)
    plt.xticks([s,t])
    plt.ylabel('neuron ID')
    plt.xlabel('t [ms]')
    ax1 = plt.subplot(gs[1])
    diffMat_21 = net.adjMat[:400,500:900] - net.adjMat0[:400,500:900]
    diffMat_12 = net.adjMat[500:900,:400] - net.adjMat0[500:900,:400]
    
    presynWeightDiff = 500*np.vstack((diffMat_21.sum(axis = 1).reshape(20,20),
                                  np.zeros((5,20)),
                                  diffMat_12.sum(axis = 1).reshape(20,20)))
                                  
    
    #plt.title('{}\nIncoming synapses diff ranging [{:.3f},{:.3f}]'.format(exp_name,diffMin,diffMax))
    cax=ax1.imshow(presynWeightDiff,cmap='seismic',
               norm=colors.SymLogNorm(linthresh=2,linscale=0.5,
                                      vmin=-100, vmax=100, base=10),
               origin='lower')
    ax1.axes.xaxis.set_ticklabels([])
    ax1.axes.yaxis.set_ticklabels([])
    ax1.axes.yaxis.set_label_position("right")
    # fig.colorbar(cax,ax=ax1)
    plt.xlabel('X Loc')
    plt.ylabel('Y Loc')

    gKsMat = np.vstack((net.gKs[:400].reshape(20,20), 
                        np.nan*np.ones((5,20)), 
                        net.gKs[500:900].reshape(20,20)))
    plt.figure(figsize=[6,4],dpi=200)
    plt.imshow(gKsMat, origin='lower', cmap='viridis', vmin=0,vmax=1.5)
    # plt.title("gks map")
    plt.colorbar()
    plt.xlabel('X Loc')
    plt.ylabel('Y Loc')
    plt.xticks([])
    plt.yticks([])
    
#%% local Raster+incoming syn diff visual
s,t = 4700,5000 # ms
for front_str in [""]:
    file_name = "./"+front_str+"local_var_gksmin/" +\
          'gksmin_0.4_rngseed_1.json'
    with open(file_name,'r') as f:
        net = jsonpickle.decode(json.load(f)) 
    ESpikeTimes = net.spikeTimes[:,(net.spikeTimes[0,:]>s) & 
                                    (net.spikeTimes[0,:]<=t) &
                                    ((net.spikeTimes[1,:]<400) | 
                                     ((net.spikeTimes[1,:]<900) & 
                                     (net.spikeTimes[1,:]>=500)))]    
    ISpikeTimes = net.spikeTimes[:,(net.spikeTimes[0,:]>s) & 
                                    (net.spikeTimes[0,:]<=t) &
                                    ((net.spikeTimes[1,:]>=900) | 
                                     ((net.spikeTimes[1,:]<500) & 
                                     (net.spikeTimes[1,:]>=400)))] 
    
    
    fig=plt.figure(figsize=[6,4],dpi=200)
    gs = gridspec.GridSpec(1,2,width_ratios=[2,1])
    ax0 = plt.subplot(gs[0])
    
    ax0.plot(ESpikeTimes[0,:],ESpikeTimes[1,:],'k.',markersize = 2)
    ax0.plot(ISpikeTimes[0,:],ISpikeTimes[1,:],'g.',markersize = 2)
    plt.ylim(0,net.neuroNum)
    plt.xlim(s,t)
    ax1 = plt.subplot(gs[1])
    diffMat_21 = net.adjMat[:400,500:900] -net.adjMat0[:400,500:900]
    diffMat_12 = net.adjMat[500:900,:400] - net.adjMat0[500:900,:400]
    
    presynWeightDiff = 500*np.vstack((diffMat_21.sum(axis = 1).reshape(20,20),
                                  np.zeros((5,20)),
                                  diffMat_12.sum(axis = 1).reshape(20,20)))
                                  
    
    #plt.title('{}\nIncoming synapses diff ranging [{:.3f},{:.3f}]'.format(exp_name,diffMin,diffMax))
    cax=ax1.imshow(presynWeightDiff,cmap='seismic',
               norm=colors.SymLogNorm(linthresh=2,linscale=0.5,
                                      vmin=-100, vmax=100, base=10),
               origin='lower')
    ax1.axes.xaxis.set_ticklabels([])
    ax1.axes.yaxis.set_ticklabels([])
    # fig.colorbar(cax,ax=ax1)

    gKsMat = np.vstack((net.gKs[:400].reshape(20,20), 
                        np.nan*np.ones((5,20)), 
                        net.gKs[500:900].reshape(20,20)))
    plt.figure(figsize=[6,4],dpi=200)
    plt.imshow(gKsMat, origin='lower', cmap='viridis', vmin=0,vmax=1.5)
    plt.title("gks map")
    plt.colorbar()
    
#%% in and out synaptic change vs gksmin using setupSquareWaveDriveOneSpot
simulation_start_time = datetime.now()
#file_name_start = ".\\Spatial_data_gksmin_r2_NNcon\\" # windows ver.
file_name_start = "./local_var_gksmin/" # macos ver.
file_name_end   = ".json"

syn_change_in = defaultdict(list)
syn_change_out = defaultdict(list)
syn_change_in_std = defaultdict(list)
syn_change_out_std = defaultdict(list)

gksMins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
for rngseed in [1,2,3,4]:
    for gksMin in gksMins:            
        exp_name = "gksmin_{}_rngseed_{}".format(round(gksMin,1),rngseed)
        file_name = file_name_start + exp_name + file_name_end
        with open(file_name,'r') as f:
            net = jsonpickle.decode(json.load(f)) 
        diffMat_21 = net.adjMat[:400,500:900] - net.adjMat0[:400,500:900]
        diffMat_12 = net.adjMat[500:900,:400] - net.adjMat0[500:900,:400]
        
        # diffMat = net.adjMat - net.adjMat0

        net.setupSquareWaveDriveOneSpot(r_1 = 5, r_2 = 5)

        idvec = net.sqwave_idvec.copy()
        idvec[400:500] = 0
        idvec[900:]    = 0
        idvec = idvec.flatten()

        id_10 = np.arange(1000)[idvec == 10]
        id_11 = np.arange(1000)[idvec == 11]
        id_20 = np.arange(1000)[idvec == 20]-500
        id_21 = np.arange(1000)[idvec == 21]-500
        
        # syn_change_in[gksMin].append(diffMat_21[id_11,:].sum()+diffMat_12[id_21,:].sum())
        # syn_change_out[gksMin].append(diffMat_21[id_10,:].sum()+diffMat_21[id_20,:].sum())
        syn_change_in[gksMin].append(diffMat_21[id_11,:].sum())
        syn_change_out[gksMin].append(diffMat_21[id_10,:].sum())
        syn_change_in[gksMin].append(diffMat_12[id_21,:].sum())
        syn_change_out[gksMin].append(diffMat_12[id_20,:].sum())
        
        
#%% in and out plotting v
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=[6,4],dpi=200)
df_in = 10*(400/id_11.size)*pd.DataFrame(syn_change_in).T.rename_axis(['gks_min']).rename(lambda x:x+1, axis = 'columns')
plt.plot(gksMins,df_in.mean(axis=1),label='inside spot')
df_out = 10*(400/id_10.size)*pd.DataFrame(syn_change_out).T.rename_axis(['gks_min']).rename(lambda x:x+1, axis = 'columns')
plt.plot(gksMins,df_out.mean(axis=1),label='outside spot')
# plt.legend()
# plt.xlabel('hotspot $g_{Ks}$')
plt.ylabel('$\Delta W$')
plt.xticks(gksMins)
plt.xlabel('hotspot $g_{Ks}$')
plt.yticks([0,20,40,60,80],["0","20%","40%","60%","80%"])

plt.figure(figsize=[6,4],dpi=200)
df_in = 10*(400/id_11.size)*pd.DataFrame(syn_change_in).T.rename_axis(['gks_min']).rename(lambda x:x+1, axis = 'columns')
plt.plot(gksMins,df_in.std(axis=1)/8**0.5,label='inside spot')
df_out = 10*(400/id_10.size)*pd.DataFrame(syn_change_out).T.rename_axis(['gks_min']).rename(lambda x:x+1, axis = 'columns')
plt.plot(gksMins,df_out.std(axis=1)/8**0.5,label='outside spot')

plt.xlabel('hotspot $g_{Ks}$')
plt.ylabel('$SE_{\overline{\Delta W}}$')
plt.xticks(gksMins)
plt.yticks([0,10,20],["0","10%","20%"])
# plt.legend(loc=[0.5,1.2])

#%% in and out plotting
plt.figure(figsize=[6,4],dpi=200)
df_in = 10*(200/id_11.size)*pd.DataFrame(syn_change_in).T.rename_axis(['gks_min']).rename(lambda x:x+1, axis = 'columns')
plt.errorbar(gksMins,df_in.mean(axis=1),yerr=df_in.std(axis=1)/2,label='in')
df_out = 10*(200/id_10.size)*pd.DataFrame(syn_change_out).T.rename_axis(['gks_min']).rename(lambda x:x+1, axis = 'columns')
plt.errorbar(gksMins,df_out.mean(axis=1),yerr=df_out.std(axis=1)/2,label='out')
plt.legend()
plt.xlabel('gks_min')
plt.ylabel('change %')

#%% gks effect on bottom module learning pattern (choose columns)
# get the data 
simulation_start_time = datetime.now()
#file_name_start = ".\\Spatial_data_gksmin_r2_NNcon\\" # windows ver.
file_name_start = "Spatial_flatBoth_parasMod2Only_gksMin_0d6_dc_2/" # macos ver.
file_name_end   = ".json"
# Spatial_data_1spot_paramScanFlatmapOnly_gksMin_0d6_dc_2
# Spatial_data_1spot_NNcon_paramScanFlatmapOnly_gksMin_0d6_dc_2
# Spatial_flatBoth_parasMod2Only_gksMin_0d2_dc_3/
# Spatial_data_1spot_paramScanFlatmapOnly_gksMin_0d2_dc_3
# Spatial_data_1spot_NNcon_paramScanFlatmapOnly_gksMin_0d2_dc_3
dc_inputs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
gks_background = [0.1, 0.2, 0.3,0.4,0.5, 0.6,0.8, 0.9, 1.1, 1.2,1.3,1.4, 1.5]
syn_change_12 = defaultdict(list)
syn_change_21 = defaultdict(list)
MPC = defaultdict(list)
E1BurstTimes = defaultdict(list)
E2BurstTimes = defaultdict(list)

freqDiff = defaultdict(list)
s,t = 4700,4800 # ms


for front_str in ["","NoisePer400_","NoiseOn_","NoisePer100_"]:
# for front_str in ["","NoiseOn_"]:
    if front_str == "":
        noise_freq = 0
    if front_str == "NoisePer400_":
        noise_freq = 2.5
    if front_str == "NoiseOn_":
        noise_freq = 5
    if front_str == "NoisePer100_":
        noise_freq = 10
    
    for dc in [2.0]:
        for gksMax in gks_background:
            for rngseed in [1,2,3,4]:
                exp_name = "dc_{}_gksmax_{}_rngseed_{}".format(round(dc,1),round(gksMax,1),rngseed)
                file_name = "./" + front_str + file_name_start + exp_name + file_name_end
                with open(file_name,'r') as f:
                    net = jsonpickle.decode(json.load(f)) 
                print(file_name)
                print(datetime.now()-simulation_start_time)

                diffMat_21 = net.adjMat[:400,500:900] - net.adjMat0[:400,500:900]
                diffMat_12 = net.adjMat[500:900,:400] - net.adjMat0[500:900,:400]
                
                syn_change_12[noise_freq,gksMax].append(diffMat_12.sum())
                syn_change_21[noise_freq,gksMax].append(diffMat_21.sum())
                net.computeModMPC(s=4000, e=5000)
                MPC[noise_freq,gksMax].append((net.MPC12+net.MPC21)/2.0)
                E1BurstTimes[noise_freq,gksMax].append(net.E1Bursts.size)
                E2BurstTimes[noise_freq,gksMax].append(net.E2Bursts.size)
                freqDiff[noise_freq,gksMax].append((net.E1Bursts.size - net.E2Bursts.size))
#%% plot synaptic change vs gks levels
plt.rcParams.update({'font.size': 20})
fig_synchange = plt.figure(figsize=[6,4],dpi=200)
df = 1.25*pd.DataFrame(syn_change_21).T.rename_axis(['noise_freq','gks_level']).rename(lambda x:x+1, axis = 'columns')

for noise_freq in [0.0,2.5,5.0,10]:
# for noise_freq in [0.0,5.0]:
    plt.errorbar(gks_background,df.loc[noise_freq].mean(axis=1),yerr=0,#df.loc[noise_freq].std(axis=1)/2,
                 label='{} Hz'.format(noise_freq))
    plt.fill_between(gks_background, 
                     df.loc[noise_freq].mean(axis=1)-df.loc[noise_freq].std(axis=1)/2,
                     df.loc[noise_freq].mean(axis=1)+df.loc[noise_freq].std(axis=1)/2,
                     alpha=0.2)
plt.yscale('symlog',linthresh=2,linscale=0.5)
plt.axhline(y = 0, color = 'black', linestyle = '--')
plt.axvline(x = 0.6, color = 'black', linestyle = '--')
plt.ylim(-150,150)
# plt.legend()
plt.xlabel('$g_{Ks}$ (mod 1 only)')
plt.ylabel("$\Delta W$ (mod 1 to 2)")
plt.yticks([-100,-10,-2,2,10,100],["-100%","-10%","-2%","+2%","+10%","+100%"])
plt.xticks(np.arange(0.1,1.6,0.1),rotation=90)
# plt.gca().set_yticklabels([f'{x/100:.0%}' for x in plt.gca().get_yticks()]) 

fig_synchange = plt.figure(figsize=[6,4],dpi=200)
df = 1.25*pd.DataFrame(syn_change_12).T.rename_axis(['noise_freq','gks_level']).rename(lambda x:x+1, axis = 'columns')

for noise_freq in [0.0,2.5,5.0,10]:
# for noise_freq in [0.0,5.0]:
    plt.errorbar(gks_background,df.loc[noise_freq].mean(axis=1),yerr=0,#df.loc[noise_freq].std(axis=1)/2,
                 label='{} Hz'.format(noise_freq))
    plt.fill_between(gks_background, 
                     df.loc[noise_freq].mean(axis=1)-df.loc[noise_freq].std(axis=1)/2,
                     df.loc[noise_freq].mean(axis=1)+df.loc[noise_freq].std(axis=1)/2,
                     alpha=0.2)
plt.yscale('symlog',linthresh=2,linscale=0.5)
plt.axhline(y = 0, color = 'black', linestyle = '--')
plt.axvline(x = 0.6, color = 'black', linestyle = '--')
plt.ylim(-150,150)
# plt.legend()
plt.xlabel('$g_{Ks}$ (mod 1 only)')
plt.ylabel("$\Delta W$ (mod 2 to 1)")
plt.yticks([-100,-10,-2,2,10,100],["-100%","-10%","-2%","+2%","+10%","+100%"])
plt.xticks(np.arange(0.1,1.6,0.1),rotation=90)

# plot mpc vs gks levels
fig_MPC = plt.figure(figsize=[6,4],dpi=200)
df = pd.DataFrame(MPC).T.rename_axis(['noise_freq','gks_level']).rename(lambda x:x+1, axis = 'columns')

for noise_freq in [0.0,2.5,5.0,10]:
# for noise_freq in [0.0,5.0]:
    plt.errorbar(gks_background,df.loc[noise_freq].mean(axis=1),yerr=0,#df.loc[noise_freq].std(axis=1)/2,
                 label='{} Hz'.format(noise_freq))
    plt.fill_between(gks_background, 
                     df.loc[noise_freq].mean(axis=1)-df.loc[noise_freq].std(axis=1)/2,
                     df.loc[noise_freq].mean(axis=1)+df.loc[noise_freq].std(axis=1)/2,
                     alpha=0.2)
plt.ylim(0,1.1)
plt.xlabel('$g_{Ks}$ (mod 1 only)')
plt.axvline(x = 0.6, color = 'black', linestyle = '--')
# plt.legend()
plt.ylabel("MPC (mod level)")
plt.xticks(np.arange(0.1,1.6,0.1),rotation=90)

# plot freq diff vs gks levels
fig_freqdiff = plt.figure(figsize=[6,4],dpi=200)
df = pd.DataFrame(freqDiff).T.rename_axis(['noise_freq','gks_level']).rename(lambda x:x+1, axis = 'columns')

for noise_freq in [0.0,2.5,5.0,10]:
# for noise_freq in [0.0,5.0]:
    plt.errorbar(gks_background,df.loc[noise_freq].mean(axis=1),yerr=0,#df.loc[noise_freq].std(axis=1)/2,
                 label='{} Hz'.format(noise_freq))
    plt.fill_between(gks_background, 
                     df.loc[noise_freq].mean(axis=1)-df.loc[noise_freq].std(axis=1)/2,
                     df.loc[noise_freq].mean(axis=1)+df.loc[noise_freq].std(axis=1)/2,
                     alpha=0.2)
# plt.ylim(0,1)
# plt.legend()
plt.axhline(y = 0, color = 'black', linestyle = '--')
plt.axvline(x = 0.6, color = 'black', linestyle = '--')
plt.xlabel('$g_{Ks}$ (mod 1 only)')
plt.ylabel("$\Delta f$ [Hz]")
plt.xticks(np.arange(0.1,1.6,0.1),rotation=90)
# plt.yscale('symlog')
#%% spot case: gks effect on bottom module learning pattern (choose columns)
# get the data 
simulation_start_time = datetime.now()
#file_name_start = ".\\Spatial_data_gksmin_r2_NNcon\\" # windows ver.
file_name_start = "Spatial_data_1spot_paramScanFlatmapOnly_gksMin_0d6_dc_2/" # macos ver.
file_name_end   = ".json"
# Spatial_data_1spot_paramScanFlatmapOnly_gksMin_0d6_dc_2
# Spatial_data_1spot_NNcon_paramScanFlatmapOnly_gksMin_0d6_dc_2
# Spatial_flatBoth_parasMod2Only_gksMin_0d2_dc_3/
# Spatial_data_1spot_paramScanFlatmapOnly_gksMin_0d2_dc_3
# Spatial_data_1spot_NNcon_paramScanFlatmapOnly_gksMin_0d2_dc_3
dc_inputs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
gks_background = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5]
syn_change_in = defaultdict(list)
syn_change_out = defaultdict(list)
MPC = defaultdict(list)
E1SpikeTimes = defaultdict(list)
E2SpikeTimes = defaultdict(list)
freqDiff = defaultdict(list)

for front_str in ["","NoisePer400_","NoiseOn_","NoisePer100_"]:
# for front_str in ["","NoiseOn_"]:
    if front_str == "":
        noise_freq = 0
    if front_str == "NoisePer400_":
        noise_freq = 2.5
    if front_str == "NoiseOn_":
        noise_freq = 5
    if front_str == "NoisePer100_":
        noise_freq = 10
    
    for dc in [3.5]:
        for gksMax in gks_background:
            for rngseed in [1,2,3,4]:
                exp_name = "dc_{}_gksmax_{}_rngseed_{}".format(round(dc,1),round(gksMax,1),rngseed)
                file_name = "./" + front_str + file_name_start + exp_name + file_name_end
                with open(file_name,'r') as f:
                    net = jsonpickle.decode(json.load(f)) 
                print(file_name)
                print(datetime.now()-simulation_start_time)

                diffMat_21 = net.adjMat[:400,500:900] - net.adjMat0[:400,500:900]
                diffMat_12 = net.adjMat[500:900,:400] - net.adjMat0[500:900,:400]
                
                net.setupSquareWaveDriveOneSpot(r_1 = 6, r_2 = 6)

                idvec = net.sqwave_idvec.copy()
                idvec[400:500] = 0
                idvec[900:]    = 0
                idvec = idvec.flatten()

                id_10 = np.arange(1000)[idvec == 10]
                id_11 = np.arange(1000)[idvec == 11]
                id_20 = np.arange(1000)[idvec == 20]-500
                id_21 = np.arange(1000)[idvec == 21]-500
                
                syn_change_in[noise_freq,gksMax].append(diffMat_21[id_11,:].sum())
                syn_change_out[noise_freq,gksMax].append(diffMat_21[id_10,:].sum())
                net.computeModMPC()
                MPC[noise_freq,gksMax].append((net.MPC12+net.MPC21)/2.0)
                E1SpikeTimes[noise_freq,gksMax].append(net.E1SpikeTimes.size)
                E2SpikeTimes[noise_freq,gksMax].append(net.E2SpikeTimes.size)
                freqDiff[noise_freq,gksMax].append((net.E1SpikeTimes.size - net.E2SpikeTimes.size)/400)
#%% spot case: plot synaptic change vs gks levels
fig_synchange = plt.figure(figsize=[6,4],dpi=200)
df = 1.25*(id_11.size/400)*pd.DataFrame(syn_change_in).T.rename_axis(['noise_freq','gks_level']).rename(lambda x:x+1, axis = 'columns')
plt.figure(figsize=[6,4],dpi=200)
for noise_freq in [0.0,2.5,5.0,10]:
# for noise_freq in [0.0,5.0]:
    plt.errorbar(gks_background,df.loc[noise_freq].mean(axis=1),yerr=df.loc[noise_freq].std(axis=1)/2,label='{} Hz'.format(noise_freq))
plt.yscale('symlog')
plt.ylim(-100,100)
plt.legend()
plt.title("incoming synaptic change inside hotspot")

fig_synchange = plt.figure(figsize=[6,4],dpi=200)
df = 1.25*(id_10.size/400)*pd.DataFrame(syn_change_out).T.rename_axis(['noise_freq','gks_level']).rename(lambda x:x+1, axis = 'columns')
for noise_freq in [0.0,2.5,5.0,10]:
# for noise_freq in [0.0,5.0]:
    plt.errorbar(gks_background,df.loc[noise_freq].mean(axis=1),yerr=df.loc[noise_freq].std(axis=1)/2,label='{} Hz'.format(noise_freq))
plt.yscale('symlog')
plt.ylim(-100,100)
plt.legend()
plt.title("incoming synaptic change outside hotspot")
# plot mpc vs gks levels
fig_MPC = plt.figure(figsize=[6,4],dpi=200)
df = pd.DataFrame(MPC).T.rename_axis(['noise_freq','gks_level']).rename(lambda x:x+1, axis = 'columns')
plt.figure(figsize=[6,4],dpi=200)
for noise_freq in [0.0,2.5,5.0,10]:
# for noise_freq in [0.0,5.0]:
    plt.errorbar(gks_background,df.loc[noise_freq].mean(axis=1),yerr=df.loc[noise_freq].std(axis=1)/2,label='{} Hz'.format(noise_freq))
plt.ylim(0,1)
plt.legend()
plt.title("MPC")
# plot freq diff vs gks levels
fig_freqdiff = plt.figure(figsize=[6,4],dpi=200)
df = pd.DataFrame(freqDiff).T.rename_axis(['noise_freq','gks_level']).rename(lambda x:x+1, axis = 'columns')

for noise_freq in [0.0,2.5,5.0,10]:
# for noise_freq in [0.0,5.0]:
    plt.errorbar(gks_background,df.loc[noise_freq].mean(axis=1),yerr=df.loc[noise_freq].std(axis=1)/2,label='{} Hz'.format(noise_freq))
# plt.ylim(0,1)
plt.legend()
plt.title("freq diff")
plt.yscale('symlog')

#%% F-I curve
from NeuroNet import NeuroNet
from matplotlib import cm
DCpoints = np.linspace(-0.5,3.5,100).reshape(-1,1)
gKsPoints = np.linspace(0,1.5,6).reshape(-1,1)
plt.figure(figsize=[6,4],dpi=200)
cmap = cm.get_cmap('viridis')

plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
for gKs in gKsPoints:
    net = NeuroNet(neuroNum=DCpoints.size,tEnd = 4000) 
    net.gKs = gKs*np.ones(net.neuroNum).reshape(-1,1)
    net.Idrive = DCpoints.copy()
    net.zerolikeInitialStates()
    net.runSimulation(isNet=False)
    # get freq for each DCpoints
    freq = []
    for idx in range(net.neuroNum):
        intervals = np.diff(net.spikeTimes[0,net.spikeTimes[1,:] == idx])
        if intervals.size<4:
            freq.append(0)
        else:
            freq.append(1000/intervals[3:].mean())
    freq = np.array(freq)
    plt.plot(DCpoints[freq>0], freq[freq>0], 
             label = "$g_{Ks}$"+"={}".format(gKs), 
             color = cmap(net.gKs[idx]/1.5))
    plt.legend()
    plt.xlim(-0.5,3.5)


#%% SFA
from NeuroNet import NeuroNet
from matplotlib import cm
# import matplotlib.pyplot as plt
# import numpy as np
numSubplots = 6
net = NeuroNet(neuroNum=numSubplots,tEnd=400)
net.gKs = np.linspace(0,1.5,numSubplots).reshape(-1,1)
net.Idrive = 1.5*np.ones(net.neuroNum).reshape(-1,1)
net.zerolikeInitialStates(logV = True)
vPoints_before = net.vPoints.copy()
net.runSimulation(isNet=False,logV=True)
net.tPoints = np.concatenate((net.tPoints_before,net.tPoints))
net.vPoints = np.hstack((vPoints_before,net.vPoints))
# Plotting
fig, axs = plt.subplots(net.neuroNum,sharex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='Arial')
plt.rc('xtick', labelsize=20)
cmap = cm.get_cmap('viridis')
# colors = [[0, 0.4470, 0.7410],
#           [0.8500, 0.3250, 0.0980],
#           [0.9290, 0.6940, 0.1250],
#           [0.4940, 0.1840, 0.5560]]

# fig.suptitle('')
for idx in range(net.neuroNum):
    axs[idx].plot(net.tPoints,
                  net.vPoints[idx,:],
                  color = cmap(net.gKs[idx]/1.5))
    
    axs[idx].set_xlim([-50,net.tPoints[-1]])
    # axs[idx].set_ylabel('$g_{K_s} = $' + '{}'.format(net.gKs.squeeze()[idx]) + '\n V [mV]')
    # axs[idx].set_ylabel('V [mV]')
    axs[idx].set_yticks([])
    axs[idx].set_xticks(np.linspace(0,net.tEnd,5))
    
plt.tight_layout()


#%% adjMat diff module 1 to 2
diffMat_12 = net.adjMat[500:900,:400] - net.adjMat0[500:900,:400]

plt.figure()
plt.imshow(diffMat_12,cmap = "seismic",
           vmax = max(abs(diffMat_12.max()),abs(diffMat_12.min())),
           vmin = -max(abs(diffMat_12.max()),abs(diffMat_12.min())),
           origin = 'lower')
plt.colorbar()
plt.title("from 1 to 2")
plt.show()


#%% Incoming synaptic diff visual
exp_name = "test"
plt.figure()
presynWeightDiff = np.vstack((diffMat_21.sum(axis = 1).reshape(20,20),
                              np.zeros((5,20)),
                              diffMat_12.sum(axis = 1).reshape(20,20)))
                              
diffMin = presynWeightDiff.min()
diffMax = presynWeightDiff.max()      
plt.title('{}\nIncoming synapses diff ranging [{:.3f},{:.3f}]'.format(exp_name,diffMin,diffMax))
plt.imshow(presynWeightDiff,cmap='seismic',
           vmax=max(abs(diffMin),abs(diffMax)),
           vmin=-max(abs(diffMin),abs(diffMax)),
           origin='lower')
plt.colorbar()
plt.show()
plt.tight_layout()
#%% MPC scan
simulation_start_time = datetime.now()
# file_name_start = ".\\Spatial_data_NNcon_dcboth_2spot\\" # windows ver.
uncoupled_file_name_start = "./NoisePer100_Spatial_flatBoth_parasMod2Only_gksMin_0d6_dc_2_uncoupled/" # macos ver.
file_name_start = "./NoisePer100_Spatial_flatBoth_parasMod2Only_gksMin_0d6_dc_2/" # macos ver.
file_name_end   = ".json"

dc_inputs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
gks_background = [0, 0.3, 0.6, 0.9, 1.2, 1.5]

MPC12 = defaultdict(list)
MPC21 = defaultdict(list)
E1SpikeTimes = defaultdict(list)
E2SpikeTimes = defaultdict(list)
uncoupled_E1SpikeTimes = defaultdict(list)
uncoupled_E2SpikeTimes = defaultdict(list)
E1BurstTimes = defaultdict(list)
E2BurstTimes = defaultdict(list)

eps = 0

for dc in dc_inputs:
    for gksMax in gks_background:
        for rngseed in [1,2,3,4]:
            exp_name = "dc_{}_gksmax_{}_rngseed_{}".format(round(dc,1),round(gksMax,1),rngseed)
            file_name = file_name_start + exp_name + file_name_end
            with open(file_name,'r') as f:
                net = jsonpickle.decode(json.load(f)) 
            net.computeModMPC()
            # MPC12[dc,gksMax].append(net.MPC12)
            # MPC21[dc,gksMax].append(net.MPC21)
            E1SpikeTimes[dc,gksMax].append(net.E1SpikeTimes.size)
            E2SpikeTimes[dc,gksMax].append(net.E2SpikeTimes.size)
            
            
            file_name = uncoupled_file_name_start + exp_name + file_name_end
            with open(file_name,'r') as f:
                net = jsonpickle.decode(json.load(f)) 
            net.computeModMPC()
            # MPC12[dc,gksMax].append(net.MPC12)
            # MPC21[dc,gksMax].append(net.MPC21)
            uncoupled_E1SpikeTimes[dc,gksMax].append(net.E1SpikeTimes.size)
            uncoupled_E2SpikeTimes[dc,gksMax].append(net.E2SpikeTimes.size)
            
            # E1BurstTimes[dc,gksMax].append(net.E1Bursts.size)
            # E2BurstTimes[dc,gksMax].append(net.E2Bursts.size)
            # syn_change_12[dc,gksMax].append((diffMat_12>eps).sum() - (diffMat_12<-eps).sum())
            # syn_change_21[dc,gksMax].append((diffMat_21>eps).sum() - (diffMat_21<-eps).sum())
            
            print(file_name)
            print(datetime.now()-simulation_start_time)
#%% data transform
df1 = pd.DataFrame(E1SpikeTimes).T.rename_axis(['dc','gks_level']).rename(lambda x:x+1, axis = 'columns')
df2 = pd.DataFrame(E2SpikeTimes).T.rename_axis(['dc','gks_level']).rename(lambda x:x+1, axis = 'columns')
udf1 = pd.DataFrame(uncoupled_E1SpikeTimes).T.rename_axis(['dc','gks_level']).rename(lambda x:x+1, axis = 'columns')
udf2 = pd.DataFrame(uncoupled_E2SpikeTimes).T.rename_axis(['dc','gks_level']).rename(lambda x:x+1, axis = 'columns')
#%% colormap
plt.figure()
sns.heatmap((df1-df2-udf1+udf2).mean(axis=1).unstack(level=0),center=0)
plt.title("(E1-E2)-(UE1-UE2) SpikeCounts")

#%% data transform
df1 = pd.DataFrame(E1BurstTimes).T.rename_axis(['dc','gks_level']).rename(lambda x:x+1, axis = 'columns')
df2 = pd.DataFrame(E2BurstTimes).T.rename_axis(['dc','gks_level']).rename(lambda x:x+1, axis = 'columns')

#%% colormap
plt.figure()
sns.heatmap((df1-df2).mean(axis=1).unstack(level=0), center=0)
plt.title("E1-E2 BurstCounts")
#%% data transform
df = pd.DataFrame(MPC12).T.rename_axis(['dc','gks_level']).rename(lambda x:x+1, axis = 'columns')
#%% colormap
plt.figure()
sns.heatmap(df.mean(axis=1).unstack(level=0),vmin=0, vmax=1)
plt.title("MPC12")
# plt.title("synaptic change (vote diff) from 2 to 1")


#%% parameter scan for one spot (DC only)
simulation_start_time = datetime.now()
# file_name_start = ".\\Spatial_data_NNcon_dcboth_2spot\\" # windows ver.
file_name_start = "./NoisePer100_Spatial_flatBoth_parasMod2Only_gksMin_0d6_dc_2/" # macos ver.
file_name_end   = ".json"

dc_inputs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
gks_background = [0, 0.1, 0.2, 0.3, 0.6, 0.9, 1.2, 1.5]

dc_inputs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
gks_background = [0, 0.3, 0.6, 0.9, 1.2, 1.5]
inSynDiff = defaultdict(list) # outgoing
outSynDiff = defaultdict(list) # incoming

eps = 0

for dc in dc_inputs:
    for gksMax in [1.5]:
        for rngseed in [1]:
            exp_name = "dc_{}_gksmax_{}_rngseed_{}".format(round(dc,1),round(gksMax,1),rngseed)
            file_name = file_name_start + exp_name + file_name_end
            with open(file_name,'r') as f:
                net = jsonpickle.decode(json.load(f)) 
            diffMat_21 = net.adjMat[:400,500:900] -net.adjMat0[:400,500:900]
            diffMat_12 = net.adjMat[500:900,:400] - net.adjMat0[500:900,:400]
            inSD = pd.DataFrame(diffMat_21.sum(axis = 1),index = net.distToR_1[:400].flatten()).rename_axis('d').groupby('d').mean()
            outSD = pd.DataFrame(diffMat_12.sum(axis = 0),index = net.distToR_1[:400].flatten()).rename_axis('d').groupby('d').mean()
            inSynDiff[dc].append(inSD.loc[:,0].to_numpy())
            outSynDiff[dc].append(outSD.loc[:,0].to_numpy())
            print(file_name)
            print(datetime.now()-simulation_start_time)
dist2center = inSD.index.to_numpy()
#%% data trans           
df_mean = pd.DataFrame({k: np.mean(v,axis=0) for k,v in inSynDiff.items()}).T.rename_axis('dc').rename(lambda x:dist2center[x], axis = 'columns').T
df_sem = pd.DataFrame({k: 0.5*np.std(v,axis=0) for k,v in inSynDiff.items()}).T.rename_axis('dc').rename(lambda x:dist2center[x], axis = 'columns').T
#%% plotting
plt.figure()
df_mean.plot()

plt.xlabel("distance to center")
plt.ylabel('incoming synaptic change')

#%% data trans           
df = pd.DataFrame({k: np.mean(v,axis=0) for k,v in outSynDiff.items()}).T.rename_axis('dc').rename(lambda x:dist2center[x], axis = 'columns')

#%% plotting
plt.figure()
df.T.plot()
plt.xlabel("distance to center")
plt.ylabel('outgoing synaptic change')

