# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 20:13:39 2019

@author: Asus
"""

import numpy as np
import wfdb
import os
import glob
import matplotlib.pyplot as plt
from wfdb.processing import normalize_bound

t1 = 90
t2 = 162

beats = []
beats_noisy = []
labels = []
idy = 0 
paths_data = ['../Dataset/*/*e24.dat','../Dataset/*/*e_6.dat']

for idx, dta in enumerate(paths_data):
    dataa = glob.glob(dta)
    for pathdata in dataa:
        print('before ' + pathdata)
        pathdata = os.path.splitext(pathdata)[0]
        print('after ' + pathdata)
        record = wfdb.rdrecord(pathdata, sampfrom=0)
        record_dict = record.__dict__
        
        annotation = wfdb.rdann(pathdata, 'atr', sampfrom=0)
        ann_dict = annotation.__dict__
        
        sinal = record_dict['p_signal'][:,0]
        peaks = ann_dict['sample']
        symbol = ann_dict['symbol']
        idy = idy+1
        print(idy)
        for idz, peak in enumerate(peaks):
            beats.append([])
            labels.append([])
            if (peak - t1) >= 0 and (peak + t2) <650000: 
                    beat = sinal[peak - t1 : peak + t2]
                    #beat = normalize_bound(beat, lb=0, ub=1)
                    beats[idy].append(beat)
                    labels[idy].append(symbol[idz])
        beats[idy] = np.array(beats[idy])
        labels[idy] = np.array(labels[idy]).astype(str)
        
beats_noisy = np.concatenate((beats[3],beats[4]),axis=0)
beats = np.concatenate((beats[1],beats[2]),axis=0)
labels = np.concatenate((labels[1],labels[2]),axis=0)

N=[]
V=[]
R=[]
A=[]
P=[]
oth=[]

feature=[]
feature_noisy=[]
label=[]
other=[]
other_noisy=[]
other_label=[]

for i in range(len(labels)):
    if(labels[i] == 'N'):
        N.append(i)
        feature.append(beats[i])
        feature_noisy.append(beats_noisy[i])
        label.append(0)
    elif(labels[i] == 'V'):
        V.append(i)
        feature.append(beats[i])
        feature_noisy.append(beats_noisy[i])
        label.append(1)
    elif(labels[i] == 'R'):
        R.append(i)
        feature.append(beats[i])
        feature_noisy.append(beats_noisy[i])
        label.append(2)
    elif(labels[i] == 'A'):
        A.append(i)
        feature.append(beats[i])
        feature_noisy.append(beats_noisy[i])
        label.append(3)
    elif(labels[i] == 'x'):
        P.append(i)
        feature.append(beats[i])
        feature_noisy.append(beats_noisy[i])
        label.append(4)
    else:
        oth.append(i)
        other.append(beats[i])
        other_noisy.append(beats_noisy[i])
        other_label.append(labels[i])
        print(i,labels[i])

feature = np.array(feature)
feature_noisy = np.array(feature_noisy)
label = np.array(label)
other = np.array(other)
other_noisy = np.array(other_noisy)
other_label = np.array(other_label)

#np.save('../data/feature.npy', feature)
#np.save('../data/feature_noisy.npy', feature_noisy)
#np.save('../data/label.npy', label)
#np.save('../data/other.npy', other)
#np.save('../data/other_noisy.npy', other_noisy)
#np.save('../data/other_label.npy', other_label)        