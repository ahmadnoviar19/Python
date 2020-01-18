# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 20:13:39 2019

@author: Asus
"""

import numpy as np
import wfdb
import os
import glob
from wfdb.processing import normalize_bound

t1 = 90
t2 = 162

beats = []
labels = []
idy = 0 
paths_data = ['Dataset/*.dat']

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
            if (peak - t1) >= 0 and (peak + t2) <650000: 
                beat = sinal[peak - t1 : peak + t2]
                beat = normalize_bound((beat), lb=0,ub=1)        
                beats.append(beat)
                labels.append(symbol[idz])
                
beats = np.array(beats)
labels = np.array(labels).astype(str)

N=0
abn=0

feature=[]
label=[]

for i in range(len(labels)):
    if(labels[i] == 'N' or labels[i] == 'L' or labels[i] == 'R' or labels[i] == 'A' or labels[i] == 'a' or labels[i] == 'J' or labels[i] == 'S'):
        N=N+1
        feature.append(beats[i])
        label.append(0)
    else:
        abn=abn+1
        feature.append(beats[i])
        label.append(1)
        print(i,labels[i])

feature = np.array(feature)
label = np.array(label)

# dari sini untuk npy
np.save('feature.npy', feature)
np.save('label.npy', label)

# dari sini untuk CSV
np.savetxt("feature.csv", feature, delimiter=",")
np.savetxt("label.csv", label, delimiter=",")
