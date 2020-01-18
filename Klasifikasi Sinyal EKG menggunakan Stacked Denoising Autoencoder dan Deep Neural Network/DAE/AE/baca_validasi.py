# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:59:53 2019

@author: Asus
"""

import numpy as np
import pickle

nam = 'data/Validasi'
pickle_in = open(nam,'rb')
valisf = pickle.load(pickle_in)

# =============================================================================
# sens = [vali[0][6][1]]+[vali[1][6][1]]+[vali[2][6][1]]+[vali[3][6][1]]+[vali[4][6][1]]
# avg_sens = np.average(sens)
# spec = [vali[0][7][1]]+[vali[1][7][1]]+[vali[2][7][1]]+[vali[3][7][1]]+[vali[4][7][1]]
# avg_spec = np.average(spec)
# prec = [vali[0][8][1]]+[vali[1][8][1]]+[vali[2][8][1]]+[vali[3][8][1]]+[vali[4][8][1]]
# avg_prec = np.average(prec)
# fscr = [vali[0][9][1]]+[vali[1][9][1]]+[vali[2][9][1]]+[vali[3][9][1]]+[vali[4][9][1]]
# avg_fscr = np.average(fscr)
# errt = [vali[0][10][1]]+[vali[1][10][1]]+[vali[2][10][1]]+[vali[3][10][1]]+[vali[4][10][1]]
# avg_errt = np.average(errt)
# accu = [vali[0][11][1]]+[vali[1][11][1]]+[vali[2][11][1]]+[vali[3][11][1]]+[vali[4][11][1]]
# avg_accu = np.average(accu) 
#    
# lotr = [vali[0][0]]+[vali[1][0]]+[vali[2][0]]+[vali[3][0]]+[vali[4][0]]
# avg_lotr = np.average(lotr)
# lote = [vali[0][1]]+[vali[1][1]]+[vali[2][1]]+[vali[3][1]]+[vali[4][1]]
# avg_lote = np.average(lote)
# actr = [vali[0][2]]+[vali[1][2]]+[vali[2][2]]+[vali[3][2]]+[vali[4][2]]
# avg_actr = np.average(actr)
# acte = [vali[0][3]]+[vali[1][3]]+[vali[2][3]]+[vali[3][3]]+[vali[4][3]]
# avg_acte = np.average(acte)
# =============================================================================