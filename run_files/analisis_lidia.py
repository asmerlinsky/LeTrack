#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:04:44 2021

@author: lidia
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import json
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmax
from varname import nameof

with open('marker_dict.json', 'r') as fp:
    json_dict = json.load(fp)

#los segmentos posteriores no fueron medidos
#LEECH 00
f21_03_19_0 = pd.read_csv("output/tracked_data/21-03-19_0.csv", header=None)
f_21_03_19_0= np.transpose(f21_03_19_0)
time=list(range(1, len(f_21_03_19_0)+1))
# timeT=[i * 0.033 for i in time]
# f_21_03_19_0.insert(loc=0, column="time", value=timeT)

#LEECH 1- SEGMENTO POSTERIOR MUY LARGO
x=json_dict["21-04-26_00.AVI"]
f21_04_26_00 = pd.read_csv("output/tracked_data/21-04-26_00.csv", header=None)
f_21_04_26_00= np.transpose(f21_04_26_00)
time=list(range(1, len(f_21_04_26_00)+1))
# timeT=[i * 0.033 for i in time]
# f_21_04_26_00.insert(loc=0, column="time", value=timeT)

#LEECH 04 
x=json_dict["21-04-28_06.AVI"]
f21_04_28_06 = pd.read_csv("output/tracked_data/21-04-28_06.csv", header=None)
f_21_04_28_06= np.transpose(f21_04_28_06)
time=list(range(1, len(f_21_04_28_06)+1))
# timeT=[i * 0.033 for i in time]
# f_21_04_28_06.insert(loc=0, column="time", value=timeT)


#####
#este es el script que armé para analizar
# dividí las marcas en dos: anteriores (hasta mid_column) y posteriores (mid_column a end_column)
#file es el nombre universal y para cada animal reemplazo con el nombre que corresponde que es f_identidad del file.
file=f_21_04_26_00  #aca hay que reemplazar por el nombre de cada file
file_dict = {'f_21_04_26_00': f_21_04_26_00, 'f_21_04_28_06':f_21_04_28_06, 'f_21_03_19_0': f_21_03_19_0}

for key, file in file_dict.items():
    end_column=len(file.columns)-1
    mid_column=round(end_column/2)
    time=list(range(1, len(file)+1))
    timeT=[i * 0.033 for i in time]

    file_ff=gaussian_filter1d(file, sigma=5, axis=0)
    file_f=pd.DataFrame(file_ff)
    file_f.insert(loc=0, column="time", value=timeT)

    file_dd=file_f.diff()
    file_d=file_dd/0.033
    del file_d["time"]
    file_d.insert(loc=0, column="time", value=timeT)


    plt.figure()  #una figura para cada ejemplo
    for i in range (0,mid_column+1):
        ax1 = plt.subplot(211)
        plt.plot(file_f["time"], file_f[i])
        plt.setp(ax1.get_xticklabels(), fontsize=6)
        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(file_f["time"], file_d[i])
        plt.legend(loc=5, bbox_to_anchor=(1, 1))
        plt.suptitle(key)
    plt.show()
