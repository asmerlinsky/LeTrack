#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 18:13:30 2021

@author: lidia
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import json
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmax
from scipy.signal import find_peaks


#LEECH 02 
#x=json_dict["21-04-26_27.AVI"]
f21_04_26_27 = pd.read_csv("output_latest/tracked_data/21-04-26_27.csv", header=None)
# f21_04_26_27 = pd.read_csv("/Users/lidia/Dropbox/Data/Agustin/crawling video/tracked_data/21-04-26_27.csv", header=None)
f_21_04_26_27= np.transpose(f21_04_26_27)
time=list(range(1, len(f_21_04_26_27)+1))
timeT=[i * 0.033 for i in time]
f_21_04_26_27.insert(loc=0, column="time", value=timeT)

#LEECH 03 
#x=json_dict["21-04-28_02.AVI"]
f21_04_28_02 = pd.read_csv("output_latest/tracked_data/21-04-28_00.csv", header=None)
# f21_04_28_02 = pd.read_csv("/Users/lidia/Dropbox/Data/Agustin/crawling video/tracked_data/21-04-28_00.csv", header=None)
f_21_04_28_02= np.transpose(f21_04_28_02)
time=list(range(1, len(f_21_04_28_02)+1))
timeT=[i * 0.033 for i in time]
f_21_04_28_02.insert(loc=0, column="time", value=timeT)

#LEECH 04 
#x=json_dict["21-04-28_06.AVI"]
f21_04_28_06 = pd.read_csv("output_latest/tracked_data/21-04-28_06.csv", header=None)
# f21_04_28_06 = pd.read_csv("/Users/lidia/Dropbox/Data/Agustin/crawling video/tracked_data/21-04-28_06.csv", header=None)
f_21_04_28_06= np.transpose(f21_04_28_06)
time=list(range(1, len(f_21_04_28_06)+1))
timeT=[i * 0.033 for i in time]
f_21_04_28_06.insert(loc=0, column="time", value=timeT)

#LEECH 06 
#x=json_dict["21-04-30_04.AVI"]
# f21_04_30_04 = pd.read_csv("/Users/lidia/Dropbox/Data/Agustin/crawling video/tracked_data/21-04-30_04.csv", header=None)
f21_04_30_04 = pd.read_csv("output_latest/tracked_data/21-04-30_04.csv", header=None)
f_21_04_30_04= np.transpose(f21_04_30_04)
time=list(range(1, len(f_21_04_30_04)+1))
timeT=[i * 0.033 for i in time]
f_21_04_30_04.insert(loc=0, column="time", value=timeT)

####################
####################

file=f_21_04_26_27
time=list(range(1, len(file)+1))
timeT=[i * 0.033 for i in time]
#end_column=len(file.columns)-1
#mid_column=round(end_column/2)

file_ff=gaussian_filter1d(file,sigma=10,axis=0)
file_f=pd.DataFrame(file_ff)
del(file_f[0])
#file_f.insert(loc=0, column="time", value=timeT)  NO INCLUYO EL TIEMPO

#diferencio cada segmento
file_dd=file_f.diff()
file_d=file_dd/0.033

#plot de cada segmento superpuesto
plt.figure(1)
plt.axhline(0, color='black')
for i in range(1,len(file_f.columns)):
    ax1 = plt.subplot(211)
    plt.plot(timeT, file_f[i])
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(timeT, file_d[i])
    plt.legend(loc=5, bbox_to_anchor=(1, 1))

#encontrar picos en el segmento medio de elongacion y contraccion
peaks_E_mid,_= find_peaks(file_d[5], distance=100, height=10)
peaks_C_mid,_= find_peaks(-file_d[5], distance=100, height=20)

#graficar para ver donde calzan los maximos
plt.figure(2)
plt.plot(file_d[5])
plt.plot(peaks_E_mid, file_d[5][peaks_E_mid], "x", color = 'r')
plt.plot(peaks_C_mid, file_d[5][peaks_C_mid], "x", color = 'b')
plt.show()

#comienzo y fin de cada ciclo
cycles_on=peaks_C_mid[0:len(peaks_C_mid)-1]
cycles_off=peaks_C_mid[1:len(peaks_C_mid)]

#recortar cada ciclo en un diccionario, cada linea es un ciclo y cada columna un segmento
file_d_cycles={}
for i in range(len(cycles_on)):
    data=file_d[cycles_on[i]:cycles_off[i]]
    file_d_cycles[i]=data    

#generar un eje x que vaya de 0 a 1
file_d_x={}
for i in range(len(cycles_on)):
    dx=1/(len(file_d_cycles[i])-1)
    data_xx=np.arange(0,1+dx,dx)
    if len(data_xx) == len(file_d_cycles[i]):
        data_x=data_xx
    else:
        data_x=data_xx[0:-1]
    file_d_x[i]=data_x


#plotear por segmento superponiendo ciclos
#segmentos anteriores
plt.figure(11)
plt.axhline(0, color='black')
for i in range(1,10):
    ax1 = plt.subplot(511)
    plt.plot(file_d_x[i], file_d_cycles[i][1])
    plt.title('segments 1-2-3-4-5')
    ax2 = plt.subplot(512)
    plt.plot(file_d_x[i], file_d_cycles[i][2])
    ax3 =plt.subplot(513)
    plt.plot(file_d_x[i], file_d_cycles[i][3])
    ax4 =plt.subplot(514)
    plt.plot(file_d_x[i], file_d_cycles[i][4])
    ax5 =plt.subplot(515)
    plt.plot(file_d_x[i], file_d_cycles[i][5])

#segmentos posteriores
plt.figure(12)
plt.axhline(0, color='black')
for i in range(1,10):
    ax1 = plt.subplot(511)
    plt.plot(file_d_x[i], file_d_cycles[i][6])
    plt.title('segments 6-7-8-9-10')
    ax2 = plt.subplot(512)
    plt.plot(file_d_x[i], file_d_cycles[i][7])
    ax3 =plt.subplot(513)
    plt.plot(file_d_x[i], file_d_cycles[i][8])
    ax4 =plt.subplot(514)
    plt.plot(file_d_x[i], file_d_cycles[i][9])
    ax5 =plt.subplot(515)
    plt.plot(file_d_x[i], file_d_cycles[i][10])

#segmentos medios
plt.figure(13)
plt.axhline(0, color='black')
for i in range(1,10):
    ax1 = plt.subplot(511)
    plt.plot(file_d_x[i], file_d_cycles[i][3])
    plt.title('segments 3-4-5-6-7')
    ax2 = plt.subplot(512)
    plt.plot(file_d_x[i], file_d_cycles[i][4])
    ax3 =plt.subplot(513)
    plt.plot(file_d_x[i], file_d_cycles[i][5])
    ax4 =plt.subplot(514)
    plt.plot(file_d_x[i], file_d_cycles[i][6])
    ax5 =plt.subplot(515)
    plt.plot(file_d_x[i], file_d_cycles[i][7])

#averaging across ciclos NO ES POSIBLE PORQUE CADA CICLO TIENE UN DIFERENTE NRO DE PUNTOS

############
############
############


file=f_21_04_28_02
time=list(range(1, len(file)+1))
timeT=[i * 0.033 for i in time]
#end_column=len(file.columns)-1
#mid_column=round(end_column/2)

file_ff=gaussian_filter1d(file,sigma=10,axis=0)
file_f=pd.DataFrame(file_ff)
del(file_f[0])
file_f.insert(loc=0, column="time", value=timeT)

#diferencio cada segmento
file_dd=file_f.diff()
file_d=file_dd/0.033

#plot de cada segmento superpuesto
plt.figure(1)
plt.axhline(0, color='black')
for i in range(1,len(file_f.columns)):
    ax1 = plt.subplot(211)
    plt.plot(timeT, file_f[i])
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(timeT, file_d[i])
    plt.legend(loc=5, bbox_to_anchor=(1, 1))

#encontrar picos en el segmento medio de elongacion y contraccion
peaks_E_mid,_= find_peaks(file_d[5], distance=100, height=5)
peaks_C_mid,_= find_peaks(-file_d[5], distance=100, height=10)

#graficar para ver donde calzan los maximos
plt.figure(2)
plt.plot(file_d[5])
plt.plot(peaks_E_mid, file_d[5][peaks_E_mid], "x", color = 'r')
plt.plot(peaks_C_mid, file_d[5][peaks_C_mid], "x", color = 'b')
plt.show()

#comienzo y fin de cada ciclo
cycles_on=peaks_C_mid[0:len(peaks_C_mid)-1]
cycles_off=peaks_C_mid[1:len(peaks_C_mid)]

#recortar cada ciclo en un diccionario, cada linea es un ciclo y cada columna un segmento
file_d_cycles={}
for i in range(len(cycles_on)):
    data=file_d[cycles_on[i]:cycles_off[i]]
    file_d_cycles[i]=data    

#generar un eje x que vaya de 0 a 1
file_d_x={}
for i in range(len(cycles_on)):
    dx=1/(len(file_d_cycles[i])-1)
    data_xx=np.arange(0,1+dx,dx)
    if len(data_xx) == len(file_d_cycles[i]):
        data_x=data_xx
    else:
        data_x=data_xx[0:-1]
    file_d_x[i]=data_x


#plotear por segmento superponiendo ciclos
#segmentos anteriores
plt.figure(11)
plt.axhline(0, color='black')
for i in range(5,9):
    ax1 = plt.subplot(511)
    plt.plot(file_d_x[i], file_d_cycles[i][1])
    plt.title('segments 1-2-3-4-5')
    ax2 = plt.subplot(512)
    plt.plot(file_d_x[i], file_d_cycles[i][2])
    ax3 =plt.subplot(513)
    plt.plot(file_d_x[i], file_d_cycles[i][3])
    ax4 =plt.subplot(514)
    plt.plot(file_d_x[i], file_d_cycles[i][4])
    ax5 =plt.subplot(515)
    plt.plot(file_d_x[i], file_d_cycles[i][5])
#segmentos posteiores
plt.figure(12)
plt.axhline(0, color='black')
for i in range(5,9):
    ax1 = plt.subplot(411)
    plt.plot(file_d_x[i], file_d_cycles[i][6])
    plt.title('segments 6-7-8-9-10')
    ax2 = plt.subplot(412)
    plt.plot(file_d_x[i], file_d_cycles[i][7])
    ax3 =plt.subplot(413)
    plt.plot(file_d_x[i], file_d_cycles[i][8])
    ax4 =plt.subplot(414)
    plt.plot(file_d_x[i], file_d_cycles[i][9])
    

#segmentos medios
plt.figure(13)
plt.axhline(0, color='black')
for i in range(1,10):
    ax1 = plt.subplot(511)
    plt.plot(file_d_x[i], file_d_cycles[i][3])
    plt.title('segments 3-4-5-6-7')
    ax2 = plt.subplot(512)
    plt.plot(file_d_x[i], file_d_cycles[i][4])
    ax3 =plt.subplot(513)
    plt.plot(file_d_x[i], file_d_cycles[i][5])
    ax4 =plt.subplot(514)
    plt.plot(file_d_x[i], file_d_cycles[i][6])
    ax5 =plt.subplot(515)
    plt.plot(file_d_x[i], file_d_cycles[i][7])

############
############
############

file=f_21_04_28_06
time=list(range(1, len(file)+1))
timeT=[i * 0.033 for i in time]
#end_column=len(file.columns)-1
#mid_column=round(end_column/2)

file_ff=gaussian_filter1d(file,sigma=10,axis=0)
file_f=pd.DataFrame(file_ff)
del(file_f[0])
file_f.insert(loc=0, column="time", value=timeT)

#diferencio cada segmento
file_dd=file_f.diff()
file_d=file_dd/0.033

#plot de cada segmento superpuesto
plt.figure(1)
plt.axhline(0, color='black')
for i in range(1,len(file_f.columns)):
    ax1 = plt.subplot(211)
    plt.plot(timeT, file_f[i])
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(timeT, file_d[i])
    plt.legend(loc=5, bbox_to_anchor=(1, 1))

#encontrar picos en el segmento medio de elongacion y contraccion
peaks_E_mid,_= find_peaks(file_d[6], distance=100, height=20)
peaks_C_mid,_= find_peaks(-file_d[6], distance=100, height=30)

#graficar para ver donde calzan los maximos
plt.figure(2)
plt.plot(file_d[5])
plt.plot(peaks_E_mid, file_d[5][peaks_E_mid], "x", color = 'r')
plt.plot(peaks_C_mid, file_d[5][peaks_C_mid], "x", color = 'b')
plt.show()

#comienzo y fin de cada ciclo
cycles_on=peaks_C_mid[0:len(peaks_C_mid)-1]
cycles_off=peaks_C_mid[1:len(peaks_C_mid)]

#recortar cada ciclo en un diccionario, cada linea es un ciclo y cada columna un segmento
file_d_cycles={}
for i in range(len(cycles_on)):
    data=file_d[cycles_on[i]:cycles_off[i]]
    file_d_cycles[i]=data    

#generar un eje x que vaya de 0 a 1
file_d_x={}
for i in range(len(cycles_on)):
    dx=1/(len(file_d_cycles[i])-1)
    data_xx=np.arange(0,1+dx,dx)
    if len(data_xx) == len(file_d_cycles[i]):
        data_x=data_xx
    else:
        data_x=data_xx[0:-1]
    file_d_x[i]=data_x

#plotear por segmento superponiendo ciclos
#segmentos anteriores
plt.figure(11)
plt.axhline(0, color='black')
for i in range(1,10):
    ax1 = plt.subplot(511)
    plt.plot(file_d_x[i], file_d_cycles[i][1])
    plt.title('segments 1-2-3-4-5')
    ax2 = plt.subplot(512)
    plt.plot(file_d_x[i], file_d_cycles[i][2])
    ax3 =plt.subplot(513)
    plt.plot(file_d_x[i], file_d_cycles[i][3])
    ax4 =plt.subplot(514)
    plt.plot(file_d_x[i], file_d_cycles[i][4])
    ax5 =plt.subplot(515)
    plt.plot(file_d_x[i], file_d_cycles[i][5])

plt.figure(12)
plt.axhline(0, color='black')
for i in range(1,10):
    ax1 = plt.subplot(511)
    plt.plot(file_d_x[i], file_d_cycles[i][6])
    plt.title('segments 6-7-10-11-12')
    ax2 = plt.subplot(512)
    plt.plot(file_d_x[i], file_d_cycles[i][7])
    ax3 =plt.subplot(513)
    plt.plot(file_d_x[i], file_d_cycles[i][10])
    ax4 =plt.subplot(514)
    plt.plot(file_d_x[i], file_d_cycles[i][11])
    ax5 =plt.subplot(515)
    plt.plot(file_d_x[i], file_d_cycles[i][12])

#figura segmentos medios
plt.figure(13)
plt.axhline(0, color='black')
for i in range(1,10):
    ax1 = plt.subplot(511)
    plt.plot(file_d_x[i], file_d_cycles[i][3])
    plt.title('segments 3-4-5-6-7')
    ax2 = plt.subplot(512)
    plt.plot(file_d_x[i], file_d_cycles[i][4])
    ax3 =plt.subplot(513)
    plt.plot(file_d_x[i], file_d_cycles[i][5])
    ax4 =plt.subplot(514)
    plt.plot(file_d_x[i], file_d_cycles[i][6])
    ax5 =plt.subplot(515)
    plt.plot(file_d_x[i], file_d_cycles[i][7])
    
 ############
############
############
    
file = f_21_04_30_04
time=list(range(1, len(file)+1))
timeT=[i * 0.033 for i in time]
#end_column=len(file.columns)-1
#mid_column=round(end_column/2)

file_ff=gaussian_filter1d(file,sigma=10,axis=0)
file_f=pd.DataFrame(file_ff)
del(file_f[0])
file_f.insert(loc=0, column="time", value=timeT)

#diferencio cada segmento
file_dd=file_f.diff()
file_d=file_dd/0.033

#plot de cada segmento superpuesto
plt.figure(1)
plt.axhline(0, color='black')
for i in range(1,len(file_f.columns)):
    ax1 = plt.subplot(211)
    plt.plot(timeT, file_f[i])
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(timeT, file_d[i])
    plt.legend(loc=5, bbox_to_anchor=(1, 1))

#encontrar picos en el segmento medio de elongacion y contraccion
peaks_E_mid,_= find_peaks(file_d[6], distance=100, height=10)
peaks_C_mid,_= find_peaks(-file_d[6], distance=100, height=10)

#graficar para ver donde calzan los maximos
plt.figure(2)
plt.plot(file_d[5])
plt.plot(peaks_E_mid, file_d[5][peaks_E_mid], "x", color = 'r')
plt.plot(peaks_C_mid, file_d[5][peaks_C_mid], "x", color = 'b')
plt.show()

#comienzo y fin de cada ciclo
cycles_on=peaks_C_mid[0:len(peaks_C_mid)-1]
cycles_off=peaks_C_mid[1:len(peaks_C_mid)]

#recortar cada ciclo en un diccionario, cada linea es un ciclo y cada columna un segmento
file_d_cycles={}
for i in range(len(cycles_on)):
    data=file_d[cycles_on[i]:cycles_off[i]]
    file_d_cycles[i]=data    

#generar un eje x que vaya de 0 a 1
file_d_x={}
for i in range(len(cycles_on)):
    dx=1/(len(file_d_cycles[i])-1)
    data_xx=np.arange(0,1+dx,dx)
    if len(data_xx) == len(file_d_cycles[i]):
        data_x=data_xx
    else:
        data_x=data_xx[0:-1]
    file_d_x[i]=data_x

    
#plotear por segmento superponiendo ciclos
#segmentos anteriores
plt.figure()
plt.axhline(0, color='black')
for i in range(1,10):
    ax1 = plt.subplot(511)
    plt.plot(file_d_x[i], file_d_cycles[i][1])
    plt.title('segments 1-2-3-4-5')
    ax2 = plt.subplot(512)
    plt.plot(file_d_x[i], file_d_cycles[i][2])
    ax3 =plt.subplot(513)
    plt.plot(file_d_x[i], file_d_cycles[i][3])
    ax4 =plt.subplot(514)
    plt.plot(file_d_x[i], file_d_cycles[i][4])
    ax5 =plt.subplot(515)
    plt.plot(file_d_x[i], file_d_cycles[i][5])

#segmentos posteriores
plt.figure()
plt.axhline(0, color='black')
for i in range(1,10):
    ax1 = plt.subplot(511)
    plt.plot(file_d_x[i], file_d_cycles[i][6])
    plt.title('segments 6-7-8-9-10')
    ax2 = plt.subplot(512)
    plt.plot(file_d_x[i], file_d_cycles[i][7])
    ax3 =plt.subplot(513)
    plt.plot(file_d_x[i], file_d_cycles[i][8])
    ax4 =plt.subplot(514)
    plt.plot(file_d_x[i], file_d_cycles[i][9])
    ax5 =plt.subplot(515)
    plt.plot(file_d_x[i], file_d_cycles[i][10])

#segmentos medios
plt.figure()
plt.axhline(0, color='black')
for i in range(1,10):
    ax1 = plt.subplot(511)
    plt.plot(file_d_x[i], file_d_cycles[i][3])
    plt.title('segments 3-4-5-6-7')
    ax2 = plt.subplot(512)
    plt.plot(file_d_x[i], file_d_cycles[i][4])
    ax3 =plt.subplot(513)
    plt.plot(file_d_x[i], file_d_cycles[i][5])
    ax4 =plt.subplot(514)
    plt.plot(file_d_x[i], file_d_cycles[i][6])
    ax5 =plt.subplot(515)
    plt.plot(file_d_x[i], file_d_cycles[i][7])
