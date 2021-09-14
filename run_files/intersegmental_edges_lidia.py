#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:03:39 2021

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

#with open('/Users/lidia/Dropbox/Data/Agustin/crawling video/marker_dict.json.mdlp', 'r') as fp:
#json_dict = json.load(fp)

#video=np. load('/Users/lidia/Dropbox/Data/Agustin/crawling video/tracked_data/21-04-26_22.npy',encoding = "latin1") #Load file.
#video_s=np. load('/Users/lidia/Dropbox/Data/Agustin/crawling video/tracked_data/21-04-26_22.npy',encoding = "latin1").shape #Load file.
#video[:,2]
#plt.plot(video[:,0], video[:,1])
#plt.show

#LEECH 00
f21_04_22_3 = pd.read_csv("/Users/lidia/Dropbox/Data/Agustin/crawling video/tracked_data_edges/21-04-22_3.csv", header=None) 
f_21_04_22_3= np.transpose(f21_04_22_3)
#establecer índice en que se completan 11 ciclos
I=2240 
     
#LEECH 01  - en un CICLO LA COLA SALE DEL CAMPO DE FILMACION. NO LO TOMO
#f21_04_26_00 = pd.read_csv("/Users/lidia/Dropbox/Data/Agustin/crawling video/tracked_data_edges/21-04-26_00.csv", header=None) 
#f_21_04_26_00= np.transpose(f21_04_26_00)

#LEECH 02 - MUY CORTO Y LUEGO SE DESORDENA!!!
#f21_04_26_05 = pd.read_csv("/Users/lidia/Dropbox/Data/Agustin/crawling video/tracked_data_edges/21-04-26_05.csv", header=None) 
#f_21_04_26_05= np.transpose(f21_04_26_05)

#LEECH 03 
f21_04_28_02 = pd.read_csv("/Users/lidia/Dropbox/Data/Agustin/crawling video/tracked_data_edges/21-04-28_02.csv", header=None) 
f_21_04_28_02= np.transpose(f21_04_28_02)

#LEECH 04 
f21_04_28_06 = pd.read_csv("/Users/lidia/Dropbox/Data/Agustin/crawling video/tracked_data_edges/21-04-28_06.csv", header=None) 
f_21_04_28_06= np.transpose(f21_04_28_06)

#LEECH 05 NO LO TOMO, MUY RUIDOS0
#f21_04_30_27 = pd.read_csv("/Users/lidia/Dropbox/Data/Agustin/crawling video/tracked_data_edges/21-04-30_27.csv", header=None) 
#f_21_04_30_27= np.transpose(f21_04_30_23)

#LEECH 06 
f21_04_30_04 = pd.read_csv("/Users/lidia/Dropbox/Data/Agustin/crawling video/tracked_data_edges/21-04-30_04.csv", header=None) 
f_21_04_30_04= np.transpose(f21_04_30_04)

############
############
############

#REEMPLAZAR PARA CADA video!!!!

#LEECH 00
#file=f_21_04_22_3
#name="leech0"
#filename="f_21_04_22_3"
#I=2200

#LEECH 03 
#file=f_21_04_28_02
#name="leech3"
#filename="f_21_04_28_02"
#I= 2500

#LEECH 04 
#file=f_21_04_28_06
#name="leech4"
#filename="f_21_04_28_06"
#I=1800

#LEECH 06 
file=f_21_04_30_04
name="leech6"
filename="f_21_04_30_04"
I=2175

#eje x en tiempo real
#time=list(range(1, len(file)+1))
#timeT=[i * 0.033 for i in time]

#EVALUAR CADA FILE PARA ASEGURARSE QUE LAS MEDICIONES OCURREN CORRECTAMENTE
#for i in range(L):   # plotear para encontrar rango en que los ciclos esten completos  
#     plt.plot(file[i])
#     #plt.plot(timeT, lengths[i])
#     plt.legend(loc=1)


#filtro y convierto en dataframe
file_ff=gaussian_filter1d(file,sigma=10,axis=0)
file_f=pd.DataFrame(file_ff)
L=len(file.columns)
 
#calculo de la longitud total momento a momento
total_length=file_f.sum(axis = 1)

# plotear para encontrar rango en que los ciclos se inicien desde un mínimo y determinar el indice I
for i in range(L):   
     plt.plot(file_f[i])
     #plt.plot(timeT, lengths[i])
     plt.legend(loc=1)

 
#si no empieza de un mínimo corregir para que solo queden oscilaciones completa eliminando las filas necesarias y reseteando
#N = 80
#file_f = file_f.tail(file_f.shape[0] -N)
#file_f = file_f.reset_index(drop=True)
#total_length=total_length.tail(total_length.shape[0] -N)
#total_length = total_length.reset_index(drop=True)

#reducir a las primeras 11 oscilaciones
Total_length=total_length.head(I)
File_f=file_f.head(I)
File_f.to_csv(filename) #exportar para cross-correl

#calcular el aumento de cada fragmento
lengths={} 
for i in range(1,L):     
   lengths[0]=File_f[0]
   lengths[i]=lengths[i-1]+File_f[i]
Lengths = pd.DataFrame.from_dict(lengths)

#calcular en cuanto aumenta RELATIVAMENTE con cada fragmento
rel_lengths={} 
for i in range(L):     
   rel_lengths[i]=Lengths[i]/Lengths[L-1]
Rel_length = pd.DataFrame.from_dict(rel_lengths)
Rel_Leng_mean=Rel_length.mean(axis=0)
#Rel_Leng_mean.to_csv(namerl) 

#ploteo de todos los fragmentos y del total
plt.figure(1)
ax1 = plt.subplot(411)
plt.plot(Total_length)
for i in range(L):     
     ax2 = plt.subplot(412, sharex=ax1)
     plt.plot(File_f[i])
     ax2 = plt.subplot(413, sharex=ax1)
     plt.plot(Lengths[i])
     ax3=plt.subplot(414, sharex=ax1)
     plt.plot(Rel_length[i])


#el pico de maxima longitud, a partir del cual comienza el acortamiento, para calcular el periodo de la conducta
peaks,_= find_peaks(Total_length, distance=150, height=300) #AJUSTAR LA ALTURA
Peaks = pd.DataFrame.from_dict(peaks)
Period=Peaks.diff()
Period_t=Period*0.033

#picos de los fragmentos
peaks_s={}
for i in range(L):
   peak_s,_= find_peaks(File_f[i], distance=150, height=20) #AJUSTAR DISTANCIA Y HEIGHT
   peaks_s[i]=peak_s
Peaks_s = pd.DataFrame.from_dict(peaks_s) 

#corroborar donde caen los picos
plt.figure(3)
for i in range(L):
     plt.plot(File_f[i])
     plt.legend(loc=1)   
     plt.plot(Peaks_s[i], File_f[i][Peaks_s[i]], "x", color = 'r')

#calcula delays entre picos, pero luego esto debera hacerse por cross correl   
delays={}
for i in range(1,L):
     delay= Peaks_s[i]-Peaks_s[i-1]
     delays[i]=delay
Delays = pd.DataFrame.from_dict(delays)
Delays_t=Delays*0.033
#plt.figure(4)
#for i in range(1,L-1):
#    plt.plot(Delays_t[i])
#    plt.legend(loc=1) 
Delays_t_mean=Delays_t.mean(axis=0)

sumary={}
sumary["rel leng"]=Rel_Leng_mean
sumary["period"]=Period_t 
sumary["delays"]=Delays_t_mean

#Sumary={}
Sumary[name]=sumary


