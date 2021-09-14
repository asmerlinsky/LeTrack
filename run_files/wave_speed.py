import json
import os
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import scipy.signal as spsig
from scipy.stats import median_abs_deviation
import Utils.opencvUtils as cvU
import seaborn as sns
import pandas as pd
from scipy.signal import resample
plt.rcParams.update({'figure.max_open_warning': 0})


if __name__ == '__main__':

    num_bins = 20
    seg_start = 0
    num_segments = 'all'
    if num_segments != 'all':
        mid_segment = np.floor(num_segments/2).astype(int)
    filter_length = 3
    fps = 30
    NO_EDGES = False
    save_fig = False
    show_plot = True

    data_path = "output_latest/tracked_data/"

    with open('marker_dict.json', 'r') as fp:
        video_params = json.load(fp)

    npy_files = glob.glob('output_latest/tracked_data/*.npy')

    lens = []
    # for npy_file in npy_files:
    npy_file = npy_files[18]
    trace = np.load(npy_file)
    seg_len = np.sqrt(np.sum(np.power(np.diff(trace, axis=0), 2), axis=2))
    cum_len = np.cumsum(seg_len, axis=0)
    smoothed_seg_len, speed = cvU.getSmoothingAndSpeedFromSegLen(seg_len, filter_length)
    total_length = smoothed_seg_len.sum(axis=0)
    cycle_idxs = cvU.getCycleIndexes(total_length, speed[1], speed_threshold=median_abs_deviation(speed, axis=None)/2, fps=30, filter_length=filter_length)


    # %%


    norm_len = smoothed_seg_len#/total_length

    fig, ax = plt.subplots(smoothed_seg_len.shape[0]-1, 1, sharex=True)
    for i in range(1, norm_len.shape[0]-1):
        ax[i-1].plot(norm_len[i], label=i)
        ax[i-1].grid()
        for j in cycle_idxs:
            ax[i-1].axvline(j, c='r')
    ax[-1].plot(total_length, label='tl')
    for j in cycle_idxs:
        ax[-1].axvline(j, c='r')
    ax[-1].grid()
    fig.legend()

    norm_cum_len = np.cumsum(norm_len,axis=0)
    fig, ax = plt.subplots()
    for elem in norm_cum_len:
        ax.plot(elem)


    cum_len = np.cumsum(seg_len, axis=0)
    fig, ax = plt.subplots()
    for elem in cum_len:
        ax.plot(elem)
