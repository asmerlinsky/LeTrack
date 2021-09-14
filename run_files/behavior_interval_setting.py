import json
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig
from scipy.stats import median_abs_deviation
from varname import nameof
import Utils.opencvUtils as cvU


plt.ioff()

if __name__ == '__main__':
    # TRACKING_COLOR = cv2.COLOR_BGR2GRAY
    TRACKING_COLOR = cv2.COLOR_BGR2HSV

    num_bins = 10
    seg_start = 0
    num_segments = 9
    filter_length = 3
    fps = 30
    save_fig = False
    show_plot = True

    print(f"{nameof(save_fig)} is set to {save_fig}")
    print(f"{nameof(show_plot)} is set to {show_plot}")

    data_path = "output/tracked_data/"

    with open('marker_dict.json', 'r') as fp:
        video_params = json.load(fp)

    npy_files = glob.glob('output1/tracked_data/*.npy')
    remove_files = glob.glob('output1/tracked_data/*_centered*.npy')
    npy_files = [fn.replace("\\", "/") for fn in npy_files if fn not in remove_files]

    binned_cycle_speeds = []
    leech_no = []
    for npy_file in npy_files:

        if 'FAILED' in npy_file:
            print("{} failed, continuing".format(npy_file))
            continue

        basename = cvU.getBasenameFromNpy(npy_file)
        avi_basename = basename + ".AVI"
        if avi_basename not in list(video_params) or not video_params[avi_basename]['analyze']:
            print("{} is not in video params, skipping".format(basename))
            continue

        print("Running %s" % basename)

        trace = np.load(npy_file)
        trace = trace[seg_start:seg_start+num_segments+1]
        if (trace.shape[0] < seg_start+num_segments):
            print("{} has not enough segments".format(basename))
            continue


        seg_len = np.sqrt(np.sum(np.power(np.diff(trace, axis=0), 2), axis=2))

        total_length = seg_len.sum(axis=0)

        smoothed_seg_len = []
        speed = []

        time_array = np.arange(0, trace.shape[1] / fps, 1 / fps)

        rmn = seg_len.shape[1] % filter_length
        if rmn == 0:
            pass
        else:
            seg_len = seg_len[:, :-rmn]

        bwr = plt.get_cmap('bwr')
        for i in range(seg_len.shape[0]):
            ssl = np.mean(seg_len[i].reshape(-1, filter_length), axis=1)

            speed.append(ssl[2:] - ssl[:-2])
            smoothed_seg_len.append(ssl[1:-1])

            resampled_seglen = smoothed_seg_len[-1] / smoothed_seg_len[-1].max()
            if i == 0:
                resampled_ta = time_array[int(np.floor(filter_length))::filter_length][:resampled_seglen.shape[0]]
                extent = [resampled_ta[0] - (resampled_ta[1] - resampled_ta[0]) / 2.,
                          resampled_ta[-1] + (resampled_ta[1] - resampled_ta[0]) / 2., 0, 1]

        speed = np.array(speed)
        speed_thres = median_abs_deviation(speed.flatten())

        smoothed_seg_len = np.array(smoothed_seg_len)

        smoothed_total_len = smoothed_seg_len.sum(axis=0)
        avg_len = np.mean(smoothed_total_len)

        mins = spsig.find_peaks(-smoothed_total_len, -avg_len, distance=int(4 * fps / filter_length))[0]

        maxs = spsig.find_peaks(smoothed_total_len, avg_len, distance=int(4 * fps / filter_length))[0]

        ampl = np.mean(smoothed_total_len[maxs]) - np.mean(smoothed_total_len[mins])

        mins = spsig.find_peaks(-smoothed_total_len, -(avg_len-(ampl/8)), distance=int(4 * fps / filter_length))[0]

        min_times = resampled_ta[mins]

        fig4, ax4 = plt.subplots()
        fig4.suptitle(basename)
        ax4.plot(resampled_ta, smoothed_total_len)
        ax4.scatter(resampled_ta[mins], smoothed_total_len[mins], c='r', marker='x')

        speed = np.array(speed)

        cycle_start_idxs = np.array(
            [np.where((speed[0] > speed_thres) & (np.arange(speed.shape[1]) >= min_idx))[0][0] for
             min_idx in mins[:-2]])

        cycle_start_times = resampled_ta[cycle_start_idxs]
        ax4.scatter(cycle_start_times, smoothed_total_len[cycle_start_idxs], c='g', marker='x')

        plt.show(block=True)
