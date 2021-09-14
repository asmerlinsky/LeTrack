import json
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig
from scipy.stats import median_absolute_deviation
from varname import nameof
from Utils.opencvUtils import *

if __name__ == '__main__':
    # TRACKING_COLOR = cv2.COLOR_BGR2GRAY
    TRACKING_COLOR = cv2.COLOR_BGR2HSV

    filter_length = 3
    fps = 30
    save_fig = False
    show_plot = True

    print(f"{nameof(save_fig)} is set to {save_fig}")
    print(f"{nameof(show_plot)} is set to {show_plot}")

    data_path = "tracked_data/"

    with open('marker_dict.json', 'r') as fp:
        video_params = json.load(fp)

    npy_files = glob.glob('output/tracked_data/21-04-26*.npy')
    remove_files = glob.glob('output/tracked_data/21-04-26*_centered*.npy')
    npy_files = [fn for fn in npy_files if fn not in remove_files]
    npy_file = npy_files[-6]

    basename = getBasenameFromNpy(npy_file)

    print("Running %s" % basename)

    trace = np.load(npy_file)
    seg_len = np.sqrt(np.sum(np.power(np.diff(trace, axis=0), 2), axis=2))




    total_length = seg_len.sum(axis=0)


    smoothed_seg_len = []
    speed = []

    time_array = np.arange(0, trace.shape[1] / fps, 1 / fps)

    # mins = spsig.find_peaks(-total_length, np.mean(-total_length), distance=int(2 * fps))[0]

    # min_times = time_array[mins]

    up_thres = 4
    low_thres = -4

    fig1, ax1 = plt.subplots(seg_len.shape[0], 1, sharex=True)
    fig1.suptitle(basename)
    fig2, ax2 = plt.subplots(seg_len.shape[0], 1, sharex=True)
    fig2.suptitle(basename)
    fig3, ax3 = plt.subplots(seg_len.shape[0], 1, sharex=True)
    fig3.suptitle(basename)
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

        if show_plot:
            ax2[i].plot(resampled_ta, resampled_seglen, c='k')


        ax1[i].imshow((-1 * speed[-1])[np.newaxis, :], cmap="bwr", vmin=low_thres, vmax=up_thres, aspect="auto",
                      extent=extent)
        # ax3[i].imshow((-1 * rect_speed)[np.newaxis, :], cmap="bwr", vmin=-1., vmax=1., aspect="auto",
        #               extent=extent)

        ax1[i].grid()
        ax3[i].grid()



    speed = np.array(speed)
    up_thres = median_absolute_deviation(speed.flatten())
    low_thres = -median_absolute_deviation(speed.flatten())

    for i in range(speed.shape[0]):

        rect_speed = np.copy(speed[i])
        rect_speed[(rect_speed > low_thres) & (rect_speed < up_thres)] = 0.
        rect_speed = np.sign(rect_speed)

        if i == 0:
            first_rect_speed = np.copy(rect_speed)
        ax1[i].imshow((-1 * speed[i])[np.newaxis, :], cmap="bwr", vmin=low_thres, vmax=up_thres, aspect="auto",
                      extent=extent)
        ax3[i].imshow((-1 * rect_speed)[np.newaxis, :], cmap="bwr", vmin=-1., vmax=1., aspect="auto",
                      extent=extent)
        ax1[i].grid()
        ax3[i].grid()


    fig1.canvas.draw()
    fig2.canvas.draw()
    if True:
        fig1.savefig("figs/" + basename + "speed_t.png", transparent=True, dpi=600)
        fig2.savefig("figs/" + basename + "seglen_t.png", transparent=True, dpi=600)

    ax2[0].get_shared_x_axes().join(ax2[0], ax1[0])
    ax2[0].get_shared_x_axes().join(ax2[0], ax3[0])
    smoothed_seg_len = np.array(smoothed_seg_len)

    # print(smoothed_seg_len.shape)
    smoothed_total_len = smoothed_seg_len.sum(axis=0)
    mins = spsig.find_peaks(-smoothed_total_len, np.mean(-smoothed_total_len), distance=int(4 * fps / filter_length))[0]
    min_times = resampled_ta[mins]

    speed = np.array(speed)
    speed_sign = np.sign(speed)

    cycle_start_idxs = [np.where((first_rect_speed > 0) & (np.arange(first_rect_speed.shape[0]) >= min_idx))[0][0] for
                        min_idx in mins[:-1]]
    # cycle_start_idxs = [np.where((speed[0]>0) & (np.arange(speed.shape[1])>=min_idx))[0][0] for min_idx in mins[:-1]]
    cycle_start_times = resampled_ta[cycle_start_idxs]

    for ax in ax1:
        # for tm in min_times:
        for tm in cycle_start_times:
            ax.axvline(tm, c='g', lw=3)
    for ax in ax3:
        # for tm in min_times:
        for tm in cycle_start_times:
            ax.axvline(tm, c='g', lw=3)

    if show_plot:
        fig4, ax4 = plt.subplots()
        fig4.suptitle(basename)
        ax4.plot(resampled_ta, smoothed_total_len)
        ax4.scatter(resampled_ta[mins], smoothed_total_len[mins], c='r', marker='x')
        ax4.scatter(cycle_start_times, smoothed_total_len[cycle_start_idxs], c='g', marker='x')

        fig5, ax5 = plt.subplots()
        fig5.suptitle(basename)
        ax5.hist(speed.flatten(), bins=500)
        ax5.axvline(up_thres, color='k')
        ax5.axvline(low_thres, color='k')
        ax1[0].get_shared_x_axes().join(ax1[0], ax4)

        if save_fig: fig4.savefig("figs/" + basename + "_hist.png", transparent=True, dpi=600)

    diff_times = np.diff(min_times)
    print("cycle mean: {}, cycle stdev: {}".format(np.mean(diff_times), np.std(diff_times)))
    print(
        "speed mean: {}, speed median: {}, speed stdev: {}".format(np.mean(speed), np.median(speed), np.std(speed)))

    # plt.close(fig2)
    # plt.close(fig1)
    # plt.close(fig5)
    # ax1[0].set_xlim((0, 20))
    # n += 1
    # break

    # coloured_by_speed = []
    # speed_frames = []
