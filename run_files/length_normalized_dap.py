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
    # TRACKING_COLOR = cv2.COLOR_BGR2GRAY
    TRACKING_COLOR = cv2.COLOR_BGR2HSV
    has_edges = True
    num_bins = 20
    seg_start = 0
    num_segments = 5
    if num_segments != 'all':
        mid_segment = np.floor(num_segments / 2).astype(int)
    filter_length = 3
    fps = 30
    NO_EDGES = True
    save_fig = False
    show_plot = True
    min_stretching_segs = 2
    min_shortening_segs = 1
    data_path = "output_latest/"

    with open('marker_dict.json', 'r') as fp:
        video_params = json.load(fp)

    npy_files = glob.glob(data_path + 'tracked_data/*.npy')

    npy_files = [fn.replace("\\", "/") for fn in npy_files]
    npy_files = np.sort(npy_files)
    binned_cycle_lengths = []

    leech_no = []
    vid_name = []
    cycle_time = []
    cycle_duration = []
    num_markers = []

    for npy_file in npy_files:

        if 'FAILED' in npy_file:
            print("{} failed, continuing".format(npy_file))
            continue

        basename = cvU.getBasenameFromNpy(npy_file)

        in_vp = False

        for vp in list(video_params):
            if basename in vp:
                ext_basename = vp
                in_vp = True
                break
        if not in_vp:
            print("{} is not in video params, skipping".format(basename))
            continue
        if not video_params[ext_basename]['analyze']:
            print("Analyze for {} is false, skipping".format(basename))
            continue

        print("Running %s" % basename)
        if NO_EDGES:
            trace = np.load(npy_file)
        else:
            trace = np.load(npy_file)[1:-1]

        seg_len = np.sqrt(np.sum(np.power(np.diff(trace, axis=0), 2), axis=2))
        if num_segments == 'all':
            seg_idxs = np.arange(seg_len.shape[0])
            mid_segment = np.floor(seg_idxs.shape[0] / 2).astype(int)
        elif NO_EDGES:
            seg_idxs = np.linspace(0, seg_len.shape[0] - 1, num=num_segments + 1, dtype=int, endpoint=True)
        else:
            seg_idxs = np.linspace(0, seg_len.shape[0] - 1, num=num_segments + 1, dtype=int, endpoint=True)

        empty_len = np.empty((seg_idxs.shape[0] - 1, seg_len.shape[1]))

        for i in range(seg_idxs.shape[0] - 1):
            empty_len[i] = seg_len[seg_idxs[i]:seg_idxs[i + 1]].sum(axis=0)

        seg_len = empty_len
        shortest_seg = np.argmin(np.diff(seg_idxs))
        #
        # trace = trace[seg_start:seg_start + num_segments + 1]
        # if (trace.shape[0] < seg_start + num_segments):
        #     print("{} has not enough segments".format(basename))
        #     continue
        #
        # seg_len = np.sqrt(np.sum(np.power(np.diff(trace, axis=0), 2), axis=2))

        total_length = seg_len.sum(axis=0)

        smoothed_seg_len = []
        speed = []

        time_array = np.arange(0, trace.shape[1] / fps, 1 / fps)

        smoothed_seg_len, speed = cvU.getSmoothingAndSpeedFromSegLen(seg_len, filter_length)

        speed_thres = median_abs_deviation(speed[shortest_seg])

        resampled_ta = time_array[int(np.floor(filter_length))::filter_length][:smoothed_seg_len[-1].shape[0]]
        extent = [resampled_ta[0] - (resampled_ta[1] - resampled_ta[0]) / 2.,
                  resampled_ta[-1] + (resampled_ta[1] - resampled_ta[0]) / 2., 0, 1]

        smoothed_total_len = smoothed_seg_len.sum(axis=0)

        mins = cvU.getLengthMins(smoothed_total_len, fps, filter_length, min_distance=2)

        mean_cycle = np.mean(np.diff(resampled_ta[mins]))
        median_cycle = np.median(np.diff(resampled_ta[mins]))

        larger_than_10 = (np.diff(resampled_ta[mins]) > 10).sum() / mins.shape[0]
        if larger_than_10 > .15:
            cd = np.diff(resampled_ta[mins])
            cd = cd[cd > 8]
            threshold = np.mean(cd) / 2
        else:
            threshold = np.max((mean_cycle / 2, 3))

        mins = np.delete(mins, np.array(np.where(np.diff(mins) < threshold * fps / filter_length)[0]) + 1)

        min_times = resampled_ta[mins]

        fig4, ax4 = plt.subplots(2, 1)
        fig4.suptitle("{}\n threshold={:.1f}, el {:.2f} son >10".format(basename, threshold, larger_than_10))
        ax4[0].plot(resampled_ta, smoothed_total_len)
        ax4[0].scatter(resampled_ta[mins], smoothed_total_len[mins], c='r', marker='x')

        speed = np.array(speed)

        cycle_idxs = []
        val = 0
        for min_idx in mins:
            try:
                # val = np.where((speed[1] > speed_thres) & (np.arange(speed.shape[1]) >= min_idx))[0][0]
                val = np.where((speed[0] > speed_thres) & (np.arange(speed.shape[1]) >= min_idx) & (
                            np.arange(speed.shape[1]) > val))[0][0]
                cycle_idxs.append(val)
            except IndexError:
                break

        cycle_idxs = np.array(cycle_idxs)
        print(cycle_idxs)
        cycle_times = resampled_ta[cycle_idxs]

        cycle_start_idxs = cycle_idxs[:-1]
        cycle_end_idxs = cycle_idxs[1:]

        if type(video_params[ext_basename]['analysis_interval']) == str:
            pass
        else:

            interval_times = np.array(video_params[ext_basename]['analysis_interval']).flatten()

            selection = np.searchsorted(interval_times, resampled_ta[cycle_idxs])
            selection = np.array((selection[:-1], selection[1:]))
            good_idxs = np.all(selection % 2, axis=0)
            cycle_start_idxs = cycle_start_idxs[good_idxs]
            cycle_end_idxs = cycle_end_idxs[good_idxs]

        cycle_start_times = resampled_ta[cycle_start_idxs]
        cycle_end_times = resampled_ta[cycle_end_idxs]
        ax4[0].scatter(cycle_start_times, smoothed_total_len[cycle_start_idxs], c='g', marker='x')
        ax4[0].scatter(cycle_end_times, smoothed_total_len[cycle_end_idxs], c='b', marker='x')

        err = False

        for i in range(cycle_start_idxs.shape[0]):
            idx_start = cycle_start_idxs[i]
            idx_end = cycle_end_idxs[i]
            if not err and (idx_end - idx_start) < 21:
                print("{} has a short cycle".format(basename))
                err = True

            length_matrix = resample(smoothed_seg_len[:, idx_start:idx_end], num=num_bins, axis=1)

            length_matrix -= length_matrix.min(axis=1).reshape(-1, 1)
            length_matrix /= length_matrix.max(axis=1).reshape(-1, 1)

            binned_cycle_lengths.append(length_matrix)

        cycle_time.extend(resampled_ta[cycle_start_idxs])
        cycle_duration.extend(resampled_ta[cycle_end_idxs] - resampled_ta[cycle_start_idxs])
        vid_name.extend([basename] * cycle_start_idxs.shape[0])
        leech_no = np.append(leech_no, np.repeat(int(video_params[ext_basename]['leech']), cycle_start_idxs.shape[0]))
        ax4[1].hist(resampled_ta[cycle_end_idxs] - resampled_ta[cycle_start_idxs], bins=20)

    # plt.close('all')

    binned_cycle_lengths = np.array(binned_cycle_lengths)

    df = pd.DataFrame(leech_no, columns=['leech_no'])

    df['video_name'] = vid_name
    df['cycle_time'] = cycle_time
    df['cycle_duration'] = cycle_duration

    reseting_cycles, errored_rows = cvU.checkCyclesReset(binned_cycle_lengths, min_stretching_segs=min_stretching_segs,
                                                         min_shortening_segs=min_shortening_segs)
    print(reseting_cycles.sum(), reseting_cycles.shape[0])
    # reshaped_bcs = binned_cycle_lengths.reshape(binned_cycle_lengths.shape[0], num_bins * num_segments)

    df['cycle_reset'] = reseting_cycles
    errors = np.zeros(reseting_cycles.shape[0])

    if errored_rows:
        errors[errored_rows] = 1
    df['errored_cycle'] = errors
    # plt.close('all')

    filename = "{}crawling_length_normalized_{}segments_{}bins_vidname_cycleno".format(data_path, num_segments,
                                                                                       num_bins)
    cvU.pickleDfArrayStore(filename, df, processed_matrix=binned_cycle_lengths, globs=globals())
