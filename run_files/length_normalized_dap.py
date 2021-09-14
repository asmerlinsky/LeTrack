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
    num_bins = 10
    seg_start = 0
    num_segments = 8
    if num_segments != 'all':
        mid_segment = np.floor(num_segments/2).astype(int)
    filter_length = 3
    fps = 30
    NO_EDGES = False
    save_fig = False
    show_plot = True

    data_path = "output_edges/"

    with open('marker_dict.json', 'r') as fp:
        video_params = json.load(fp)

    npy_files = glob.glob(data_path+'tracked_data/*.npy')

    npy_files = [fn.replace("\\", "/") for fn in npy_files]
    npy_files = np.sort(npy_files)
    binned_cycle_lengths = []

    leech_no = []
    vid_name = []
    cycle_time = []
    num_markers = []

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
        if has_edges:
            trace = np.load(npy_file)[1:-1]
        else:
            trace = np.load(npy_file)

        seg_len = np.sqrt(np.sum(np.power(np.diff(trace, axis=0), 2), axis=2))
        if num_segments=='all':
            seg_idxs = np.arange(seg_len.shape[0])
            mid_segment = np.floor(seg_idxs.shape[0]/2).astype(int)
        elif NO_EDGES:
            seg_idxs = np.linspace(0,seg_len.shape[0]-1, num=num_segments+1, dtype=int, endpoint=True)
        else:
            seg_idxs = np.linspace(0,seg_len.shape[0]-1, num=num_segments+1, dtype=int, endpoint=True)

        empty_len = np.empty((seg_idxs.shape[0]-1,seg_len.shape[1]))

        for i in range(seg_idxs.shape[0]-1):
            empty_len[i] = seg_len[seg_idxs[i]:seg_idxs[i+1]].sum(axis=0)

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

        mins = cvU.getLengthMins(smoothed_total_len, fps, filter_length)

        min_times = resampled_ta[mins]

        fig4, ax4 = plt.subplots()
        fig4.suptitle(basename)
        ax4.plot(resampled_ta, smoothed_total_len)
        ax4.scatter(resampled_ta[mins], smoothed_total_len[mins], c='r', marker='x')

        speed = np.array(speed)

        cycle_idxs = []
        for min_idx in mins:
            try:
                val = np.where((speed[1] > speed_thres) & (np.arange(speed.shape[1]) >= min_idx))[0][0]
                cycle_idxs.append(val)
            except IndexError:
                break


        cycle_idxs = np.array(cycle_idxs)

        cycle_times = resampled_ta[cycle_idxs]
        cycle_start_idxs = cycle_idxs[:-1]
        cycle_end_idxs = cycle_idxs[1:]



        if type(video_params[avi_basename]['analysis_interval']) == str:
            pass
        else:

            interval_times = np.array(video_params[avi_basename]['analysis_interval']).flatten()

            selection = np.searchsorted(interval_times, resampled_ta[cycle_idxs])
            selection = np.array((selection[:-1], selection[1:]))
            good_idxs = np.all(selection % 2, axis=0)
            cycle_start_idxs = cycle_start_idxs[good_idxs]
            cycle_end_idxs = cycle_end_idxs[good_idxs]

        cycle_start_times = resampled_ta[cycle_start_idxs]
        cycle_end_times = resampled_ta[cycle_end_idxs]
        ax4.scatter(cycle_start_times, smoothed_total_len[cycle_start_idxs], c='g', marker='x')
        ax4.scatter(cycle_end_times, smoothed_total_len[cycle_end_idxs], c='b', marker='x')
        err=False
        for i in range(cycle_start_idxs.shape[0]):
            idx_start = cycle_start_idxs[i]
            idx_end = cycle_end_idxs[i]
            if not err and (idx_end-idx_start)<21:
                print("{} has a short cycle".format(basename))
                err = True

            length_matrix = resample(smoothed_seg_len[:, idx_start:idx_end], num=num_bins, axis=1)


            length_matrix /= length_matrix.max(axis=1).reshape(-1, 1)


            binned_cycle_lengths.append(length_matrix)


        cycle_time.extend(resampled_ta[cycle_start_idxs])
        vid_name.extend([basename]*cycle_start_idxs.shape[0])
        leech_no = np.append(leech_no, np.repeat(video_params[avi_basename]['leech'], cycle_start_idxs.shape[0]))

    # plt.close('all')

    binned_cycle_lengths = np.array(binned_cycle_lengths)


    reseting_cycles = cvU.checkCyclesReset(binned_cycle_lengths,min_stretching_segs=2)

    # reshaped_bcs = binned_cycle_lengths.reshape(binned_cycle_lengths.shape[0], num_bins * num_segments)

    df = pd.DataFrame(leech_no, columns=['leech_no'])

    df['video_name'] = vid_name
    df['cycle_time'] = cycle_time
    df['cycle_reset'] = reseting_cycles

    plt.close('all')

    filename = "{}crawling_length_normalized_{}segments_{}bins_vidname_cycleno".format(data_path, num_segments, num_bins)
    cvU.pickleDfArrayStore(filename, df, processed_matrix=binned_cycle_lengths, globs=globals())
    run_videos = ['21-04-26_28', '21-04-28_09', '21-04-26_08']
    part_df = df.iloc[:,:10]

    part_df['leech_no'] = df.leech_no
    part_df = part_df[[vn in run_videos for vn in df.video_name]]
    sns.pairplot(part_df, hue='leech_no', diag_kind='hist', diag_kws = {'bins': 30})
    loaded = cvU.pickleDfArrayLoad(filename)



    # for vn in run_videos:
    #     for idx in df.loc[(df.video_name==vn) & (df.cycle_no<5)].index:
    #
    #         cycle = pca.inverse_transform(tf_bcs[idx]).reshape(num_segments, num_bins)
    #
    #
    #         fig, ax = plt.subplots(cycle.shape[0])
    #         fig.suptitle("file {}\n cycle {}".format(vn, df.cycle_no[idx]))
    #
    #         for j in range(cycle.shape[0]):
    #             ax[j].imshow(cycle[j][np.newaxis, :], cmap="bwr", vmin=-1., vmax=1., aspect="auto")
    #

    mixture = GaussianMixture(n_components=4, n_init=10, max_iter=1000)
    pred = mixture.fit_predict(df.iloc[:,:n_comps].values)
    df['pred'] = pred
    nc = 8
    sns.pairplot(df, vars=['col_'+str(i) for i in range(nc)],
                 hue='leech_no', palette=sns.color_palette()[:np.unique(df.leech_no).shape[0]])
    sns.pairplot(df, vars=['col_'+str(i) for i in range(nc)],
                 hue='cycle_reset')
    sns.pairplot(df, vars=['col_'+str(i) for i in range(nc)],
                 hue='pred', palette=sns.color_palette()[:np.unique(df.pred).shape[0]])
    sns.pairplot(df, vars=['col_'+str(i) for i in range(nc)], diag_kind='hist', diag_kws = {'bins':50})


    two_clust_df = df.loc[[row in [0, 1] for row in df.pred]].copy()
    unique_leech_df = two_clust_df[two_clust_df.leech_no==2].copy()
    unique_leech_df.drop(columns='leech_no', inplace=True)
    sns.pairplot(unique_leech_df, vars=['col_'+str(i) for i in range(8)],
                     hue='pred', palette=sns.color_palette()[:np.unique(unique_leech_df.pred).shape[0]], diag_kind='hist')


    sns.pairplot(two_clust_df, vars=['col_'+str(i) for i in range(8)],
                     hue='pred', palette=sns.color_palette()[:np.unique(two_clust_df.pred).shape[0]], diag_kind='hist')

    for i in range(mixture.means_.shape[0]):

        cycle = pca.inverse_transform(mixture.means_[i]).reshape(num_segments, num_bins)
        # cycle = binned_cycle_speeds[i]
        # cycle = np.round(cycle)
        fig, ax = plt.subplots(cycle.shape[0])
        fig.suptitle("center {}".format(i))

        for j in range(cycle.shape[0]):
            ax[j].plot(cycle[j], c='k')


    mat = binned_lengths[152]
    cvU.plotBinnedLengths(mat)

    for row in df[df.video_name=='21-04-28_04'].index:
        mat = binned_cycle_lengths[row]
        cvU.plotBinnedLengths(mat)