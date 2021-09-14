import json
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import scipy.signal as spsig
from scipy.stats import median_abs_deviation
from varname import nameof
import Utils.opencvUtils as cvU
import seaborn as sns
import pandas as pd


if __name__ == '__main__':
    # TRACKING_COLOR = cv2.COLOR_BGR2GRAY
    TRACKING_COLOR = cv2.COLOR_BGR2HSV

    num_bins = 6
    seg_start = 0
    num_segments = 9
    filter_length = 3
    fps = 30
    save_fig = False
    show_plot = True

    print(f"{nameof(save_fig)} is set to {save_fig}")
    print(f"{nameof(show_plot)} is set to {show_plot}")

    with open('marker_dict.json', 'r') as fp:
        video_params = json.load(fp)

    npy_files = glob.glob('output_latest/tracked_data/*.npy')

    npy_files = [fn.replace("\\", "/") for fn in npy_files]

    binned_cycle_speeds = []
    leech_no = []
    vid_name = []
    cycle_no = []
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
        trace = trace[seg_start:seg_start + num_segments + 1]
        if (trace.shape[0] < seg_start + num_segments):
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

        mins = spsig.find_peaks(-smoothed_total_len, -(avg_len - (ampl / 8)), distance=int(4 * fps / filter_length))[0]

        min_times = resampled_ta[mins]

        fig4, ax4 = plt.subplots()
        fig4.suptitle(basename)
        ax4.plot(resampled_ta, smoothed_total_len)
        # ax4.scatter(resampled_ta[mins], smoothed_total_len[mins], c='r', marker='x')

        speed = np.array(speed)

        cycle_idxs = []
        for min_idx in mins:
            try:
                val = np.where((speed[0] > speed_thres) & (np.arange(speed.shape[1]) >= min_idx))[0][0]
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

        for i in range(cycle_start_idxs.shape[0]):
            idx_start = cycle_start_idxs[i]
            idx_end = cycle_end_idxs[i]
            bins = np.linspace(cycle_start_times[i], cycle_end_times[i], num_bins, endpoint=False)
            digitized = np.digitize(resampled_ta[idx_start:idx_end], bins)

            speed_matrix = np.zeros((speed.shape[0], num_bins))
            for j in range(num_bins):
                speed_matrix[:, j] = speed[:, idx_start:idx_end][:, digitized == j + 1].mean(axis=1)

            pct = np.percentile(speed_matrix, [5, 95])
            speed_matrix[speed_matrix < 0] = speed_matrix[speed_matrix < 0] / np.abs(pct[0])
            speed_matrix[speed_matrix > 0] = speed_matrix[speed_matrix > 0] / np.abs(pct[1])

            binned_cycle_speeds.append(speed_matrix)
        cycle_no.extend(list(range(cycle_start_idxs.shape[0])))
        vid_name.extend([basename]*cycle_start_idxs.shape[0])
        leech_no = np.append(leech_no, np.repeat(video_params[avi_basename]['leech'], cycle_start_idxs.shape[0]))

    plt.close('all')

    binned_cycle_speeds = np.array(binned_cycle_speeds)


    reshaped_bcs = binned_cycle_speeds.reshape(binned_cycle_speeds.shape[0], num_bins * num_segments)
    n_comps = 15
    pca = PCA(n_components=n_comps)
    tf_bcs = pca.fit_transform(reshaped_bcs)

    fig, ax = plt.subplots()
    ax.scatter(np.arange(n_comps), pca.explained_variance_ratio_)


    columns = ['col_{}'.format(i) for i in range(n_comps)]
    df = pd.DataFrame(tf_bcs, columns=columns)
    df['leech_no'] = leech_no
    df['video_name'] = vid_name
    df['cycle_no'] = cycle_no


    part_df = df.iloc[:,:5]
    part_df['leech_no'] = df.leech_no

    fig, ax = plt.subplots()
    sns.pairplot(part_df, hue='leech_no', diag_kind='hist')

    #
    ### CADA AUTOVECTOR DEL PCA
    for i in range(5):
        cycle = pca.components_[i].reshape(num_segments, num_bins)

        fig, ax = plt.subplots(cycle.shape[0])
        fig.suptitle("component {}".format(i))

        for j in range(cycle.shape[0]):
            ax[j].imshow(cycle[j][np.newaxis, :], cmap="bwr", vmin=-.5, vmax=.5, aspect="auto")


    ### POR CLUSTER
    for i in range(mixture.means_.shape[0]):

        cycle = pca.inverse_transform(mixture.means_[i]).reshape(num_segments, num_bins)

        # arg = np.argmin(np.power(tf_bcs - mixture.means_[i], 2).sum(axis=1))
        # cycle = binned_cycle_speeds[arg]

        fig, ax = plt.subplots(cycle.shape[0])
        fig.suptitle("center {}".format(i))

        for j in range(cycle.shape[0]):
            ax[j].imshow(cycle[j][np.newaxis, :], cmap="bwr", vmin=-1., vmax=1., aspect="auto",
                         extent=extent)


    ### POR LEECH
    tip_cycle = df.groupby('pred').mean()
    for i in range(tip_cycle.shape[0]):

            cycle = pca.inverse_transform(tip_cycle.values[i]).reshape(num_segments, num_bins)
            leech = tip_cycle.index[i]
            # arg = np.argmin(np.power(tf_bcs - mixture.means_[i], 2).sum(axis=1))
            # cycle = binned_cycle_speeds[arg]

            fig, ax = plt.subplots(cycle.shape[0])
            fig.suptitle("leech {}".format(leech))

            for j in range(cycle.shape[0]):
                ax[j].imshow(cycle[j][np.newaxis, :], cmap="bwr", vmin=-1., vmax=1., aspect="auto",
                             extent=extent)



    # store = pd.HDFStore("output_latest/crawling_pca.h5")
    # store.put("Dataframe", df)

    cvU.pickleDfArrayStore("output_latest/crawling_pca_20comp_vidname_cycleno.h5", df, None, globs=globals())
    loaded_df, loaded_metadata = cvU.h5load("output_latest/crawling_pca.h5")


    #
    # perp = 50
    # tsne = TSNE(perplexity=perp, early_exaggeration=6., learning_rate=200, n_iter=6000)
    # tsne_data = tsne.fit_transform(tf_bcs)
    # print(tsne.n_iter_)
    #
    # tsne_gmm = GaussianMixture(n_components=3, max_iter=1000, n_init=100)
    # tsne_pred = tsne_gmm.fit_predict(tsne_data)
    #
    # fig, ax = plt.subplots()
    # fig.suptitle("perplexity={}".format(perp))
    # ax.scatter(tsne_data[:,0], tsne_data[:,1], c=tsne_pred)
    # df['pred'] = tsne_pred
    # sns.pairplot(df, hue='pred', diag_kind='hist')
    #
    #
    # for i in range(tsne_gmm.means_.shape[0]):
    #
    #     arg = np.argmin(np.power(tsne_data - tsne_gmm.means_[i], 2).sum(axis=1))
    #     cycle = binned_cycle_speeds[arg]
    #
    #     fig, ax = plt.subplots(cycle.shape[0])
    #     fig.suptitle("center {}".format(i))
    #
    #     for j in range(cycle.shape[0]):
    #         ax[j].imshow(cycle[j][np.newaxis, :], cmap="bwr", vmin=-1., vmax=1., aspect="auto",
    #                      extent=extent)
