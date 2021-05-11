import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig
from varname import nameof

if __name__ == '__main__':
    # TRACKING_COLOR = cv2.COLOR_BGR2GRAY
    TRACKING_COLOR = cv2.COLOR_BGR2HSV

    fps = 30
    save_fig = False
    show_plot = True

    print(f"{nameof(save_fig)} is set to {save_fig}")
    print(f"{nameof(show_plot)} is set to {show_plot}")

    data_path = "tracked_data/"

    with open('marker_dict.json', 'r') as fp:
        marker_dict = json.load(fp)

    for fn in list(marker_dict):
        if not marker_dict[fn]['analyze']:
            continue
        basename = os.path.splitext(os.path.basename(fn))[0]
        filename = data_path + basename + '.npy'

        # os.system("vlc vids/%s_out.avi" % basename)

        print("Running %s" % filename)

        trace = np.load(filename)

        seg_len = np.sqrt(np.sum(np.power(np.diff(trace, axis=0), 2), axis=2))
        total_length = seg_len.sum(axis=0)

        fig0, ax0 = plt.subplots()
        ax0.plot(total_length)
        fig0.canvas.draw()
        fig0.suptitle(basename)
        if save_fig: fig0.savefig("figs/" + basename + "_totlen.png", transparent=True, dpi=600)

        smoothed_seg_len = []
        speed = []
        filter_length = 5
        time_array = np.arange(0, trace.shape[1] / fps, 1 / fps)

        mins = spsig.find_peaks(-total_length, -800, distance=int(2 * fps))[0]

        min_times = time_array[mins]
        up_thres = .05
        low_thres = -.05

        fig1, ax1 = plt.subplots(seg_len.shape[0], 1, sharex=True)
        fig1.suptitle(basename)
        fig2, ax2 = plt.subplots(seg_len.shape[0], 1, sharex=True)
        fig2.suptitle(basename)

        bwr = plt.get_cmap('bwr')
        for i in range(seg_len.shape[0]):
            ssl = spsig.convolve(seg_len[i], np.ones(filter_length) / filter_length, mode='valid')[::filter_length]
            speed.append(ssl[2:] - ssl[:-2])
            smoothed_seg_len.append(ssl[1:-1])

            resampled_seglen = smoothed_seg_len[-1] / smoothed_seg_len[-1].max()
            if i == 0:
                resampled_ta = time_array[::filter_length][:resampled_seglen.shape[0]]
                extent = [resampled_ta[0] - (resampled_ta[1] - resampled_ta[0]) / 2.,
                          resampled_ta[-1] + (resampled_ta[1] - resampled_ta[0]) / 2., 0, 1]

            if show_plot:
                ax2[i].plot(resampled_ta, resampled_seglen, c='k')

            # speed_max = np.max(np.abs(speed[-1]))
            # speed[-1] = speed[-1]

            # Renormalizado
            # ax[i].scatter(part_ta[speed[-1]>0], speed[-1][speed[-1]>0]/speed_max, c='b', s=1)
            # ax[i].scatter(part_ta[speed[-1]<0], speed[-1][speed[-1]<0]/speed_max, c='r', s=1)

            # Thresholded
            # ax[i].scatter(part_ta[speed[-1]>up_thres], speed[-1][speed[-1]>up_thres], c='b', s=1)
            # ax[i].scatter(part_ta[speed[-1]<low_thres], speed[-1][speed[-1]<low_thres], c='r', s=1)

            # Solo signo

            # ax[i].scatter(part_ta[speed[-1]>up_thres], [1] * part_ta[speed[-1]>up_thres].shape[0], c='b', s=1)
            # ax[i].scatter(part_ta[speed[-1]<low_thres], [-1] * part_ta[speed[-1]<low_thres].shape[0], c='r', s=1)

            ax1[i].imshow((-1 * speed[-1])[np.newaxis, :], cmap="bwr", vmin=-1.5, vmax=1.5, aspect="auto",
                          extent=extent)

            # [ax[i].axvline(x, c='b', lw=4) for x in part_ta[speed[-1]>up_thres]]
            # [ax[i].axvline(x, c='r', lw=4) for x in part_ta[speed[-1]<low_thres]]
            ax1[i].grid()

        fig1.canvas.draw()
        fig2.canvas.draw()
        if True:
            fig1.savefig("figs/" + basename + "speed_t.png", transparent=True, dpi=600)
            fig2.savefig("figs/" + basename + "seglen_t.png", transparent=True, dpi=600)

        smoothed_seg_len = np.array(smoothed_seg_len)

        smoothed_total_len = smoothed_seg_len.sum(axis=0)
        mins = spsig.find_peaks(-smoothed_total_len, -800, distance=int(2 * fps / filter_length))[0]
        min_times = resampled_ta[mins]
        for ax in ax1:
            for tm in min_times:
                ax.axvline(tm, c='k', lw=2)

        speed = np.array(speed)
        speed_sign = np.sign(speed)

        if show_plot:
            fig3, ax3 = plt.subplots()
            fig3.suptitle(basename)
            ax3.plot(resampled_ta, smoothed_total_len)
            ax3.scatter(resampled_ta[mins], smoothed_total_len[mins], c='r', marker='x')

            fig4, ax4 = plt.subplots()
            fig4.suptitle(basename)
            ax4.hist(speed.flatten(), bins=1000)
            ax4.axvline(up_thres, color='k')
            ax4.axvline(low_thres, color='k')
            if save_fig: fig4.savefig("figs/" + basename + "_hist.png", transparent=True, dpi=600)

        diff_times = np.diff(min_times)
        print(np.mean(diff_times), np.std(diff_times))
        # break

        # coloured_by_speed = []
        # speed_frames = []

        """
        for j in range(speed.shape[1]):
            # draw the segments
            frame = np.copy(untracked_frames[filter_length+1:-filter_length:filter_length][j])
            for i,(next,prev) in enumerate(zip(trace[:, filter_length+1:-filter_length:filter_length][1:, j],trace[:, filter_length+1:-filter_length:filter_length][:-1, j])):
                a,b = prev.ravel()
                c,d = next.ravel()
                if speed[i, j] < low_thres:
                    color = (0, 0, 255)
                elif speed[i, j]>up_thres:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
        
                frame = cv2.line(frame, (a,b),(c,d), color, 2)
                frame = cv2.circle(frame,(a,b),5, color,-1)
            img = np.copy(frame)
        
            if show_plot:
                cv2.imshow('frame',img)
        
        
            speed_frames.append(img)
        
            k = cv2.waitKey(30*filter_length) & 0xff
            if k == 27:
                break
        
        cv2.destroyAllWindows()
        
        result = cv2.VideoWriter("vids/" + basename + "_speedout.avi",
                                 cv2.VideoWriter_fourcc(*'DIVX'),
                                 fps/filter_length, size)
        for frame in speed_frames:
        
            result.write(frame)
        
        cv2.destroyAllWindows()
        result.release()
        
        """
