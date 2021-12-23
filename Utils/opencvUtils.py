import math
import glob
import json
import os
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
import scipy.signal as spsig
import pickle
from scipy.stats import zscore, median_abs_deviation

def myround(val):
    "Fix pythons round"
    d,v = math.modf(val)
    if d==0.5:
        val += 0.000000001
    return round(val)

def plot_ellipses(ax, weights, means, covars):
    for n in range(means.shape[0]):
        eig_vals, eig_vecs = np.linalg.eigh(covars[n])
        unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
        angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
        # Ellipse needs degrees
        angle = 180 * angle / np.pi
        # eigenvector normalization
        eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
        ell = mpl.patches.Ellipse(means[n], eig_vals[0], eig_vals[1],
                                  180 + angle, edgecolor='black')
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(weights[n])
        ell.set_facecolor('#56B4E9')
        ax.add_artist(ell)

def getBasenameFromNpy(npy_fn):
    npy_fn = os.path.basename(npy_fn)

    parts = re.split("[_.]+", npy_fn)[:2]
    return "_".join(parts)

def getLeechContour(hsv_thres):
    contours, hierarchy = cv2.findContours(hsv_thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    parents = np.unique(hierarchy[0, :, 3])
    num_children = [np.count_nonzero(hierarchy[0, :, 3] == pr) for pr in parents]
    ct_size = 0
    for pr, nc in zip(parents, num_children):
        if len(contours[pr]) > ct_size and nc > 6:
            leech_ct = contours[pr]
            ct_size = len(contours[pr])

    return leech_ct


def getOrientation(p0, num=5):
    ini = int(np.floor(num / 2))
    end = int(myround(num / 2))
    # end = int(np.round(num / 2))

    angle = []
    for i in range(p0.shape[0]):

        if i < ini:
            m, b = np.polyfit(p0[0:i + end, 0, 0], p0[0:i + end, 0, 1], 1)
        else:
            m, b = np.polyfit(p0[i - ini:i + end, 0, 0], p0[i - ini:i + end, 0, 1], 1)
        angle.append(np.arctan(m))
    return angle


markers = []


class MarkerGenerator():
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        print("Click derecho genera un nuevo punto.\nLa ruedita desconecta la figura")
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        global ix, iy
        if event.button.name == 'RIGHT':
            global markers
            ix, iy = np.round(event.xdata), np.round(event.ydata)
            markers.append((int(ix), int(iy)))

        if event.button.name == 'MIDDLE':
            self.fig.canvas.mpl_disconnect(self.cid)

def getPcaColumns(df):
    return df.loc[:,[col for col in df.columns if 'col' in col]]

def generateMetadata(globs):

    metadata = {}
    for key in list(globs.keys()):
        if key.startswith("_") or len(key)<2:
            pass
        elif type(globs[key])==str:
            metadata[key] = globs[key]
        else:
            try:
                iter(globs[key])
                pass
            except TypeError:
                try:
                    globs[key] + 0
                    metadata[key] = globs[key]
                except:
                    pass
            except ValueError:
                pass
    return metadata

def pickleDfArrayStore(filename, df, processed_matrix=None, globs=None):
    if not os.path.splitext(filename)[1]:
        filename += '.pkl'
    store = {}
    if globs is not None:
        metadata = generateMetadata(globs)
        store['metadata'] = metadata

    if processed_matrix is not None:
        store['matrix'] = processed_matrix
    store['df'] = df

    with open(filename, 'wb') as pfile:
        pickle.dump(store, pfile)


def pickleDfArrayLoad(filename):
    if not os.path.splitext(filename)[1]:
        filename += '.pkl'
    with open(filename, 'rb') as pfile:
        store = pickle.load(pfile)
    return store

def checkCyclesReset(binned_cycle_lengths, start_segment='mid',min_stretching_segs=3, min_shortening_segs=2):
    """
    Checks whether the cycle has some type of posterior to anterior
    reset of the body at the end of it.
    Parameters
    ---------
    binned_cycle_lengths ndarray:
        Array of shape (number of cycles, number of segments, number of bins) where each row has the information for each individual cycle
        where the second axis corresponds to each segment and the last axis is the time.
    start_segment int, str:
        Which segment will be the first to start looking for stretching. Those behind (the segments closer to the tail)
         it will be searched for stretching, while those ahead(the segments closer to the head) will be searched for
         shortening. if `mid` the middle segment will be selected.
    min_stretching_segs int:
        The minimum number of segments where stretching after proper movement must be found for the cycle to be
        considered a reset one.
    min_shortening_seg int:
        The minimum number of segments where shortening after proper movement must be found for the cycle to be
        considered a reset one.

    Returns
    ---------
    reseting cycles ndarray:
        Boolean array of shape (number of cycles,) where a 1 indicates the cycle was found to be a reseting one and
        a 0 indicates it wasn't.
    """
    num_cycles = binned_cycle_lengths.shape[0]

    reseting_cycles = np.zeros(num_cycles, dtype=bool)
    errored_rows = []
    i = 0
    for length_matrix in binned_cycle_lengths:
        err = False
        num_segments = length_matrix.shape[0]
        if start_segment == 'mid':
            start_segment = int(np.floor(num_segments/2))

        segment_speeds = np.diff(length_matrix, axis=1) #gets speeds

        speed_mad = median_abs_deviation(segment_speeds, axis=None)

        # finds times and segments that are shortening
        shortening_idxs = np.where(segment_speeds < 0)
        fast_shortening_idxs = np.where(segment_speeds < -2*speed_mad)

        # finds times and segments that are stretching
        stretching_idxs = np.where(segment_speeds > 0)
        over_std_streching_idxs = np.where(segment_speeds > speed_mad)
        over_std_shortening_idxs = np.where(segment_speeds < -speed_mad)

        after_stretch = np.zeros(num_segments-start_segment, dtype=bool)
        after_short = np.zeros(start_segment, dtype=bool)

        # last_shortening = shortening_idxs[1][shortening_idxs[0]==(num_segments-1)][-1]
        try:
            last_shortening = fast_shortening_idxs[1][fast_shortening_idxs[0]==(num_segments-1)][-1]
        except IndexError:
            reseting_cycles[i] = 0
            i += 1
            continue

        j = 0

        for seg in range(start_segment):
            try:
                # gets the times where anterior segments are shortening
                temp_idxs = over_std_shortening_idxs[1][over_std_shortening_idxs[0]==seg]
            except IndexError as e:
                errored_rows.append(i)
                i += 1
                err = True
                break
            if any([td > last_shortening for td in temp_idxs]): # find if the times ares after the last shortening
                after_short[j] = 1
            j += 1
        if err: continue
        j = 0
        for seg in range(start_segment, num_segments):
            try:
                #finds the first stretching for each segment
                first_strech = stretching_idxs[1][stretching_idxs[0] == seg][0]
                # finds the first shortening after the stretching
                first_short_after_strech = \
                    shortening_idxs[1][(shortening_idxs[1] > first_strech) & (shortening_idxs[0] == seg)][0]

                temp_idxs = over_std_streching_idxs[1][over_std_streching_idxs[0] == seg]
            except IndexError as e:
                errored_rows.append(i)
                err = True
                i += 1
                break
            if any([td > first_short_after_strech for td in temp_idxs]):
                #finds if theres stretching after the previuos shortening
                after_stretch[j] = 1
            j += 1
        if err: continue

        # the min stretching and shortening conditions are met, then the cycle y labeled with a 1
        if sum(after_stretch) >= min_stretching_segs and sum(after_short) >= min_shortening_segs:
            reseting_cycles[i] = 1
        i += 1
    return reseting_cycles, np.array(errored_rows, dtype=int)

def  plotBinnedLengths(cycle_length, num_segments=None, num_bins=None,zscore_speed=False):
    if num_segments is not None:
        cycle_length = cycle_length.reshape(num_segments, num_bins)
    speeds = np.diff(cycle_length, axis=1)
    if zscore_speed:
        speeds = zscore(speeds,axis=None)
        speeds[(speeds<.5) & (speeds>-.5)] = 0.
        speeds[(speeds<.5) & (speeds>-.5)] = 0.


    fig, ax = plt.subplots(cycle_length.shape[0], 2)
    for j in range(cycle_length.shape[0]):
        ax[j, 0].plot(cycle_length[j], c='k')
        if zscore_speed:
            ax[j, 1].imshow(speeds[j][np.newaxis, :], cmap="bwr", vmin=-1, vmax=1, aspect="auto")
        else:
            ax[j, 1].imshow(speeds[j][np.newaxis, :], cmap="bwr", vmin=-.15, vmax=.15, aspect="auto")

    return fig, ax


def plot3Dscatter(array3D, fig_ax_pair=None, label=None, colorbar=False, s=5, color=None, alpha=1):
    if fig_ax_pair is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    else:
        fig, ax = fig_ax_pair

    if label is not None:
        label = str(label)
    if color is None:
        N = 21
        cmap = plt.cm.jet
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        ax.scatter(array3D[:, 0], array3D[:, 1], array3D[:, 2],
                   c=plt.cm.jet(np.linspace(0, 1, array3D.shape[0])), label=label, s=s)
        if colorbar:
            plt.colorbar(sm, ticks=np.linspace(0, 1, N),
                         boundaries=np.arange(-0.05, 1.1, .05))
    else:
        ax.scatter(array3D[:, 0], array3D[:, 1], array3D[:, 2],
                   c=color, label=label, s=s, alpha=alpha)
    return fig, ax

def getSmoothingAndSpeedFromSegLen(segment_length, filter_length):
    smoothed_seg_len = []
    speed = []
    rmn = segment_length.shape[1] % filter_length

    if rmn == 0:
        pass
    else:
        segment_length = segment_length[:, :-rmn]

    bwr = plt.get_cmap('bwr')
    for i in range(segment_length.shape[0]):
        ssl = np.mean(segment_length[i].reshape(-1, filter_length), axis=1)

        speed.append(ssl[2:] - ssl[:-2])
        smoothed_seg_len.append(ssl[1:-1])

    speed = np.array(speed)
    smoothed_seg_len = np.array(smoothed_seg_len)
    return smoothed_seg_len, speed

def getLengthMins(total_length, fps, filter_length, min_distance=2):

    avg_len = np.mean(total_length)

    mins = spsig.find_peaks(-total_length, -avg_len, distance=int(min_distance * fps / filter_length))[0]
    maxs = spsig.find_peaks(total_length, avg_len, distance=int(min_distance * fps / filter_length))[0]

    normalization_length = np.median(total_length[maxs])

    ampl = np.mean(total_length[maxs]) - np.mean(total_length[mins])

    true_mins = spsig.find_peaks(-total_length, -(avg_len - (ampl / 8)), distance=int(min_distance * fps / filter_length))[0]
    return true_mins

def getCycleIndexes(total_length, cheking_speed, speed_threshold, fps, filter_length):

    mins = getLengthMins(total_length, fps, filter_length)
    cycle_idxs = []
    for min_idx in mins:
        try:
            val = np.where((cheking_speed > speed_threshold) & (np.arange(cheking_speed.shape[0]) >= min_idx))[0][0]
            cycle_idxs.append(val)
        except IndexError:
            break


    return np.array(cycle_idxs)


class TrackingError(Exception):
    '''Raise when a specific subset of values in context of app is wrong'''

    def __init__(self, message, failing_st, *args):
        self.message = message  # without this you may get DeprecationWarning
        # Special attribute you desire with your Error,
        # perhaps the value that caused the error?:
        self.failing_st = failing_st
        # allow users initialize misc. arguments as any other builtin Error
        super(TrackingError, self).__init__(message, failing_st, *args)


if __name__ == "__main__":
    with open('vids/marker_dict.json', 'r') as fp:
        video_params = json.load(fp)

    vid_path = "vids/"

    video_list = glob.glob(vid_path + "*.MOV")

    filename = list(video_params)[1]

    basename = os.path.splitext(filename)[0]

    cap = cv2.VideoCapture(vid_path + filename)

    coords = []

    ret, old_frame = cap.read()
    cap.release()
    fig, ax = plt.subplots()

    ax.imshow(old_frame)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)


