import glob
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


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
    ini = np.floor(num / 2).astype(int)
    end = np.round(num / 2).astype(int)
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
