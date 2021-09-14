import gc
import glob
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from varname import nameof

save_vid = False
save_fig = False
show_plot = True
update_json = False
print(f"{nameof(save_vid)} is set to {save_vid}")
print(f"{nameof(save_fig)} is set to {save_fig}")
print(f"{nameof(show_plot)} is set to {show_plot}")
print(f"{nameof(update_json)} is set to {update_json}")

vid_path = "../NeuroData/videos_pruebas_beh/"

video_list = glob.glob(vid_path + "*.AVI")

filename = os.path.basename(video_list[30])

with open('vids/marker_dict.json', 'r') as fp:
    marker_dict = json.load(fp)

# filename = str(sys.argv[1])

print("Running")

basename = os.path.splitext(filename)[0]

cap = cv2.VideoCapture(vid_path + filename)
# cap = cv2.VideoCapture("vids/test.mpg")


# Take first frame and find corners in it
ret, old_frame = cap.read()

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = cap.get(cv2.CAP_PROP_FPS)
# params for ShiTomasi corner detection


# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(10, 10),
                 maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_hsv = cv2.cvtColor(old_frame, cv2.COLOR_BGR2HSV)

fig, ax = plt.subplots(3, 1)
for i in range(old_hsv.shape[2]):
    ax[i].hist(old_hsv[:, :, i].flatten(), bins=256)

plt.figure()
plt.imshow(old_frame)

plt.figure()
plt.imshow(old_hsv[:, :, 2])

contour_low_thres = (95, 0, 70)
contour_up_thres = (105, 255, 255)

contour_hsv_thresholded = cv2.inRange(old_hsv, contour_low_thres, contour_up_thres)
contour_hsv_thresholded = (~contour_hsv_thresholded.astype(bool)).astype(np.uint8) * 255
blobs = cv2.inRange(old_hsv[:, :, 2], 0, 100)
blobs = (~blobs.astype(bool)).astype(np.uint8) * 255
blobs *= contour_hsv_thresholded

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True)
ax[0].imshow(old_frame)
ax[1].imshow(old_hsv)
ax[2].imshow(contour_hsv_thresholded)

contours, hierarchy = cv2.findContours(contour_hsv_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

ct_size = 0
for ct in contours:
    if ct.shape[0] > ct_size:
        ct_size = ct.shape[0]
        leech_ct = ct

ax[0].scatter(leech_ct[:, :, 0], leech_ct[:, :, 1])
ax[1].scatter(leech_ct[:, :, 0], leech_ct[:, :, 1])
ax[2].scatter(leech_ct[:, :, 0], leech_ct[:, :, 1])

blob_contours, hierarchy = cv2.findContours(blobs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
fig2, ax2 = plt.subplots()
ax2.imshow(blobs)
good_blobs = []
i = 0
for ct in blob_contours:
    if ct.shape[0] > 10:
        ax2.scatter(ct[:, :, 0], ct[:, :, 1], s=5)
        i += 1

mu = cv2.moments(leech_ct)
cov_mat = np.array(((mu['mu20'], mu['mu11']),
                    (mu['mu11'], mu['mu02'])
                    )
                   )
ang = np.arctan(2 * mu['mu11'] / (mu['mu20'] - mu['mu02'])) / 2
xm, ym = mu['m10'] / mu['m00'], mu['m01'] / mu['m00']
a, b, c, d = xm - 800, ym - 800 * np.tan(ang), xm + 800, ym + 800 * np.tan(ang)
ax[0].plot((a, c), (b, d), c='r')

# Create a mask image for drawing purposes

# untracked_frames = []
n = 0

# cap = cv2.VideoCapture(vid_path + filename)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    mask = np.zeros_like(frame)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    contour_hsv_thresholded = cv2.inRange(hsv_frame, contour_low_thres, contour_up_thres)

    contour_hsv_thresholded = (~contour_hsv_thresholded.astype(bool)).astype(np.uint8) * 255

    blobs = cv2.inRange(hsv_frame[:, :, 2], 0, 70)
    blobs = (~blobs.astype(bool)).astype(np.uint8) * 255
    blobs *= contour_hsv_thresholded

    contours, hierarchy = cv2.findContours(contour_hsv_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ct_size = 0
    for ct in contours:
        if ct.shape[0] > ct_size:
            ct_size = ct.shape[0]
            leech_ct = ct
    mask = cv2.drawContours(mask, [leech_ct], 0, (250, 0, 0), 3)

    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break

    blob_contours, hierarchy = cv2.findContours(blobs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    good_blobs = []
    i = 0
    for ct in blob_contours:
        try:
            if ct.shape[0] > 10:
                mu = cv2.moments(ct)
                xm, ym = int(round(mu['m10'] / mu['m00'])), int(round(mu['m01'] / mu['m00']))
                dist = leech_ct - np.array((xm, ym))
                dist = np.sqrt(np.sum(np.power(dist, 2), axis=2))
                if np.min(dist) > 10 and cv2.pointPolygonTest(leech_ct, (xm, ym), measureDist=False) == 1:
                    good_blobs.append((xm, ym))
                    mask = cv2.circle(mask, (xm, ym), 5, (50, 50, 200), 2)
                    mask = cv2.drawContours(mask, [ct], 0, (0, 255, 0), 3)
                    mask = cv2.putText(mask, str(i), (xm, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 10)
                    i += 1
        except ZeroDivisionError:
            pass

    img = cv2.add(frame, mask)

    cv2.imshow('img', img)

    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
print("Frames processed")

if save_vid:
    result = cv2.VideoWriter("vids/" + basename + "_out.avi",
                             cv2.VideoWriter_fourcc(*'DIVX'),
                             fps, size)
    for j in range(len(untracked_frames)):

        frame = np.copy(untracked_frames[j])

        for i, (next, prev) in enumerate(zip(good_new[1:], good_new[:-1])):
            a, b = prev.ravel()
        c, d = next.ravel()
        frame = cv2.line(frame, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        result.write(frame)

    cv2.destroyAllWindows()
    result.release()
    print("Video saved")

# plt.close('all')
del tracked_frames
gc.collect()
