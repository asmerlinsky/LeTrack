import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

filename = 'test.mpg'

cap = cv2.VideoCapture(filename)

# Take first frame and find corners in it
ret, old_frame = cap.read()

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = cap.get(cv2.CAP_PROP_FPS)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(10, 10),
                 maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

plt.imshow(old_frame)

points = (
    (511, 414),
    (564, 422),
    (603, 428),
    (657, 441),
    (716, 453),
    (771, 458),
    (807, 474),
    (852, 499),
    (898, 510),
    (957, 549),
    (1025, 558),
    (1068, 574),
    (1128, 588)
)

p0 = np.array(points, dtype=np.float32)

p0 = p0.reshape((p0.shape[0], 1, p0.shape[1]))

plt.scatter(p0[:, 0, 0], p0[:, 0, 1], marker='x', c='r')

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
tracked_frames = []
untracked_frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    untracked_frames.append(np.copy(frame))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if not all(st):
        break

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    try:
        trace = np.concatenate((trace, p1), axis=1)
    except NameError:
        trace = np.copy(p1)

    # draw the tracks
    # for i,(new,old) in enumerate(zip(good_new,good_old)):
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #     mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    # img = cv2.add(frame,mask)
    # cv2.imshow('frame',img)
    #

    # draw the segments
    for i, (next, prev) in enumerate(zip(good_new[1:], good_new[:-1])):
        a, b = prev.ravel()
        c, d = next.ravel()
        frame = cv2.line(frame, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = frame
    cv2.imshow('frame', img)

    tracked_frames.append(img)

    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()

# Guarda los frames con el trackeo en un archivo .avi
result = cv2.VideoWriter(os.path.splitext(filename)[0] + "_out.avi",
                         cv2.VideoWriter_fourcc(*'DIVX'),
                         fps, size)
for frame in tracked_frames:
    result.write(frame)

cv2.destroyAllWindows()
result.release()
