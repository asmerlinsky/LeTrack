import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig

save_vid = False
plot_mode = 2
# cap = cv2.VideoCapture('CIMG4777.MOV')
# cap = cv2.VideoCapture('cut_CIMG4777.mkv')
cap = cv2.VideoCapture('test.mpg')

# Take first frame and find corners in it
ret, old_frame = cap.read()

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fps = cap.get(cv2.CAP_PROP_FPS)
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.1,
                      minDistance=100,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
p0r = p0.reshape((p0.shape[0], p0.shape[2]))

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
while (ret):
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

if save_vid:
    result = cv2.VideoWriter('filename.avi',
                             cv2.VideoWriter_fourcc(*'DIVX'),
                             fps, size)
    for frame in tracked_frames:
        result.write(frame)

    cv2.destroyAllWindows()
    result.release()

plt.figure()
for i in range(trace.shape[0]):
    plt.plot(trace[i, :int(trace.shape[1] / 2), 0], -trace[i, :int(trace.shape[1] / 2), 1])

seg_len = np.sqrt(np.sum(np.power(np.diff(trace, axis=0), 2), axis=2))

smoothed_seg_len = []
speed = []
filter_length = 3
time_array = np.arange(0, trace.shape[1] / fps, 1 / fps)
up_thres = .61
low_thres = -.12
if plot_mode == 1:
    fig, ax = plt.subplots(2, 1)
    for i in range(seg_len.shape[0]):
        ssl = spsig.convolve(seg_len[i], np.ones(filter_length) / filter_length, mode='valid')

        speed.append(np.diff(ssl))
        smoothed_seg_len.append(spsig.convolve(ssl, (1, 1), mode='valid'))

        ax[0].plot(time_array[:smoothed_seg_len[-1].shape[0]], smoothed_seg_len[-1], label=str(i))
        ax[1].plot(time_array[:speed[-1].shape[0]], speed[-1])
    fig.legend()
elif plot_mode == 2:
    fig, ax = plt.subplots(seg_len.shape[0], 1)
    for i in range(seg_len.shape[0]):
        ssl = spsig.convolve(seg_len[i], np.ones(filter_length) / filter_length, mode='valid')

        speed.append(np.diff(ssl))
        smoothed_seg_len.append(spsig.convolve(ssl, (1, 1), mode='valid'))

        rescaled_seglen = smoothed_seg_len[-1] / smoothed_seg_len[-1].max()
        part_ta = time_array[:rescaled_seglen.shape[0]]

        ax[i].plot(part_ta, rescaled_seglen, c='k')

        # speed_max = np.max(np.abs(speed[-1]))
        speed[-1] = speed[-1]

        # ax[i].scatter(part_ta[speed[-1]>0], speed[-1][speed[-1]>0]/speed_max, c='b', s=1)
        # ax[i].scatter(part_ta[speed[-1]<0], speed[-1][speed[-1]<0]/speed_max, c='r', s=1)
        ax[i].scatter(part_ta[speed[-1] > up_thres], speed[-1][speed[-1] > up_thres], c='b', s=1)
        ax[i].scatter(part_ta[speed[-1] < low_thres], speed[-1][speed[-1] < low_thres], c='r', s=1)

        ax[i].grid()
    fig.legend()

smoothed_seg_len = np.array(smoothed_seg_len)
speed = np.array(speed)
speed_sign = np.sign(speed)

plt.figure()
plt.hist(speed.flatten(), bins=400)
plt.axvline(up_thres, color='k')
plt.axvline(low_thres, color='k')

shape_diff = trace.shape[1] - smoothed_seg_len.shape[1]

coloured_by_speed = []
speed_frames = []

speed_thres = .5
for j in range(trace.shape[1] - shape_diff - 1):
    # draw the segments
    frame = np.copy(untracked_frames[j + int(shape_diff / 2)])
    for i, (next, prev) in enumerate(zip(trace[1:, j + int(shape_diff / 2)], trace[:-1, j + int(shape_diff / 2)])):
        a, b = prev.ravel()
        c, d = next.ravel()
        if speed[i, j] < low_thres:
            color = (0, 0, 255)
            cv2.putText(frame, "THIS SHOULD HAVE RED", (300, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=1, thickness=5)
        elif speed[i, j] > up_thres:
            color = (0, 255, 0)
            cv2.putText(frame, "THIS SHOULD HAVE GREEN", (900, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=1, thickness=5)
        else:
            color = (255, 0, 0)
            cv2.putText(frame, "THIS SHOULD HAVE BLUE", (1500, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=1, thickness=5)

        frame = cv2.line(frame, (a, b), (c, d), color, 2)
        frame = cv2.circle(frame, (a, b), 5, color, -1)
    img = np.copy(frame)
    cv2.imshow('frame', img)

    speed_frames.append(img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()

result = cv2.VideoWriter('filename.avi',
                         cv2.VideoWriter_fourcc(*'DIVX'),
                         fps, size)
for frame in speed_frames:
    result.write(frame)

cv2.destroyAllWindows()
result.release()
