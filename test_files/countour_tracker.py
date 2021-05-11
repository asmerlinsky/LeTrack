import time
import sys
import time

sys.path.append(os.getcwd())
from varname import nameof
from Utils.opencvUtils import *

# TRACKING_COLOR = cv2.COLOR_BGR2GRAY
TRACKING_COLOR = cv2.COLOR_BGR2HSV

logf = open("contour_tracker_errors.log", "a")

if __name__ == "__main__":
    save_vid = False
    save_fig = False
    show_plot = True

    print(f"{nameof(save_vid)} is set to {save_vid}")
    print(f"{nameof(save_fig)} is set to {save_fig}")
    print(f"{nameof(show_plot)} is set to {show_plot}")

    vid_path = "../NeuroData/videos_pruebas_beh/"

    with open('marker_dict.json', 'r') as fp:
        video_params = json.load(fp)

    for filename in list(video_params)[60:]:

        print("Running")

        basename = os.path.splitext(filename)[0]
        cap = cv2.VideoCapture(vid_path + filename)

        # Take first frame
        ret, old_frame = cap.read()

        if not ret:
            print("Couldn't load %s, continuing with the next video" % basename)
            continue
        else:
            print("Processing file %s" % basename)

        if video_params[filename]['analyze'] == False:
            print("Skipping file %s, 'analyze' is %b" % basename)
            continue

        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=tuple(video_params[filename]['lk_params']['winSize']),
                         maxLevel=3,
                         criteria=tuple(video_params[filename]['lk_params']['criteria']))

        thres_params = {key: tuple(item) for key, item in video_params[filename]['thres_params'].items()}

        p0 = np.array(video_params[filename]['markers'], dtype=np.float32)
        color = np.random.randint(0, 255, (p0.shape[0], 3))

        p0 = p0.reshape((p0.shape[0], 1, p0.shape[1]))
        # Create some random colors

        old_tracked = cv2.cvtColor(old_frame, TRACKING_COLOR)
        if old_tracked.ndim == 3:  ## if hsv use only v
            old_tracked = old_tracked[:, :, 2]

        old_hsv = cv2.cvtColor(old_frame, cv2.COLOR_BGR2HSV)
        #
        # fig, ax = plt.subplots(3,1)
        # for i in range(old_hsv.shape[2]):
        #     ax[i].hist(old_hsv[:,:,i].flatten(), bins=256)

        # plt.figure()
        # plt.imshow(old_frame)

        # plt.figure()
        # plt.imshow(old_hsv[:,:,2])

        # levanto los marcadores
        old_hsv_thresholded = cv2.inRange(old_hsv, thres_params['low_thres'], thres_params['up_thres'])
        # levanto el contorno
        contour_hsv_thresholded = cv2.inRange(old_hsv, thres_params['contour_low_thres'],
                                              thres_params['contour_up_thres'])

        # fig, ax = plt.subplots(4, 1, sharex=True, sharey=True)
        # ax[0].imshow(old_hsv_thresholded)
        # ax[1].imshow(old_frame)
        # ax[2].imshow(old_hsv)
        # ax[3].imshow(contour_hsv_thresholded)

        contours, hierarchy = cv2.findContours(contour_hsv_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ct_size = 0
        for ct in contours:
            if ct.shape[0] > ct_size:
                ct_size = ct.shape[0]
                leech_ct = ct

        # ax[0].scatter(leech_ct[:,:,0], leech_ct[:, :, 1])
        # ax[1].scatter(leech_ct[:,:,0], leech_ct[:, :, 1])
        # ax[2].scatter(leech_ct[:,:,0], leech_ct[:, :, 1])

        # plt.scatter(p0[:, 0, 0], p0[:,0 , 1], marker='x', c='r')

        # Create a mask image for drawing purposes
        # mask = np.zeros_like(old_frame)
        # untracked_frames = []

        # cap = cv2.VideoCapture(vid_path + filename)
        cap_result = cv2.VideoWriter("vids/" + basename + "_out.avi",
                                     cv2.VideoWriter_fourcc(*'DIVX'),
                                     fps, size)
        try:
            n = 0
            start = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                tracking_frame = cv2.cvtColor(frame, TRACKING_COLOR)
                if tracking_frame.ndim == 3:  ##si da 3 fue hsv
                    hsv_thres = cv2.inRange(tracking_frame, thres_params['contour_low_thres'],
                                            thres_params['contour_up_thres'])
                    tracking_frame = tracking_frame[:, :, 2]
                else:
                    hsv_thres = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), thres_params['contour_low_thres'],
                                            thres_params['contour_up_thres'])

                leech_ct = getLeechContour(hsv_thres)

                ct_mask = np.zeros_like(tracking_frame)
                ct_mask = cv2.drawContours(ct_mask, [leech_ct], 0, 255, 1)

                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_tracked, tracking_frame, p0, None, **lk_params)
                if not all(st):
                    failing_st = np.where(~st)[0]
                    raise TrackingError("Some markers failed to be tracked", failing_st)

                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                try:
                    trace = np.concatenate((trace, p1), axis=1)
                except NameError:
                    trace = np.copy(p1)
                    pass

                # angles = getOrientation(p1, num=7)
                # mask = np.zeros_like(old_frame)
                # for i in np.linspace(0, p1.shape[0], 4, dtype=int, endpoint=False):
                #     x = np.mean((p1[i,0,0], p1[i+1,0,0])).astype(int)
                #     y = np.mean((p1[i,0,1], p1[i+1,0,1])).astype(int)
                #
                #     frame = cv2.line(frame, (x-500,y - int(500*np.sin(angles[i]))),(x + 500,y + int(500*np.sin(angles[i]))), (0, 250, 0), 2)

                angles = getOrientation(p1, num=7)

                mask = np.zeros_like(old_frame)
                corrected_p1 = np.zeros_like(p1)
                for i in range(p1.shape[0]):
                    mask = np.zeros_like(tracking_frame)
                    a, b = p1[i, 0, 0] - 1500 * np.sin(angles[i]), p1[i, 0, 1] + 1500 * np.cos(angles[i])
                    a, b = int(a), int(b)
                    c, d = p1[i, 0, 0] + 1500 * np.sin(angles[i]), p1[i, 0, 1] - 1500 * np.cos(angles[i])
                    c, d = int(c), int(d)
                    mask = cv2.line(mask, (a, b), (c, d), 255, 2)
                    intersect = np.argwhere((ct_mask & mask) > 0)[(0, -1), :]

                    corrected_p1[i, 0, 0] = np.mean(intersect[:, 1])
                    corrected_p1[i, 0, 1] = np.mean(intersect[:, 0])
                    frame = cv2.circle(frame, (corrected_p1[i, 0, 0], corrected_p1[i, 0, 1]), 5, (0, 0, 255), 2)

                try:
                    centered_trace = np.concatenate((centered_trace, corrected_p1), axis=1)
                except NameError:
                    centered_trace = np.copy(corrected_p1)

                frame = cv2.drawContours(frame, [leech_ct], 0, (250, 0, 0), 3)
                # draw the segments
                for i, (next, prev) in enumerate(zip(good_new[1:], good_new[:-1])):
                    a, b = prev.ravel()
                    c, d = next.ravel()
                    frame = cv2.line(frame, (a, b), (c, d), color[i].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                frame = cv2.circle(frame, (good_new[-1, 0], good_new[-1, 1]), 5, color[i].tolist(), -1)

                # cv2.imshow('frame',img)
                cap_result.write(frame)

                # k = cv2.waitKey(5) & 0xff
                # if k == 27:
                #     break

                # Now update the previous frame and previous points
                old_tracked = tracking_frame.copy()
                p0 = good_new.reshape(-1, 1, 2)
                # if n%10==0:
                #     print(f"untracked_frames size is {len(untracked_frames)*untracked_frames[-1].nbytes/(1024**2)}")
                n += 1

                if n % 100 == 0:
                    sys.stdout.write('\rframe no:%i of %i (%is/%is)' % (
                        n, total_frames, int(n / fps), int(total_frames / fps)) + ' ' * 20)
                    sys.stdout.flush()  # important

            print("Frames processed")
            end = time.time()
            print("Took %f s per frame" % ((end - start) / total_frames))
            # np.save('tracked_data/' + basename, trace)
            np.save('tracked_data/' + basename + "_centered", centered_trace)

        except Exception as e:
            logf.write(
                "Failed to process {0} in frame {1} (of {2} frames): {3}\n".format(basename, n, total_frames, str(e)))
            np.save('tracked_data/' + basename + "_centered_FAILED", centered_trace)
        finally:
            cv2.destroyAllWindows()
            cap_result.release()
            cap.release()
            del trace, centered_trace
            pass
    logf.close()
