import numpy as np
from varname import nameof

from Utils.opencvUtils import *

# TRACKING_COLOR = cv2.COLOR_BGR2GRAY
TRACKING_COLOR = cv2.COLOR_BGR2HSV

if __name__ == "__main__":
    save_vid = True
    save_fig = True
    show_plot = False
    failed = False
    base_path = 'output_edges/'
    data_path = 'tracked_data_latest/'
    video_path = 'videos_latest/'
    # data_path = 'tracked_data/'
    # video_path = 'videos/'
    print(f"{nameof(save_vid)} is set to {save_vid}")
    print(f"{nameof(save_fig)} is set to {save_fig}")
    print(f"{nameof(show_plot)} is set to {show_plot}")

    vid_path = "../NeuroData/videos_pruebas_beh/"

    with open('marker_dict.json', 'r') as fp:
        video_params = json.load(fp)

    for filename in list(video_params):

        print("Running")

        if not video_params[filename]['analyze']:
            continue

        basename = os.path.splitext(filename)[0]
        cap = cv2.VideoCapture(vid_path + filename)

        # Take first frame
        ret, old_frame = cap.read()

        if not ret:
            print("Couldn't load %s, continuing with the next video" % basename)
            continue

        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = cap.get(cv2.CAP_PROP_FPS)

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
        cap_result = cv2.VideoWriter("{}{}{}_out.avi".format(base_path, video_path, basename ),
                                     cv2.VideoWriter_fourcc(*'DIVX'),
                                     fps, size)

        n = 0
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

            contours, hierarchy = cv2.findContours(hsv_thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ct_size = 0
            for ct in contours:
                if ct.shape[0] > ct_size:
                    ct_size = ct.shape[0]
                    leech_ct = ct

            ct_mask = np.zeros_like(tracking_frame)
            ct_mask = cv2.drawContours(ct_mask, [leech_ct], 0, 255, 1)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_tracked, tracking_frame, p0, None, **lk_params)
            if not all(st):
                break

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]


            angles = getOrientation(p1, num=5)


            mask = np.zeros_like(old_frame)
            corrected_p1 = np.zeros_like(p1)
            start_end = []
            try:
                for i in (0, -1):
                    mask = np.zeros_like(tracking_frame)
                    a, b = p1[i, 0, 0] + 1500 * np.cos(angles[i]), p1[i, 0, 1] + 1500 * np.sin(angles[i])
                    # a, b = p1[i, 0, 0] - 1500 * np.sin(angles[i]), p1[i, 0, 1] + 1500 * np.cos(angles[i])
                    a, b = int(a), int(b)
                    c, d = p1[i, 0, 0] - 1500 * np.cos(angles[i]), p1[i, 0, 1] - 1500 * np.sin(angles[i])
                    # c, d = p1[i, 0, 0] + 1500 * np.sin(angles[i]), p1[i, 0, 1] - 1500 * np.cos(angles[i])
                    c, d = int(c), int(d)
                    mask = cv2.line(mask, (a, b), (c, d), 255, 2)
                    intersect = np.argwhere((ct_mask & mask) > 0)[(0, -1), :][:,::-1]

                    closest_intersect = np.argmin(np.sqrt(np.power(intersect-p1[i],2).sum(axis=1)))
                    true_intersect = intersect[closest_intersect].astype(int)

                    frame = cv2.circle(frame, tuple(true_intersect), 5, (0, 0, 255), 2)
                    frame = cv2.line(frame, (a, b), (c, d), 255, 2)
                    start_end.append(true_intersect)
            except IndexError as ie:
                print("{} failed with error:\n{}".format(filename, ie))
                failed = True
                break

            p1 = np.vstack((start_end[0][np.newaxis,np.newaxis], p1))
            p1 = np.vstack((p1, start_end[-1][np.newaxis, np.newaxis]))

            try:
                trace = np.concatenate((trace, p1), axis=1)
            except NameError:
                trace = np.copy(p1)

            frame = cv2.drawContours(frame, [leech_ct], 0, (250, 0, 0), 3)
            # draw the segments
            for i, (next, prev) in enumerate(zip(good_new[1:], good_new[:-1])):
                a, b = prev.ravel().astype(int)
                c, d = next.ravel().astype(int)
                frame = cv2.line(frame, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            frame = cv2.circle(frame, (int(good_new[-1, 0]), int(good_new[-1, 1])), 5, color[i].tolist(), -1)

            # cv2.imshow('frame',img)
            cap_result.write(frame)

            # Now update the previous frame and previous points
            old_tracked = tracking_frame.copy()
            p0 = good_new.reshape(-1, 1, 2)

            n += 1

        cv2.destroyAllWindows()
        cap_result.release()
        cap.release()
        print("Frames processed")
        if not failed:
            np.save(base_path + data_path + basename, trace)
        else:
            os.rename(base_path + video_path + basename + "_out.avi",
                      base_path + video_path + basename + "_outCTFAILED.avi")
            failed = False

        del trace

