import sys

from Utils.opencvUtils import *

TRACKING_COLOR = cv2.COLOR_BGR2HSV
gen_markers = False
preprocessing = True
frame_processing = True
update_dict = False
update_json = False

"""
#para renombrar
vid_path = "../NeuroData/videos_pruebas_beh/"
    # vid_path = "vids/"

video_list = sorted(glob.glob(vid_path + "DSCN*.AVI"))
for i in range(len(video_list)):
    str_i = str(i)
    if len(str_i)==1:
        str_i = "0" +str_i

    os.rename(video_list[i], vid_path + "21-04-30_"+str_i+".AVI")

"""


def updateJson():
    with open("marker_dict.json", 'w') as f:
        json.dump(video_params, f)
        print("Updated json")


if __name__ == "__main__":

    with open('marker_dict.json', 'r') as fp:
        video_params = json.load(fp)

    vid_path = "../NeuroData/videos_pruebas_beh/"
    # vid_path = "vids/"

    # video_list = sorted(glob.glob(vid_path + "21-04-26_26.AVI"))

    filename = os.path.basename('DSC_8274.MOV')
    print(filename)
    if any([filename == fn for fn in list(video_params)]):
        print("El archivo ya esta subido a 'video_params'")

    # filename = list(video_params)[7]
    try:
        fn_dict = video_params[filename]
        try:
            print(fn_dict['comments'])
        except KeyError:
            print("No hay comentarios")
    except KeyError:
        print("No hay info del video")
        pass

    print("Running")

    # parametro para buscar los markers

    basename = os.path.splitext(filename)[0]

    cap = cv2.VideoCapture(vid_path + filename)
    # cap = cv2.VideoCapture("vids/test.mpg")

    # Take first frame and find corners in it
    ret, old_frame = cap.read()

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # params for ShiTomasi corner detection

    # Create some random colors
    color = np.random.randint(50, 255, (100, 3))

    old_tracked = cv2.cvtColor(old_frame, TRACKING_COLOR)
    if old_tracked.ndim == 3:  ## if hsv use only v
        old_tracked = old_tracked[:, :, 2]

    old_hsv = cv2.cvtColor(old_frame, cv2.COLOR_BGR2HSV)

    fig, ax = plt.subplots(3, 1)
    for i in range(old_hsv.shape[2]):
        ax[i].hist(old_hsv[:, :, i].flatten(), bins=256)

    fig, ax = plt.subplots()
    ax.imshow(old_frame)
    if gen_markers:
        mg = MarkerGenerator(fig, ax)

    if preprocessing:
        if not markers:
            p0 = np.array(video_params[filename]['markers'], dtype=np.float32)
        else:
            p0 = np.array(markers, dtype=np.float32)

        print("Usando los parametros del script")

        low_thres_marker = (30, 40, 100)
        up_thres_marker = (70, 230, 200)

        # parametro para buscar el background
        contour_low_thres = (0, 0, 60)
        contour_up_thres = (255, 100, 255)

        thres_params = dict(low_thres=low_thres_marker,
                            up_thres=up_thres_marker,
                            contour_low_thres=contour_low_thres,
                            contour_up_thres=contour_up_thres,
                            )
        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(30, 30),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    else:
        print("Usando parametros que están en 'video_params'")
        try:
            p0 = np.array(video_params[filename]['markers'], dtype=np.float32)
        except KeyError:
            print("No hay markers para este archivo en 'video_params'")
            raise

        try:
            thres_params = {key: tuple(item) for key, item in video_params[filename]['thres_params'].items()}
        except KeyError:
            print("No hay parámetros de threshold para este archivo")
            raise

        try:

            lk_params = dict(winSize=tuple(video_params[filename]['lk_params']['winSize']),
                             maxLevel=3,
                             criteria=tuple(video_params[filename]['lk_params']['criteria']))
        except:
            print("No hay parámetros para el algo lk")
            raise

    p0 = p0.reshape((p0.shape[0], 1, p0.shape[1]))

    #
    # plt.figure()
    # plt.imshow(old_hsv)

    # levanto los marcadores
    old_hsv_thresholded = cv2.inRange(old_hsv, thres_params['low_thres'], thres_params['up_thres'])
    # levanto el contorno
    contour_hsv_thresholded = cv2.inRange(old_hsv, thres_params['contour_low_thres'], thres_params['contour_up_thres'])

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    ax.flatten()[0].imshow(old_hsv_thresholded)
    ax.flatten()[1].imshow(old_frame)
    ax.flatten()[2].imshow(old_hsv)
    ax.flatten()[3].imshow(contour_hsv_thresholded)

    contours, hierarchy = cv2.findContours(contour_hsv_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    parents = np.unique(hierarchy[0, :, 3])
    num_children = [np.count_nonzero(hierarchy[0, :, 3] == pr) for pr in parents]
    ct_size = 0
    for pr, nc in zip(parents, num_children):
        if len(contours[pr]) > ct_size and nc > 6:
            leech_ct = contours[pr]
            ct_size = len(contours[pr])
    #
    # for i in range(len(contours)):

    #     if (hierarchy[0,i,2] != -1) and any ([hierarchy[0, i, 3] == val for val in [-1, 0]]):
    #         if len(contours[i])>ct_size:
    #             leech_ct = contours[i]
    #             ct_size = len(contours[i])

    #
    ax.flatten()[0].scatter(leech_ct[:, :, 0], leech_ct[:, :, 1])
    ax.flatten()[1].scatter(leech_ct[:, :, 0], leech_ct[:, :, 1])
    ax.flatten()[2].scatter(leech_ct[:, :, 0], leech_ct[:, :, 1])

    ax.flatten()[1].scatter(p0[:, 0, 0], p0[:, 0, 1], marker='x', c='r')
    ax.flatten()[2].scatter(p0[:, 0, 0], p0[:, 0, 1], marker='x', c='r')
    ax.flatten()[3].scatter(p0[:, 0, 0], p0[:, 0, 1], marker='x', c='r')

    # Create a mask image for drawing purposes
    # mask = np.zeros_like(old_frame)`
    # untracked_frames = []
    frame_no = 0
    # cap = cv2.VideoCapture(vid_path + filename)
    print(' ')
    if frame_processing:

        while True:

            # untracked_frames.append(np.copy(frame))

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

            # contours, hierarchy = cv2.findContours(hsv_thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # ct_size = 0
            # for i in range(len(contours)):
            #     if (hierarchy[0,i,2] != -1) and (hierarchy[0, i, 3] == 0):
            #         if len(contours[i])>ct_size:
            #             leech_ct = contours[i]
            #             ct_size = len(contours[i])
            # if ct_size == 0:
            #     print("NO ENCONTRE NINGUNO QUE CUMPLA LOS REQUISITOS")

            leech_ct = getLeechContour(hsv_thres)

            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_tracked, tracking_frame, p0, None, **lk_params)
            if not all(st):
                break

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            frame = cv2.drawContours(frame, [leech_ct], 0, (250, 0, 0), 3)
            # draw the segments
            for i, (next, prev) in enumerate(zip(good_new[1:], good_new[:-1])):
                a, b = prev.ravel().astype(int)
                c, d = next.ravel().astype(int)
                frame = cv2.line(frame, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            frame = cv2.circle(frame, (good_new[-1, 0], good_new[-1, 1]), 5, color[i].tolist(), -1)
            img = frame
            cv2.imshow('frame', img)

            k = cv2.waitKey(5) & 0xff
            if k == 27:
                break

            # Now update the previous frame and previous points
            old_tracked = tracking_frame.copy()
            p0 = good_new.reshape(-1, 1, 2)
            frame_no += 1

            if frame_no % 10 == 0:
                sys.stdout.write('\rframe no:%i of %i (%is/%is)' % (
                frame_no, total_frames, int(frame_no / fps), int(total_frames / fps)) + ' ' * 20)
                sys.stdout.flush()  # important

        cv2.destroyAllWindows()
        cap.release()

        print("read %i out of %i frames" % (frame_no, total_frames))

        comments = """Es una prueba del nuevo color"""
    # markers = video_params[filename]['markers']
    #
    if update_dict:
        video_dict = dict(markers=markers,
                          lk_params=lk_params,
                          thres_params=thres_params,
                          comments=comments,
                          analyze=False,
                          leech=7,
                          )
        video_params[filename] = video_dict
        print("Updated Dict")
    else:
        print("Not updating dict")
    if update_json:
        updateJson()
    else:
        print("Not updating json")
