import json
import os

import cv2
import numpy as np

if __name__ == '__main__':

    data_path = "tracked_data/"

    with open('marker_dict.json', 'r') as fp:
        marker_dict = json.load(fp)

    file_list = list(marker_dict)

    basename = os.path.splitext(os.path.basename(file_list[6]))[0]
    filename = data_path + basename + '_centered.npy'

    print("Running %s" % filename)

    trace = np.load(filename)

    cap = cv2.VideoCapture('vids/%s_out.avi' % basename)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_size = 150
    cap_list_out = []
    for i in range(trace.shape[0]):
        cap_list_out.append(cv2.VideoWriter("vids/zoomed/%s_%i.avi" % (basename, i),
                                            cv2.VideoWriter_fourcc(*'DIVX'),
                                            fps, (out_size, out_size))
                            )
    j = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for i in range(trace.shape[0]):
            y, x = np.round(trace[i, j]).astype(int)
            x_from = int(x - out_size / 2)
            x_to = int(x + out_size / 2)
            y_from = int(y - out_size / 2)
            y_to = int(y + out_size / 2)
            frame_copy = frame[x_from:x_to, y_from:y_to].copy()
            dx = 25
            dy = dx

            grid_color = (213, 213, 213)
            frame_copy[:, ::dy] = grid_color
            frame_copy[::dx, :] = grid_color

            cap_list_out[i].write(frame_copy)

        j += 1

    cv2.destroyAllWindows()
    cap.release()
    for out_cap in cap_list_out:
        out_cap.release()
