import json
import glob
import numpy as np
import Utils.opencvUtils as cvU


if __name__ == '__main__':


    data_path = "output_edges/tracked_data_latest/"

    with open('marker_dict.json', 'r') as fp:
        video_params = json.load(fp)

    npy_files = glob.glob("{}*.npy".format(data_path))
    for npy_file in npy_files:

        if 'FAILED' in npy_file:
            print("{} failed, continuing".format(npy_file))
            continue

        basename = cvU.getBasenameFromNpy(npy_file)

        trace = np.load(npy_file)
        seg_len = np.sqrt(np.sum(np.power(np.diff(trace, axis=0), 2), axis=2))

        np.savetxt("{}{}.csv".format(data_path, basename), seg_len, delimiter=',')