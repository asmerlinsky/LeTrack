import numpy as np
import Utils.opencvUtils as cvU
from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS, DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import umap
from scipy.stats import median_abs_deviation

if __name__ == "__main__":


    # filename = "output_latest/crawling_length_normalized_5segments_10bins_20compPCA_vidname_cycleno"
    filename = "output_latest/crawling_length_normalized_5segments_20bins_vidname_cycleno"
    fig_path = "output_latest/figures/"
    store = cvU.pickleDfArrayLoad(filename)
    binned_lengths = store['matrix']
    reset_cycles = cvU.checkCyclesReset(binned_lengths, min_stretching_segs=3)

    df = store['df']
    metadata = store['metadata']

    crawlings = binned_lengths[df[df.video_name=='21-04-22_3'].index]
    cvU.plotBinnedLengths(crawlings[0], zscore_speed=True)
    speeds = np.diff(crawlings,axis=2)
    fig, ax = plt.subplots()
    ax.hist(speeds.flatten(),bins=20)
    ax.axvline()
    print("{:.4f}, {:.4f}, {:.4f}".format(np.mean(speeds), np.std(speeds), median_abs_deviation(speeds, axis=None)))