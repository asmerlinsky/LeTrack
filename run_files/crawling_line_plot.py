import numpy as np
import Utils.opencvUtils as cvU

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":


    # filename = "output_latest/crawling_length_normalized_5segments_10bins_20compPCA_vidname_cycleno"
    filename = "output_latest/crawling_length_normalized_5segments_20bins_vidname_cycleno"
    fig_path = "output_latest/figures/"
    store = cvU.pickleDfArrayLoad(filename)
    binned_lengths = store['matrix']
    # binned_lengths = binned_lengths[:,1:-1,:]
    df = store['df']
    metadata = store['metadata']

    print(df.head())
    print(df.describe())
    cmap = plt.get_cmap('tab10')

    df_list = []
    for trial in range(binned_lengths.shape[0]):
        for segment in range(binned_lengths.shape[1]):
            for tm in range(binned_lengths.shape[2]):
                df_list.append([trial, segment, tm, binned_lengths[trial, segment, tm]])

    df = pd.DataFrame(data=df_list, columns=['trial', 'segment_no', 'time', 'length'])

    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="time", y="length", hue="segment_no", ci='sd')
    ax.set_ylim((0, 1.1))
    ax.grid(which='both')

    fig, ax = plt.subplots(binned_lengths.shape[1], 1)
    for i in df.segment_no.unique():
        sns.boxplot(x="time", y="length", data=df[df.segment_no==i], ax=ax[i])
        ax[i].set_ylim((0, 1.1))
        ax[i].grid(which='both')

    fig, ax = plt.subplots(binned_lengths.shape[1], 1)
    for i in df.segment_no.unique():
        sns.lineplot(data=df[df.segment_no==i], x="time", y="length", ci='sd', ax=ax[i])
        ax[i].set_ylim((0, 1.1))
        ax[i].grid(which='both')