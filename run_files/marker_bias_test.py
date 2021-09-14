"""
The idea for this file is to check for possible classification bias due to videos/marker placement
And if the classification task is help by some of this
"""

from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import Utils.opencvUtils as cvU
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize, StandardScaler
import json
from numpy import interp
import seaborn as sns

if __name__=="__main__":


    with open('marker_dict.json', 'r') as fp:
        video_params = json.load(fp)

    filename = "output_latest/crawling_length_normalized_5segments_10bins_20compPCA_vidname_cycleno"
    fig_path = "output_latest/figures/"
    store = cvU.pickleDfArrayLoad(filename)
    binned_lengths = store['matrix']
    # binned_lengths = binned_lengths[:,1:-1,:]
    reshaped_lengths = binned_lengths.reshape(binned_lengths.shape[0], binned_lengths.shape[1]*binned_lengths.shape[2])

    cmap = plt.get_cmap('tab20')

    df = store['df']
    metadata = store['metadata']

    leech_index = np.in1d(df.leech_no.values, (2, 3, 4, 6))
    # leech_index = np.ones(df.shape[0], dtype=bool)
    df = df.iloc[leech_index]
    df = df.reset_index().drop(axis='columns', labels='index')
    dropped_lengths = binned_lengths.reshape(binned_lengths.shape[0], binned_lengths.shape[1] * binned_lengths.shape[2])[leech_index]

    scaled_lengths = StandardScaler().fit_transform(dropped_lengths)
    tsne = TSNE(perplexity=30., early_exaggeration=12., n_iter=2000)
    tsne_data = tsne.fit_transform(scaled_lengths)

    df['tsne_0'] = tsne_data[:,0]
    df['tsne_1'] = tsne_data[:,1]

    sns.scatterplot(data=df[df.leech_no==2.], x='tsne_0', y='tsne_1', hue='video_name')


    fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
    j = 0
    for ln in df.leech_no.unique():
        part_df = df[df.leech_no==ln]

        i = 0
        for vn in part_df.video_name.unique():
            ax.flatten()[j].scatter(part_df.tsne_0[part_df.video_name==vn], part_df.tsne_1[part_df.video_name==vn], label=vn, c=(cmap(i),))
            i += 1
        for idx, row in part_df.iterrows():
            ax.flatten()[j].text(row.tsne_0, row.tsne_1, row.video_name.split('_')[-1], size=12)
        j += 1

    fig, ax = plt.subplots()
    part_df = df[df.leech_no==6]

    i = 0
    for vn in part_df.video_name.unique():
        ax.scatter(part_df.tsne_0[part_df.video_name==vn], part_df.tsne_1[part_df.video_name==vn], label=vn, c=(cmap(i),))
        i += 1
    for idx, row in part_df.iterrows():
        ax.text(row.tsne_0, row.tsne_1, row.video_name.split('-')[-1], size=12)



