# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:06:44 2021

@author: main
"""

import numpy as np
import Utils.opencvUtils as cvU
from sklearn.manifold import TSNE
from sklearn.cluster import OPTICS, DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from tslearn import metrics, barycenters
import dtw
from scipy.stats import median_abs_deviation
transform = False
cmap = plt.get_cmap('tab10')
if __name__ == "__main__":

    # filename = "output_latest/crawling_length_normalized_5segments_10bins_vidname_cycleno"
    filename = "output_latest/crawling_length_normalized_5segments_20bins_vidname_cycleno"
    # filename = "output_latest/crawling_length_normalized_5segments_40bins_vidname_cycleno"
    
    fig_path = "output_latest/figures/" #este es el lugar donde se guardan las figuras, tiene que existir o tira error al guardar
    store = cvU.pickleDfArrayLoad(filename)
    binned_lengths = store['matrix']

    
    ## genero una matriz sin el primer y ultimo segment(se podría hacer el tratamiento con esta matriz adaptando un poquito el código)
    # dropped_lengths = binned_lengths[:, 1:-1, :]

    df = store['df']

    metadata = store['metadata']
    print(df.head())
    print(df.describe())
    df['leech_no'] = pd.to_numeric(df.leech_no).astype(int)

    leech_index = np.in1d(df.leech_no.values, (2, 3, 4, 6, 8, 10, 11)) ## tomo las filas con ciclos de estas sanguis
    df = df[leech_index]
    df.reset_index(inplace=True, drop=True)
    binned_lengths = binned_lengths[leech_index]

    reset_cycles, errored_rows = cvU.checkCyclesReset(binned_lengths, min_stretching_segs=2, min_shortening_segs=metadata['min_shortening_segs'])
    df['cycle_reset'] = reset_cycles


    # if len(errored_rows)>0:
    #     errored_rows = np.append(errored_rows, errored_rows+1)
    if (df.cycle_duration<3).sum()>0:
        errored_rows = np.append(errored_rows, df.index[df.cycle_duration<3].values)
    if len(errored_rows>0):
        reset_cycles = np.delete(reset_cycles, errored_rows.astype(int))
        df.drop(errored_rows.astype(int), inplace=True, errors='ignore')
        df.reset_index(inplace=True, drop=True)
        binned_lengths = np.delete(binned_lengths, errored_rows.astype(int), axis=0)





    reshaped_binned_lengths = binned_lengths.reshape(binned_lengths.shape[0], binned_lengths.shape[1] * binned_lengths.shape[2])



    dist_matrix = np.zeros((binned_lengths.shape[0], binned_lengths.shape[0]))
    transposed_lengths = binned_lengths.transpose((0, 2, 1))

    dist_matrix = metrics.cdist_dtw(transposed_lengths)
    #
    # for i in range(binned_lengths.shape[0]):
    #     for j in range(i, binned_lengths.shape[0]):
    #         dist_matrix[i,j] = dtw.dtw(transposed_lengths[i], transposed_lengths[j]).distance
    #         # dist_matrix[i,j] = dtw.dtw(binned_lengths[i].T, binned_lengths[j].T).normalizedDistance
    #         dist_matrix[j,i] = dist_matrix[i, j]
    #
    
    #%%
    Encoder = LabelEncoder()
    df['encoded_video_name'] = Encoder.fit_transform(df.video_name)
    df['encoded_leech_no'] = Encoder.fit_transform(df.leech_no)
    
    tsne = TSNE(perplexity=15., early_exaggeration=12., n_iter=2000, metric='precomputed', random_state=2022)
    tsne_data = tsne.fit_transform(dist_matrix)

    ##Este valor da una nocion de como resulto el fit del tsne. Es comparativo, pero si fiteo dos veces, el que tenga
    ##el menor valor es mejor
    print(tsne.kl_divergence_)
    
    
    print(silhouette_score(tsne_data, df.encoded_video_name))
    print(silhouette_score(tsne_data, df.encoded_leech_no))
    print(silhouette_score(tsne_data, df.cycle_reset))

    #%%

    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(tsne_data[:, 0], tsne_data[:, 1])

    H = np.histogram2d(tsne_data[:, 0], tsne_data[:, 1], bins=10)

    mappable = ax[1].imshow(H[0].T, origin='lower')
    fig.colorbar(mappable)
    # fig.savefig(fig_path+'SC_tsne_solo_L2436', dpi=600, transparent=True) #descomentar para guardar la figura. entre comillas eva el nombre


    tsne_df = pd.DataFrame(tsne_data, columns=['tsne_0', 'tsne_1'])
    
    # %% Este grafico genera la figura del tsne coloreado segun el 'reset' del ciclo
    fig, ax = plt.subplots()
    for i in np.unique(df.cycle_reset.values):
        ax.scatter(tsne_data[df.cycle_reset == i, 0], tsne_data[df.cycle_reset == i, 1], c=(cmap(int(i)),), label=i)
    fig.suptitle('cycle reset')
    fig.legend()
    # fig.savefig(fig_path+'SC_tsne_por_cycle_reset_solo_L2346', dpi=600, transparent=True)
    j = 0
    fig, ax = plt.subplots()
    for i in np.unique(df.leech_no.values):
        ax.scatter(tsne_data[df.leech_no == i, 0], tsne_data[df.leech_no == i, 1], c=(cmap(int(j)),), label=i)
        j += 1
    fig.suptitle('leech_no')
    fig.legend()
    #
    # k = 0
    # fig, ax = plt.subplots()
    # for i in np.unique(df.leech_no.values):
    #     for j in np.unique(df.video_date[df.leech_no==i]):
    #         ax.scatter(tsne_data[(df.leech_no == i) & (df.video_date==j) , 0],
    #                    tsne_data[(df.leech_no == i) & (df.video_date==j), 1], c=(cmap(int(k)),), label='L{}d{}'.format(i, j))
    #
    #         k += 1
    #
    # fig.suptitle('leech_no_by_date')
    # fig.legend()
    #

     # %% DBSCAN, se puede jugar con los parametros `eps` y `min_samples`
    
    clustering = DBSCAN(eps=11, min_samples=50)
    pred = clustering.fit_predict(tsne_data)
    print(clustering.core_sample_indices_.shape[0])
    df['pred'] = pred
    tsne_df['pred'] = pred
    tsne_df['cycle_reset'] = df.cycle_reset
    tsne_df['leech_no'] = df.leech_no

    for i in np.unique(pred):
        print(i, np.count_nonzero(i == pred))

    fig, ax = plt.subplots()
    for i in np.unique(tsne_df.pred):
        if i == -1:
            col = 20
        else:
            col = i
        ax.scatter(tsne_data[tsne_df.pred == i, 0], tsne_data[tsne_df.pred == i, 1], c=(cmap(int(col)),), label=col)
    fig.suptitle('pred')
    fig.legend()
    # fig.savefig(fig_path+'tsne_directo_por_pred_solo_L2346', dpi=600, transparent=True) #descomentar para guardar la figura. entre comillas eva el nombre

    # tsne_df = tsne_df[tsne_df.leech_no==6.]
    sns.pairplot(tsne_df.iloc[clustering.core_sample_indices_], vars=['tsne_0', 'tsne_1'], hue='pred',
                 palette=sns.color_palette()[:np.unique(tsne_df.pred[clustering.core_sample_indices_]).shape[0]])
    # plt.savefig(fig_path+'dbscan_pairplot_core_samples_solo_L2346', dpi=600, transparent=True) #descomentar para guardar la figura. entre comillas va el nombre
    sns.pairplot(tsne_df, vars=['tsne_0', 'tsne_1'], hue='pred',
                 palette=sns.color_palette()[:np.unique(tsne_df.pred).shape[0]], )

    # plt.savefig(fig_path+'dbscan_pairplot_full_samples_solo_L2346', dpi=600, transparent=True)  #descomentar para guardar la figura. entre comillas va el nombre

    print(df.drop(['cycle_time', 'errored_cycle', 'encoded_leech_no', 'encoded_video_name'], axis=1).groupby(['pred', 'cycle_reset']).describe())
    #%%
    good_mask = (df.pred != -1).values
    dropped_df = df[good_mask]

    dropped_lengths = transposed_lengths[good_mask]
    closest_idx = []
    for cluster in np.sort(dropped_df.pred.unique()):
        bc = barycenters.dtw_barycenter_averaging(dropped_lengths[(dropped_df.pred==cluster).values])
        closest_idx.append(np.argmin(np.power(binned_lengths - bc.T,2).sum(axis=(1,2))))
        fig, ax = cvU.plotBinnedLengths(bc.T, zscore_speed=True)
        fig.suptitle('cluster {}'.format(cluster))

    for idx in closest_idx:
        f, a = cvU.plotBinnedLengths(binned_lengths[idx])
        fig.suptitle('idx: {}, cluster: {}'.format(idx, df.pred[idx]))

    df.drop(['cycle_reset', 'errored_cycle', 'encoded_video_name', 'encoded_leech_no'], axis=1).loc[closest_idx]

