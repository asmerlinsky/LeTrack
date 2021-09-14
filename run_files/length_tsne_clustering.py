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

if __name__ == "__main__":

    # filename = "output_latest/crawling_length_normalized_5segments_10bins_20compPCA_vidname_cycleno"
    # filename = "output_latest/crawling_length_normalized_5segments_20bins_vidname_cycleno"
    # filename = "output_edges/crawling_length_normalized_allsegments_20bins_vidname_cycleno"
    # filename = "output_edges/crawling_length_normalized_8segments_20bins_vidname_cycleno"
    filename = "output_edges/crawling_length_normalized_8segments_10bins_vidname_cycleno"

    # fig_path = "output_latest/figures_210803/"
    fig_path = "output_edges/figures/"
    store = cvU.pickleDfArrayLoad(filename)
    binned_lengths = store['matrix']
    reset_cycles = cvU.checkCyclesReset(binned_lengths, min_stretching_segs=1,min_shortening_segs=1)
    dropped_lengths = binned_lengths[:, 1:-1, :]
    df = store['df']
    metadata = store['metadata']
    df['cycle_reset'] = reset_cycles
    print(df.head())
    print(df.describe())
    cmap = plt.get_cmap('tab10')


    print("cycles by leech:\n")
    for i in np.unique(df.leech_no):
        print("{0}: {1:2.2f}".format(i, np.count_nonzero(i == df.leech_no)/df.leech_no.shape[0]))

    leech_index = np.in1d(df.leech_no.values, (2, 3, 4, 6))
    # leech_index = np.ones(df.shape[0], dtype=bool)
    df = df.iloc[leech_index]
    df = df.reset_index().drop(axis='columns', labels='index')
    dropped_lengths = dropped_lengths.reshape(dropped_lengths.shape[0], dropped_lengths.shape[1] * dropped_lengths.shape[2])[leech_index]
    reshaped_binned_lengths = binned_lengths.reshape(binned_lengths.shape[0], binned_lengths.shape[1] * binned_lengths.shape[2])[leech_index]

    scaled_lengths = StandardScaler().fit_transform(reshaped_binned_lengths)

    # %%
    tsne = TSNE(perplexity=30., early_exaggeration=12., n_iter=2000, metric='cosine')
    tsne_data = tsne.fit_transform(scaled_lengths)

    print(tsne.kl_divergence_)

    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(tsne_data[:, 0], tsne_data[:, 1])

    H = np.histogram2d(tsne_data[:, 0], tsne_data[:, 1], bins=10)

    mappable = ax[1].imshow(H[0].T, origin='lower')
    fig.colorbar(mappable)
    # fig.savefig(fig_path+'SC_tsne_solo_L2436', dpi=600, transparent=True)


    tsne_df = pd.DataFrame(tsne_data, columns=['tsne_0', 'tsne_1'])

    fig, ax = plt.subplots()
    for i in np.unique(df.cycle_reset.values):
        ax.scatter(tsne_data[df.cycle_reset == i, 0], tsne_data[df.cycle_reset == i, 1], c=(cmap(int(i)),), label=i)
    fig.suptitle('cycle reset')
    fig.legend()
    fig.savefig(fig_path+'SC_tsne_por_cycle_reset_solo_L2346', dpi=600, transparent=True)

    fig, ax = plt.subplots()
    for i in np.unique(df.leech_no.values):
        ax.scatter(tsne_data[df.leech_no == i, 0], tsne_data[df.leech_no == i, 1], c=(cmap(int(i)),), label=i, s=50)
    fig.suptitle('leech')
    fig.legend()
    fig.savefig(fig_path+'SC_tsne_por_leech_solo_L2346', dpi=600, transparent=True)
    #
    # tsne = TSNE(n_components=3, perplexity=30., early_exaggeration=12., n_iter=2000)
    # tsne_data = tsne.fit_transform(scaled_lengths)
    # print(tsne.kl_divergence_)
    # cvU.plot3Dscatter(tsne_data, color=2*df.leech_no, s=30)

    # %%
    clustering = DBSCAN(eps=3.5, min_samples=15)
    pred = clustering.fit_predict(tsne_data)
    print(clustering.core_sample_indices_.shape[0])
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
    fig.suptitle('leech')
    fig.legend()
    fig.savefig(fig_path+'tsne_directo_por_pred_solo_L2346', dpi=600, transparent=True)

    # tsne_df = tsne_df[tsne_df.leech_no==6.]
    sns.pairplot(tsne_df.iloc[clustering.core_sample_indices_], vars=['tsne_0', 'tsne_1'], hue='pred',
                 palette=sns.color_palette()[:np.unique(tsne_df.pred[clustering.core_sample_indices_]).shape[0]])
    plt.savefig(fig_path+'dbscan_pairplot_core_samples_solo_L2346', dpi=600, transparent=True)
    sns.pairplot(tsne_df, vars=['tsne_0', 'tsne_1'], hue='pred',
                 palette=sns.color_palette()[:np.unique(tsne_df.pred).shape[0]])

    # plt.savefig(fig_path+'dbscan_pairplot_full_samples_solo_L2346', dpi=600, transparent=True)


    # %%
    nc = 7
    cols = ['col_{}'.format(i) for i in range(nc)]

    pca = PCA(n_components=nc)
    tf_scaled_lengths = pca.fit_transform(scaled_lengths)
    db_pca_df = pd.DataFrame(tf_scaled_lengths, columns=cols)
    db_pca_df['pred'] = pred
    db_pca_df['leech_no'] = df.leech_no
    db_pca_df['cycle_reset'] = df.cycle_reset

    print_df = df[["leech_no", 'video_name']].copy()
    print_df['pred'] = pred

    desc = print_df.groupby(['leech_no', 'pred']).describe()
    desc.to_csv(fig_path+"leech_pred_groupby_L2346.csv")
    desc = print_df.groupby(['pred', 'leech_no']).describe()
    desc.to_csv(fig_path+"pred_leech_groupby_L2346.csv")


    # db_pca_df = db_pca_df[db_pca_df.pred != -1]
    # sns.pairplot(db_pca_df, vars=cols, hue='pred', diag_kind='hist',
    #              palette=sns.color_palette()[:np.unique(db_pca_df.pred).shape[0]])
    # sns.pairplot(db_pca_df, vars=cols, hue='leech_no', diag_kind='hist',
    #              palette=sns.color_palette()[:np.unique(db_pca_df.leech_no).shape[0]])
    # sns.pairplot(db_pca_df, vars=cols, hue='cycle_reset', diag_kind='hist',
    #              palette=sns.color_palette()[:np.unique(db_pca_df.cycle_reset).shape[0]])


    filtered_matrix = binned_lengths[leech_index]

    grouping = db_pca_df.pred

    for i in grouping.unique():
        # mask = ((db_pca_df.pred == i) & (db_pca_df.leech_no == 6.))
        mask = (grouping == i)
        mat = db_pca_df.loc[mask].drop(axis=1, labels=['pred', 'leech_no']).values
        closest_cycle = np.argmin(np.power(mat - mat.mean(axis=0),2).sum(axis=1))
        closest_cycle_idx = db_pca_df.index[mask][closest_cycle]
        cycle = filtered_matrix[closest_cycle_idx]

        speeds = np.diff(cycle, axis=1)
        print(speeds.max(), speeds.min())
        fig, ax = plt.subplots(cycle.shape[0], 2)
        fig.suptitle("cluster {}".format(i))

        for j in range(cycle.shape[0]):
            ax[j, 0].plot(cycle[j], c='k')
            ax[j, 1].imshow(speeds[j][np.newaxis, :], cmap="bwr", vmin=-.15, vmax=.15, aspect="auto")
        fig.savefig(fig_path+'center_cycle_{}.png'.format(i), dpi=600, transparent=True)

    # %%

    filtered_matrix = scaled_lengths.reshape(scaled_lengths.shape[0], metadata['num_segments'], metadata['num_bins'])
    lineplot_list = []
    for trial in range(filtered_matrix.shape[0]):
        for segment in range(filtered_matrix.shape[1]):
            for tm in range(filtered_matrix.shape[2]):
                lineplot_list.append([trial, segment, tm, filtered_matrix[trial, segment, tm], pred[trial]])



    lineplot_df = pd.DataFrame(data=lineplot_list, columns=['trial', 'segment_no', 'time', 'length', 'pred'])
    lineplot_df = lineplot_df[lineplot_df.pred != -1]
    fig, ax = plt.subplots(filtered_matrix.shape[1])
    for segment in range(filtered_matrix.shape[1]):
        sns.lineplot(x='time', y='length', hue='pred', data=lineplot_df[lineplot_df.segment_no==segment], ax=ax[segment],ci='sd', palette=cmap,legend=False)

    # %%
    n_components = 2
    reducer = umap.UMAP(n_neighbors=10, min_dist=.5, n_components=n_components, metric='cosine')
    filtered_matrix = binned_lengths[leech_index]

    filtered_matrix = filtered_matrix.reshape(filtered_matrix.shape[0], filtered_matrix.shape[1]*filtered_matrix.shape[2])
    filtered_matrix = StandardScaler().fit_transform(filtered_matrix)
    embedding = reducer.fit_transform(filtered_matrix)
    if n_components==2:
        cols = ['umap1', 'umap2']
        umap_df = pd.DataFrame(embedding, columns=cols)
        umap_df['leech_no'] = df.leech_no
        umap_df['cycle_type'] = reset_cycles[leech_index]
        umap_df['video_name'] = df.video_name

        sns.pairplot(umap_df, vars=cols, hue='leech_no', diag_kind='hist',
                     palette=sns.color_palette('tab10')[:np.unique(umap_df.leech_no).shape[0]])
        plt.savefig('umap_pairplot', dpi=600, transparent=True)
    elif n_components==3:
        cvU.plot3Dscatter(embedding, color=reset_cycles[leech_index],s=10)
# %%
#     n_components = 2
#     reducer = umap.UMAP(n_neighbors=5, min_dist=.2, n_components=n_components)
#     filtered_matrix = binned_lengths[leech_index]
#
#     filtered_matrix = filtered_matrix.reshape(filtered_matrix.shape[0], filtered_matrix.shape[1]*filtered_matrix.shape[2])
#     filtered_matrix = StandardScaler().fit_transform(filtered_matrix)
#     embedding = reducer.fit_transform(filtered_matrix)
#     if n_components==2:
#         cols = ['umap1', 'umap2']
#         umap_df = pd.DataFrame(embedding, columns=cols)
#         umap_df['leech_no'] = df.leech_no
#         umap_df['cycle_type'] = reset_cycles[leech_index]
#         umap_df['video_name'] = df.video_name
#         sns.pairplot(umap_df[umap_df.leech_no==6], vars=cols, hue='video_name', diag_kind='hist',
#                      palette=sns.color_palette('tab20')[:np.unique(umap_df[umap_df.leech_no==6].video_name).shape[0]])
#     elif n_components==3:
#         cvU.plot3Dscatter(embedding, color=reset_cycles[leech_index],s=10)

    # %%
    num_preds = 100
    np.random.seed(7262021)
    pred_list = []
    for i in range(num_preds):
        clustering = DBSCAN(eps=5, min_samples=20, )
        pred = clustering.fit_predict(tsne_data)
    print(clustering.core_sample_indices_.shape[0])
    tsne_df['pred'] = pred

    # %%
    from scipy.stats import median_abs_deviation
    for vn in df.video_name.unique():
        diffs = np.diff(binned_lengths[df.video_name==vn],axis=2)
        print("{}:\t {:.3f}, {:.3f}, {:.3f}".format(vn, np.mean(diffs), np.std(diffs), median_abs_deviation(diffs,axis=None)))

    for i in np.random.randint(0, binned_lengths.shape[0], size=5):
        bl_diff = np.diff(binned_lengths[i],axis=1)
        print(i, np.mean(bl_diff), np.std(bl_diff), reset_cycles[i])
        print(df.iloc[i])
        print(df.iloc[i+1].cycle_time)
        # cvU.plotBinnedLengths(binned_lengths[i], zscore_speed=True)
        cvU.plotBinnedLengths(binned_lengths[i], zscore_speed=False)

