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

from scipy.stats import median_abs_deviation
transform = False
cmap = plt.get_cmap('tab10')
if __name__ == "__main__":

    # filename = "output_latest/crawling_length_normalized_5segments_10bins_vidname_cycleno"
    filename = "output_latest/crawling_length_normalized_5segments_20bins_vidname_cycleno"
    # filename = "output_latest/crawling_length_normalized_5segments_5bins_vidname_cycleno"



    # filename = "output_edges/crawling_length_normalized_5segments_20bins_vidname_cycleno_v2"
    # filename = "output_edges/crawling_length_normalized_allsegments_20bins_vidname_cycleno"
    # filename = "output_edges/crawling_length_normalized_8segments_20bins_vidname_cycleno"
    # filename = "output_edges/crawling_length_normalized_8segments_10bins_vidname_cycleno"

    # fig_path = "output_latest/figures_210803/"
    fig_path = "output_edges/figures/" #este es el lugar donde se guardan las figuras, tiene que existir o tira error al guardar
    store = cvU.pickleDfArrayLoad(filename)
    binned_lengths = store['matrix']

    
    ## genero una matriz sin el primer y ultimo segment(se podría hacer el tratamiento con esta matriz adaptando un poquito el código)
    # dropped_lengths = binned_lengths[:, 1:-1, :]

    df = store['df']

    metadata = store['metadata']

    #%% Estas dos lineas muestran el formato del dataframe cargado, y una estadistica simple de las columnas
    print(df.head())
    print(df.describe())
    #%%
    print("cycles by leech:\n")
    for i in np.unique(df.leech_no):
        print("{0}: {1:2.2f}".format(i, np.count_nonzero(i == df.leech_no)/df.leech_no.shape[0]))

    # leech_index = np.in1d(df.leech_no.values, (2, 3, 4, 6)) ## tomo las filas con ciclos de estas sanguis


    reset_cycles, errored_rows = cvU.checkCyclesReset(binned_lengths, min_stretching_segs=metadata['min_stretching_segs'], min_shortening_segs=metadata['min_shortening_segs'])

    df.index[df.cycle_duration<3]
    if len(errored_rows)>0:
        errored_rows = np.append(errored_rows, errored_rows+1)
    if (df.cycle_duration<3).sum()>0:
        errored_rows = np.append(errored_rows, df.index[df.cycle_duration<3].values)
    if len(errored_rows>0):
        reset_cycles = np.delete(reset_cycles, errored_rows)
        df.drop(errored_rows, inplace=True, errors='ignore')
        df.reset_index(inplace=True, drop=True)
        binned_lengths = np.delete(binned_lengths, errored_rows, axis=0)


    leech_index = np.ones(binned_lengths.shape[0], dtype=bool)

    print(reset_cycles.sum() / reset_cycles.shape[0])

    ## Aca filtro para quedarme solo con los cyclos de las sanguijuelas elegidas
    df = df.iloc[leech_index]
    # df['cycle_reset'] = reset_cycles
    ## Transformo la matriz para que todos los segmentos queden en la misma fila y tener un vector por cada ciclo en vez
    # de una matriz
    reshaped_binned_lengths = binned_lengths.reshape(binned_lengths.shape[0], binned_lengths.shape[1] * binned_lengths.shape[2])[leech_index]


    #%% con esta linea podria quedarme solo con los casos donde encontro un reset en el ciclo. (o en los que no cambiando un poco esta linea.)
    # reshaped_binned_lengths = reshaped_binned_lengths[reset_cycles.astype(bool)]
    Encoder = LabelEncoder()
    df['encoded_video_name'] = Encoder.fit_transform(df.video_name)
    df['encoded_leech_no'] = Encoder.fit_transform(df.leech_no)

    if transform:
        scaled_lengths = StandardScaler().fit_transform(reshaped_binned_lengths)
        nc = 25 #Cuantos elementos voy a usar en el pca
        cols = ['col_{}'.format(i) for i in range(nc)]

        pca = PCA(n_components=nc)
        tf_scaled_lengths = pca.fit_transform(scaled_lengths)
        db_pca_df = pd.DataFrame(tf_scaled_lengths, columns=cols)
    else:
        scaled_lengths = reshaped_binned_lengths
        tf_scaled_lengths = scaled_lengths



    #%% Corro el tsne sobre los ciclos transformados antes
    tsne = TSNE(perplexity=15., early_exaggeration=12., n_iter=2000, metric='cosine', random_state=2022)
    tsne_data = tsne.fit_transform(scaled_lengths)

    ##Este valor da una nocion de como resulto el fit del tsne. Es comparativo, pero si fiteo dos veces, el que tenga
    ##el menor valor es mejor
    print(tsne.kl_divergence_)

    print(silhouette_score(tsne_data, df.encoded_video_name))
    print(silhouette_score(tsne_data, df.encoded_leech_no))
    print(silhouette_score(tsne_data, df.cycle_reset))

    print(silhouette_score(reshaped_binned_lengths, df.encoded_video_name, metric='cosine'))
    print(silhouette_score(tsne_data, df.encoded_leech_no, metric='cosine'))
    print(silhouette_score(tsne_data, df.cycle_reset, metric='cosine'))




    # %% Esta seccion grafica los datos en el mapa del tsne y un grafico de calor para dar una idea de densidad
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

    # %% Este colorea segun que leech es
    # fig, ax = plt.subplots()
    # for i in np.unique(df.leech_no.values):
    #     ax.scatter(tsne_data[df.leech_no == i, 0], tsne_data[df.leech_no == i, 1], c=(cmap(int(i)),), label=i, s=50)
    # fig.suptitle('leech')
    # fig.legend()
    # fig.savefig(fig_path+'SC_tsne_por_leech_solo_L2346', dpi=600, transparent=True) #descomentar para guardar la figura. entre comillas va el nombre


    # %% DBSCAN, se puede jugar con los parametros `eps` y `min_samples`
    
    clustering = DBSCAN(eps=9, min_samples=30)
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


    # %% Esta parte genera un nuevo dataframe con la informacion del pca para generar figuras
    db_pca_df['pred'] = pred
    db_pca_df['leech_no'] = df.leech_no
    db_pca_df['cycle_reset'] = df.cycle_reset

    print_df = df[["leech_no", 'video_name']].copy()
    print_df['pred'] = pred

    # %% guarda unos datos con estadisticas del dataframe a un csv
    # desc = print_df.groupby(['leech_no', 'pred']).describe()
    # desc.to_csv(fig_path+"leech_pred_groupby_L2346.csv")
    # desc = print_df.groupby(['pred', 'leech_no']).describe()
    # desc.to_csv(fig_path+"pred_leech_groupby_L2346.csv")

    # %% grafica el cluster
    db_pca_df = db_pca_df[db_pca_df.pred != -1] # limpia los ciclos que no cayeron en ningun cluster
    # sns.pairplot(db_pca_df, vars=cols, hue='pred', diag_kind='hist',
    #              palette=sns.color_palette()[:np.unique(db_pca_df.pred).shape[0]])

    filtered_matrix = binned_lengths

    # %% graficao el ciclo mas cercano a un ciclo promedio
    # mat = db_pca_df.drop(axis=1, labels=['pred', 'leech_no']).values
    # closest_cycle = np.argmin(np.power(mat - mat.mean(axis=0), 2).sum(axis=1))
    # closest_cycle_idx = db_pca_df.index[closest_cycle]
    # cycle = filtered_matrix[closest_cycle_idx]
    #
    # speeds = np.diff(cycle, axis=1)
    # print(speeds.max(), speeds.min())
    #
    # fig = plt.figure(constrained_layout=True)
    # gs = GridSpec(cycle.shape[0], 2, figure=fig)
    #
    # for j in range(cycle.shape[0]):
    #     ax = fig.add_subplot(gs[j,0])
    #     ax.plot(cycle[j], c='k')
    #     ax.xaxis.set_visible(False)
    #     ax.set_ylim([0, 1.1])
    # ax.xaxis.set_visible(True)
    # ax = fig.add_subplot(gs[:,1])
    # mad = median_abs_deviation(speeds, axis=None)
    # ax.imshow(speeds, cmap="bwr", vmin=-mad, vmax=mad, aspect="auto")
    #
    # fig.suptitle("average cycle\nmad={:2.5f}".format(mad))
    #

    # fig.savefig(fig_path + 'everyavgdcycle.png'.format(i), dpi=600, transparent=True)

    # %% grafica los ciclos por cluster, tomando tambien el ciclo mas cercano al ciclo promedio
    grouping = db_pca_df.pred
    closest_cycle_list = []
    for i in grouping.unique():
        # mask = ((db_pca_df.pred == i) & (db_pca_df.leech_no == 6.))
        mask = (grouping == i)
        mat = db_pca_df.loc[mask].drop(axis=1, labels=['pred', 'leech_no', 'cycle_reset']).values
        reset = cvU.checkCyclesReset(np.array([mat]), min_stretching_segs=1, min_shortening_segs=1)[0][0]
        closest_cycle = np.argmin(np.power(mat - mat.mean(axis=0),2).sum(axis=1))
        closest_cycle_idx = db_pca_df.index[mask][closest_cycle]
        closest_cycle_list.append(closest_cycle_idx)
        cycle = filtered_matrix[closest_cycle_idx]

        speeds = np.diff(cycle, axis=1)
        print(speeds.max(), speeds.min())

        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(cycle.shape[0], 2, figure=fig)

        for j in range(cycle.shape[0]):
            ax = fig.add_subplot(gs[j, 0])
            ax.plot(cycle[j], c='k')
            ax.xaxis.set_visible(False)
            ax.set_ylim([0, 1.1])
        ax.xaxis.set_visible(True)
        ax = fig.add_subplot(gs[:, 1])
        mad = 3*median_abs_deviation(speeds, axis=None)
        ax.imshow(speeds, cmap="bwr", vmin=-mad, vmax=mad, aspect="auto")
        fig.suptitle("cluster {}, reset ={} \nmad={:2.5f}".format(i, reset, mad))
        # ax.imshow(speeds, cmap="bwr", vmin=-.15, vmax=.15, aspect="auto")
        # fig.savefig(fig_path+'center_cycle_{}.png'.format(i), dpi=600, transparent=True)  #descomentar para guardar la figura. entre comillas va el nombre



    # %% Esta parte grafica la serie temporal de cada segmento para cada cluster. Cada fila corresponde a un segmento
    # y cada color corresponde a un cluster

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
