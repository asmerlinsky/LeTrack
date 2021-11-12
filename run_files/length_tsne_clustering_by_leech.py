import numpy as np
import Utils.opencvUtils as cvU
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.stats import median_abs_deviation

cmap = plt.get_cmap('tab10')

if __name__ == "__main__":

    # filename = "output_latest/crawling_length_normalized_5segments_10bins_20compPCA_vidname_cycleno"
    # filename = "output_latest/crawling_length_normalized_5segments_20bins_vidname_cycleno"
    filename = "output_edges/crawling_length_normalized_5segments_20bins_vidname_cycleno_v2"
    # filename = "output_edges/crawling_length_normalized_allsegments_20bins_vidname_cycleno"
    # filename = "output_edges/crawling_length_normalized_8segments_20bins_vidname_cycleno"
    # filename = "output_edges/crawling_length_normalized_8segments_20bins_vidname_cycleno_v2"
    # filename = "output_edges/crawling_length_normalized_8segments_10bins_vidname_cycleno"

    # fig_path = "output_latest/figures_210803/"
    fig_path = "output_edges/figures/"  #este es el lugar donde se guardan las figuras, tiene que existir o tira error al guardar
    store = cvU.pickleDfArrayLoad(filename)
    binned_lengths = store['matrix']    # Esta matriz (array de numpy) tiene la evolucion temporal de cada segmento para cada ciclo.

    ## genero una matriz sin el primer y ultimo segment (se podría hacer el tratamiento con esta matriz adaptando un poquito el código)
    # dropped_lengths = binned_lengths[:, 1:-1, :]

    df = store['df'] # Este dataframe tiene informacion sobre cada ciclo.

    metadata = store['metadata']

    #%% Estas dos lineas muestran el formato del dataframe cargado, y una estadistica simple de las columnas
    print(df.head())
    print(df.describe())
    #%%
    print("cycles by leech:\n")
    for i in np.unique(df.leech_no):
        print("Leech {0} tiene {1} ciclos".format(i, np.count_nonzero(i == df.leech_no)))



    leech_index = np.in1d(df.leech_no.values, (2, 3, 4, 6))  ## tomo las filas con ciclos de estas sanguis

    # Esta funcion chequea el reset del ciclo
    reset_cycles = cvU.checkCyclesReset(binned_lengths[leech_index], min_stretching_segs=1, min_shortening_segs=1)
    print(reset_cycles.sum() / reset_cycles.shape[0])

    ## Aca filtro para quedarme solo con los ciclos de las sanguijuelas elegidas
    df = df.iloc[leech_index]
    df['cycle_reset'] = reset_cycles

    ##Solo le cambio la forma a la matriz
    # dropped_lengths = dropped_lengths.reshape(dropped_lengths.shape[0], dropped_lengths.shape[1] * dropped_lengths.shape[2])[leech_index]
    # dropped_lengths = dropped_lengths[reset_cycles.astype(bool)]

    ## Transformo la matriz para que todos los segmentos queden en la misma fila y tener un vector por cada ciclo en vez
    # de una matriz
    reshaped_binned_lengths = binned_lengths.reshape(binned_lengths.shape[0], binned_lengths.shape[1] * binned_lengths.shape[2])[leech_index]
    
    
    ## con esta linea podria quedarme solo con los casos donde encontro un reset en el ciclo. (o en los que no cambiando un poco esta linea.)
    # reshaped_binned_lengths = reshaped_binned_lengths[reset_cycles.astype(bool)] 


    scaled_lengths = StandardScaler().fit_transform(reshaped_binned_lengths)

    nc = 7 #Cuantos elementos voy a usar en el pca

    pca = PCA(n_components=nc)
    tf_scaled_lengths = pca.fit_transform(scaled_lengths)
    
    
    cols = ['col_{}'.format(i) for i in range(nc)]
    db_pca_df = pd.DataFrame(tf_scaled_lengths, columns=cols)

    # %% Corro el tsne sobre los ciclos transformados antes (tambien se puede jugar un poco con estos parametros. principalmente `perplexity` y `early_exaggeration)
    tsne = TSNE(perplexity=15., early_exaggeration=12., n_iter=2000, metric='cosine', random_state=2022)
    tsne_data = tsne.fit_transform(scaled_lengths)

    ##Este valor da una nocion de como resulto el fit del tsne. Es comparativo, pero si fiteo dos veces, el que tenga
    ##el menor valor es mejor
    print(tsne.kl_divergence_)

    # %% Esta seccion grafica los datos en el mapa del tsne y un grafico de calor para dar una idea de densidad
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(tsne_data[:, 0], tsne_data[:, 1])

    H = np.histogram2d(tsne_data[:, 0], tsne_data[:, 1], bins=10)

    mappable = ax[1].imshow(H[0].T, origin='lower')
    fig.colorbar(mappable)
    # fig.savefig(fig_path+'SC_tsne_solo_L2436', dpi=600, transparent=True) #descomentar para guardar la figura. entre comillas va el nombre


    tsne_df = pd.DataFrame(tsne_data, columns=['tsne_0', 'tsne_1'])

    # %% Este grafico genera la figura del tsne coloreado segun el 'reset' del ciclo
    fig, ax = plt.subplots()
    for i in np.unique(df.cycle_reset.values):
        ax.scatter(tsne_data[df.cycle_reset == i, 0], tsne_data[df.cycle_reset == i, 1], c=(cmap(int(i)),), label=i)
    fig.suptitle('cycle reset')
    fig.legend()
    
    #fig.savefig(fig_path+'SC_tsne_por_cycle_reset_solo_L2346', dpi=600, transparent=True) #descomentar para guardar la figura. entre comillas va el nombre

    # %% Este colorea segun que leech es
    fig, ax = plt.subplots()
    for i in np.unique(df.leech_no.values):
        ax.scatter(tsne_data[df.leech_no == i, 0], tsne_data[df.leech_no == i, 1], c=(cmap(int(i)),), label=i, s=50)
    fig.suptitle('leech')
    fig.legend()
    # fig.savefig(fig_path+'SC_tsne_por_leech_solo_L2346', dpi=600, transparent=True) #descomentar para guardar la figura. entre comillas va el nombre


    #%% clustering by leech
    # Esta es la parte en la que separo los ciclos por leech y clustereo internamente para cada leech.
    for leech in df.leech_no.unique():

        # Genero una mascara para seleccionar los ciclo de una leech especifica.
        leech_mask = (df.leech_no==leech).values

        leech_df = df[leech_mask].copy()
        leech_tsne = tsne_data[leech_mask]
        leech_tsne_df = tsne_df[leech_mask]

        #Hago el clustering. Los valores de `eps` y `min_samples` se puede modificar y van a modificar el clustering
        clustering = DBSCAN(eps=8., min_samples=15)

        pred = clustering.fit_predict(leech_tsne)
        print("{} clusters for leech {}".format(np.unique(pred).shape[0], leech))
        leech_tsne_df['pred'] = pred
        leech_tsne_df['cycle_reset'] = leech_df.cycle_reset

        for i in np.unique(pred):
            print(i, np.count_nonzero(i == pred))


        # grafica el clustering
        fig, ax = plt.subplots()
        
        for i in leech_tsne_df.pred.unique():
            ax.scatter(leech_tsne[leech_tsne_df.pred == i, 0], leech_tsne[leech_tsne_df.pred == i, 1], label=i)
        fig.suptitle('pred for leech {}'.format(leech))
        fig.legend()
        # fig.savefig(fig_path+'tsne_directo_por_pred_solo_L{}.png'.format(leech), dpi=600, transparent=True) #descomentar para guardar la figura. entre comillas va el nombre

        leech_tsne_df.pred = leech_tsne_df.pred.astype(str)

        #A partir de aca agrupa por cada cluster que encontro (para cada leech) y grafica el ciclo real mas parecido al
        # ciclo promedio del cluster
        leech_pca_df = db_pca_df[leech_mask]
        leech_pca_df.reset_index(inplace=True)
        leech_pca_df['pred'] = pred
        grouping = leech_pca_df.pred

        for i in grouping.unique():        

            # Levanta cada ciclo de cada cluster
            mask = (grouping == i)
            mat = leech_pca_df.loc[mask].drop(axis=1, labels=['pred']).values

            #busca el ciclo mas cercano al ciclo promedio del cluster (Todo hecho en las variables pca y no tsne)
            closest_cycle = np.argmin(np.power(mat - mat.mean(axis=0),2).sum(axis=1))
            closest_cycle_idx = leech_pca_df.index[mask][closest_cycle]
            cycle = binned_lengths[leech_index][leech_mask][closest_cycle_idx]

            speeds = np.diff(cycle, axis=1)
            print(speeds.max(), speeds.min())

            fig = plt.figure(constrained_layout=True)
            gs = GridSpec(cycle.shape[0], 2, figure=fig)
            fig.suptitle("cluster {}".format(i))
            for j in range(cycle.shape[0]):
                ax = fig.add_subplot(gs[j, 0])
                ax.plot(cycle[j], c='k')
                ax.xaxis.set_visible(False)
                ax.set_ylim([0, 1.1])
            ax.xaxis.set_visible(True)
            ax = fig.add_subplot(gs[:, 1])
            
            #esta parte grafica la velocidad y la escala de colores esta delimitada por la variable mad, que en este caso es 3 veces la median absoluate deviation
            mad = 3*median_abs_deviation(speeds, axis=None)
            ax.imshow(speeds, cmap="bwr", vmin=-mad, vmax=mad, aspect="auto")
            
            
            fig.suptitle("cluster {} for leech {}\n3mad={:2.5f}".format(i, leech, mad))
            
            # fig.savefig(fig_path+'center_cycle_L{}C{}.png'.format(leech, i), dpi=600, transparent=True) #descomentar para guardar la figura. entre comillas va el nombre

