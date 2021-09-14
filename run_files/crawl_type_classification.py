from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import Utils.opencvUtils as cvU
from sklearn.metrics import roc_auc_score, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import seaborn as sns



if __name__=="__main__":

    # df, metadata = cvU.h5load("output_latest/crawling_normalizedDAP_pca_20comp_vidname_cycleno.h5")
    df, metadata = cvU.h5load("output_latest/crawling_length_normalizedDAP_pca_10comp_vidname_cycleno.h5")
    print(df.head())
    print(df.describe())

    part_df = df.iloc[:,:10]
    part_df['video_name'] = df.video_name

    run_videos = ['21-04-26_28', '21-04-28_09','21-04-26_08']



    sns.pairplot(part_df.loc[[vn in run_videos for vn in part_df.video_name]], hue='video_name', diag_kind='hist')

    color = ['r', 'g', 'b']

    j = 0
    fig, ax = plt.subplots(2,2)
    ax.flatten()[0].set_title('Dims 0-1')
    ax.flatten()[1].set_title('Dims 1-2')
    ax.flatten()[2].set_title('Dims 2-3')
    ax.flatten()[3].set_title('Dims 0-3')

    for vn in run_videos:
        sliced_df = part_df[df.video_name==vn]

        ax.flatten()[0].plot(sliced_df.iloc[:,0], sliced_df.iloc[:, 1], 'ro', alpha = 0.5, c=color[j], label=vn)
        ax.flatten()[1].plot(sliced_df.iloc[:,1], sliced_df.iloc[:, 2], 'ro', alpha = 0.5, c=color[j])
        ax.flatten()[2].plot(sliced_df.iloc[:,2], sliced_df.iloc[:, 3], 'ro', alpha = 0.5, c=color[j])
        ax.flatten()[3].plot(sliced_df.iloc[:,0], sliced_df.iloc[:, 3], 'ro', alpha = 0.5, c=color[j])

        for i in range(sliced_df.shape[0]):
            ax.flatten()[0].text(sliced_df.iloc[i,0], sliced_df.iloc[i,1], str(i), size=12)
            ax.flatten()[1].text(sliced_df.iloc[i,1], sliced_df.iloc[i,2], str(i), size=12)
            ax.flatten()[2].text(sliced_df.iloc[i,2], sliced_df.iloc[i,3], str(i), size=12)
            ax.flatten()[3].text(sliced_df.iloc[i,0], sliced_df.iloc[i,3], str(i), size=12)
        j += 1
    fig.legend()



