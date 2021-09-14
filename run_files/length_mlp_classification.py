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
    binned_lengths = binned_lengths[:,1:-1,:]
    reshaped_lengths = binned_lengths.reshape(binned_lengths.shape[0], binned_lengths.shape[1]*binned_lengths.shape[2])


    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=20, ))
    ])

    df = store['df']
    metadata = store['metadata']

    print(df.head())
    print(df.describe())


    y_data = df.leech_no.values.astype(int)

    leech_index = np.in1d(y_data, (2, 3, 4, 6))
    # leech_index = np.ones(df.shape[0], dtype=bool)
    y_data = y_data[leech_index]
    X_data = reshaped_lengths[leech_index]

    for i in (1, 2, 3):
        print("{0}: {1:2.1f}%".format(i, 100*sum(cvU.checkCyclesReset(binned_lengths[leech_index], min_stretching_segs=i))/binned_lengths[leech_index].shape[0]))


    labeled_y_data = label_binarize(y_data, classes=np.unique(y_data))
    # y_data = OneHotEncoder().fit_transform(y_data)



    X_train, X_test, y_train_labeled, y_test_labeled, y_train_non_labeled, y_test_non_labeled = train_test_split(X_data, labeled_y_data, y_data)
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)


    nc = 8
    cols = ['col_{}'.format(i) for i in range(nc)]
    tf_test_df = pd.DataFrame(X_test[:, :nc], columns=cols)
    tf_test_df['leech_no'] = y_test_non_labeled
    sns.pairplot(tf_test_df, vars=cols, hue='leech_no')

    tf_train_df = pd.DataFrame(X_train[:, :nc], columns=cols)
    tf_train_df['leech_no'] = y_train_non_labeled
    sns.pairplot(tf_train_df, vars=cols, hue='leech_no')

    classifier = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', max_iter=1000, warm_start=True)

    classifier.fit(X_train, y_train_labeled)
    classifier.fit(X_train, y_train_labeled)


    print(classifier.score(X_test, y_test_labeled))
    print(classifier.score(X_train, y_train_labeled))

    test_probas = classifier.predict_proba(X_test)
    test_probas /= test_probas.sum(axis=1).reshape(-1, 1)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(test_probas.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_labeled[:, i], test_probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_labeled.ravel(), test_probas.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(test_probas.shape[1])]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(test_probas.shape[1]):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= test_probas.shape[1]

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    fig, ax = plt.subplots()
    ax.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    ax.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'yellow'])
    for i in range(test_probas.shape[1]):
        ax.plot(fpr[i], tpr[i], lw=1,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Some extension of Receiver operating characteristic to multi-class')
    fig.legend(loc="lower right")
    # fig.savefig(fig_path+'curva_rocauc_L2346', dpi=600, transparent=True)