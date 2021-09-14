from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import Utils.opencvUtils as cvU
from sklearn.metrics import roc_auc_score, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

from numpy import interp

if __name__=="__main__":

    # df, metadata = cvU.h5load("output_latest/crawling_normalizedDAP_pca_20comp_vidname_cycleno.h5")
    df, metadata = cvU.h5load("output_latest/crawling_binnedDAP_pca_20comp_vidname_cycleno.h5")
    print(df.head())
    print(df.describe())

    X_data = df.iloc[:,:10].values

    y_data = df.leech_no.values.astype(int)
    y_data = label_binarize(y_data, classes=np.unique(y_data))
    # y_data = OneHotEncoder().fit_transform(y_data)



    X_train, X_test, y_train_labeled, y_test_labeled = train_test_split(X_data, y_data)

    classifier = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', max_iter=1000, warm_start=True)

    classifier.fit(X_train, y_train_labeled)
    classifier.fit(X_train, y_train_labeled)


    print(classifier.score(X_test, y_test_labeled))
    print(classifier.score(X_train, y_train_labeled))
    probas = classifier.predict_proba(X_test)
    probas /= probas.sum(axis=1).reshape(-1,1)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(probas.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_labeled[:, i], probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_labeled.ravel(), probas.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(probas.shape[1])]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(probas.shape[1]):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= probas.shape[1]

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'yellow'])
    for i in range(probas.shape[1]):
        plt.plot(fpr[i], tpr[i], lw=1,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
