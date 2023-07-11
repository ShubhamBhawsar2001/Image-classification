import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('D:\iisc\sem2\ML\SVM_Assignment1\SVM_Assignment1\data\mnist_test.csv')
    test_df = pd.read_csv('D:\iisc\sem2\ML\SVM_Assignment1\SVM_Assignment1\data\mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    
  X_train =  ((2. * X_train) / 255)-1
  X_test =  ((2. * X_test) / 255)-1
  return X_train, X_test
    

def plot_metrics(metrics):
   
    ks, accuracies, precisions, recalls, f1_scores = zip(*metrics)
    bar_width = 0.2

    # create the bar positions for each k
    r1 = np.arange(len(ks))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    # plot the results
    fig, ax = plt.subplots()
    ax.bar(r1, accuracies, width=bar_width, label='Accuracy')
    ax.bar(r2, precisions, width=bar_width, label='Precision')
    ax.bar(r3, recalls, width=bar_width, label='Recall')
    ax.bar(r4, f1_scores, width=bar_width, label='F1 Score')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Performance Metric')
    ax.set_ylim([0, 1])
    ax.set_xticks(r2)
    ax.set_xticklabels(ks)
    ax.legend()
    plt.show()