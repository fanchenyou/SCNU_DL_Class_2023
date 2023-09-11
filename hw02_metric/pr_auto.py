import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

'''
This is an example of evaluating model output y_pred with Precision-Recall metric
Read the API call below for a 10-class example.
For each class, take the one-against-the-rest as a binary problem, then call API 10 times
Understand this API code first, then implement your own PR function in pr_manual.py
'''

if __name__ == '__main__':
    # load 10-way classification results
    # y_pred of each data is 10-way prob. distribution
    # y_true is corresponding class label in one-hot form
    # i.e., a class-2 label is like (0,0,1,0,..,0)
    y_pred, y_true = np.load('y_pred.npy'), np.load('y_true.npy')
    n_classes = 10

    # visualize the first data instance
    print(y_pred.shape, y_true.shape)
    print(y_pred[0,:])
    print(y_true[0,:])

    # use scikit-learn package to draw PR-Curve
    precision = dict()
    recall = dict()
    for i in range(n_classes):
        '''
        input:
            y_true: the gt label of a data, e.g., (0,0,1).
            When we consider data i of class 0/1, y_true[i,0]=y_true[i,1]=0, with class 2 as y_true[i,2]=1

            y_pred: probability distribution of a data label prediction, e.g., (0.3, 0.1, 0.6)
            thus, its class-0 prob. is 0.3,  class-1 prob. is 0.1, class-2 prob. is 0.6
        '''
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i],
                                                            y_pred[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

    # calculate mean average precision
    mAP = average_precision_score(y_true, y_pred, average="micro")
    print('mean Average Precision', mAP)

    # save figure
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.savefig('fig_auto.png')
    # plt.show()
    plt.close()