from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import matplotlib.pyplot as plt


'''
Implement your own PR-curve function
NOTE: only modify TODO place. 
Leave other code blocks unchanged.
'''

def precision_recall_curve_manual(y_true, y_pred):
    th = 0.001

    precision = []
    recall = []

    while th <1.0:

        tp = np.sum( np.logical_and(y_true==1, y_pred >=th ))

        # TODO: compute fn and fp as above line
        fn = 0 # modify
        fp = 0 # modify
        # end TODO

        prec = 0
        if tp+fp>0:
            prec = 1.0*tp/(tp+fp)
        rec = 0
        if tp+fn>0:
            rec = 1.0*tp / (tp+fn)

        if prec>0 or rec > 0:
            precision.append(prec)
            recall.append(rec)

        th += 0.001

    #####################################
    # calculate the area under pr-curve
    #####################################
    # sort in X-axis with recall
    pr_pairs = [(r,p) for r, p in zip(recall, precision)]
    pr_pairs_sorted = sorted(pr_pairs)
    area = 0.0

    # TODO: compute area under P-R curve
    # hint: loop over pr_pairs_sorted, the adjacent pairs add up to area
    # the total area is the average precision

    # end TODO


    return precision, recall, area


if __name__ == '__main__':

    ##################################
    ## DO NOT CHANGE ANY CODE BELOW ##
    ##################################

    y_pred, y_true = np.load('y_pred.npy'), np.load('y_true.npy')
    n_classes = 10
    print(y_pred.shape, y_true.shape)

    precision = dict()
    recall = dict()
    aps = []
    for c in range(n_classes):
        precision[c], recall[c], ap = precision_recall_curve_manual(y_true[:, c], y_pred[:, c])
        plt.plot(recall[c], precision[c], lw=2, label='class {}'.format(c))
        aps.append(ap)

    # calculate mean average precision
    mAP = np.mean(aps)
    print('mean Average Precision', mAP)


    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.savefig('fig_manual.png')
    # plt.show()
    plt.close()