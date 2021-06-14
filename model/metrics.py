from sklearn import metrics


def roc_auc(y_true, y_score):
    return 100 * metrics.roc_auc_score(y_true=y_true, y_score=y_score)
