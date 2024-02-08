import numpy as np

def t_criterion(preds, t=0.5):
    print("t:", t)
    max = (preds == preds.max(axis=1, keepdims=1))
    positive = (preds > t)
    labels = np.logical_or(max, positive).astype("int")

    return labels

def max_criterion(preds):
    print("max_criterion")
    labels = (preds == preds.max(axis=1, keepdims=1)).astype("int")

    return labels
