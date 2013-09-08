import numpy as np


def evaluate_rmsle(pred, real):
    pred = pred.copy()
    pred[pred <= 0] = 0
    nominator = np.sqrt(np.mean((np.log(pred + 1) - np.log(real + 1)) ** 2))
    return 0.5 / nominator


def evaluate_classification(pred, real):
    return (pred == real).sum() / float(len(real))
