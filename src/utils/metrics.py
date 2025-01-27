import numpy as np

def accuracy(pred, gt):
    if pred == gt:
        return 1.0
    else:
        return 0.0
