import numpy as np
from torchmetrics.text.rouge import ROUGEScore

def accuracy(pred, gt):
    if pred == gt:
        return 1.0
    else:
        return 0.0

def get_rouge(generated_text, target_text):
    rouge = ROUGEScore()
    return rouge([generated_text], [target_text])["rougeL_fmeasure"]
