import numpy as np
from utils.constants import *

def normalize_sigmoid(Y):
    min_y = np.min(Y)
    max_y = np.max(Y)
    return (Y - min_y) / (max_y - min_y), min_y, max_y

def denormalize_sigmoid(O, min_y, max_y):
    return min_y + O * (max_y - min_y)

def normalize_tanh(Y):
    min_y = np.min(Y)
    max_y = np.max(Y)
    return 2 * (Y - min_y) / (max_y - min_y) - 1, min_y, max_y

def denormalize_tanh(O, min_y, max_y):
    return min_y + (O + 1) * (max_y - min_y) / 2

def normalize(Y, activation):
    if (activation == SIGMOID):
        return normalize_sigmoid(Y)
    return normalize_tanh(Y)

def denormalize(O, min_y, max_y, activation):
    if (activation == SIGMOID):
        return denormalize_sigmoid(O, min_y, max_y)
    return denormalize_tanh(O, min_y, max_y)