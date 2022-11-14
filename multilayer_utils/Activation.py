import numpy as np

def sigmoid_activation(H):
    V = 1/(1 + np.exp(-H))
    cache = H
    return V, cache

def sigmoid_backward(dV, cache):    # inverse sigmoid for backward propagation
    H = cache
    sigmoid = 1/(1 + np.exp(-H))
    dH = dV * sigmoid * (1 - sigmoid)
    return dH

def relu_activation(H):
    V = np.maximum(0, H)
    cache = H
    return V, cache

def relu_backward(dV, cache):   # inverse relu for backward propagation
    H = cache
    dH = np.array(dV, copy=True)
    dH[H <= 0] = 0
    return dH

def tanh_activation(H):
    V = (np.exp(H) - np.exp(-H)) / (np.exp(H) + np.exp(-H))
    cache = H
    return V, cache

def tanh_backward(dV, cache):   # inverse tanh for backward propagation
    H = cache
    tanh = (np.exp(H) - np.exp(-H)) / (np.exp(H) + np.exp(-H))
    dH = 1 - tanh**2
    return dH