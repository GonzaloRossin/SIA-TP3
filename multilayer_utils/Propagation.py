from utils.constants import *
from multilayer_utils.Activation import *

## FORWARD ############################################################################################################

def excitation_forward(X, W, b):
    H = np.dot(W,X)+b   # hyperplane with bias
    cache = (X, W, b)
    return H, cache

def excitation_activation_forward(V_prev, W, b, activation):
    H, excitation_cache = excitation_forward(V_prev, W, b)
    if (activation == SIGMOID):
        V, activation_cache = sigmoid_activation(H)
    elif (activation == TANH):
        V, activation_cache = tanh_activation(H)
    else:
        V, activation_cache = relu_activation(H)
    cache = (excitation_cache, activation_cache)
    return V, cache

def model_forward(X, parameters, apply_bias, hidden_activation, output_activation):
    V = X
    caches = []
    if (apply_bias):
        L = len(parameters) // 2    # includes weights and biases
    else:
        L = len(parameters)
    # activate hidden layers
    for l in range(1, L):   # l=0 is the input
        V_prev = V
        if (apply_bias):
            b = parameters['b'+str(l)]
        else:
            b = 0
        V, cache = excitation_activation_forward(V_prev, parameters['W'+str(l)], b, hidden_activation)
        caches.append(cache)
    # activate output layer
    if (apply_bias):
        O, cache = excitation_activation_forward(V, parameters['W'+str(L)], parameters['b'+str(L)], output_activation)
    else:
        O, cache = excitation_activation_forward(V, parameters['W'+str(L)], 0, output_activation)
    caches.append(cache)
    return O, caches

## BACKWARD ############################################################################################################

def excitation_backward(dH, cache, apply_bias):
    V_prev, W, _ = cache
    dV_prev = np.dot(W.T, dH)   # dV_prev = delta_H/delta_V_prev
    num_examples = V_prev.shape[1]
    dW = (1/num_examples) * np.dot(dH, V_prev.T)   # dW = delta_H/delta_W
    if (apply_bias):
        db = (1/num_examples) * np.sum(dH, axis=1, keepdims=True)  # db = delta_H/delta_b
    else:
        db = 0
    return dV_prev, dW, db

def excitation_activation_backward(dV, cache, activation, apply_bias):
    excitation_cache, activation_cache = cache
    if (activation == RELU):
        dH = relu_backward(dV, activation_cache)
    elif (activation == SIGMOID):
        dH = sigmoid_backward(dV, activation_cache)
    else:
        dH = tanh_backward(dV, activation_cache)
    dV_prev, dW, db = excitation_backward(dH, excitation_cache, apply_bias)
    return dV_prev, dW, db

def model_backward(O, Y, caches, hidden_activation, output_activation, apply_bias):
    gradients = {}
    L = len(caches) # number of layers
    Y = Y.reshape(O.shape)
    dO = O - Y
    '''
    if (output_activation == RELU):
        dO = O - Y  # dO = delta_Error_relu/delta_O
    elif (output_activation == SIGMOID):
        dO = - np.divide(Y, O) + np.divide(1-Y, 1-O)    # dO = delta_Error_sigmoid/delta_O
    else:
        dO = np.divide(O - Y, np.dot((O+1).T,1-O))  # dO = delta_Error_tanh/delta_O
    '''
    current_cache = caches[L-1]
    if (apply_bias):
        gradients['dV'+str(L-1)], gradients['dW'+str(L)], gradients['db'+str(L)] = excitation_activation_backward(dO, current_cache, output_activation, apply_bias)
    else:
        gradients['dV'+str(L-1)], gradients['dW'+str(L)], _ = excitation_activation_backward(dO, current_cache, output_activation, apply_bias)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        if (apply_bias):
            gradients['dV'+str(l)], gradients['dW'+str(l+1)], gradients['db'+str(l+1)] = excitation_activation_backward(gradients['dV'+str(l+1)], current_cache, hidden_activation, apply_bias)
        else:
            gradients['dV'+str(l)], gradients['dW'+str(l+1)], _ = excitation_activation_backward(gradients['dV'+str(l+1)], current_cache, hidden_activation, apply_bias)
    return gradients