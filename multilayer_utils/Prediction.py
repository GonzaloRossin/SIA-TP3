import numpy as np
from utils.constants import *
from multilayer_utils.Propagation import model_forward

def logistic_prediction(X, trained_parameters, hidden_activation, apply_bias):
    O, _ = model_forward(X, trained_parameters, apply_bias, hidden_activation, SIGMOID)
    P = 1 * (O > 0.5)
    return O, P

def linear_prediction(X, trained_parameters, hidden_activation, apply_bias):
    O, _ = model_forward(X, trained_parameters, apply_bias, hidden_activation, RELU)
    return O, _

def tanh_prediction(X, trained_parameters, hidden_activation, apply_bias):
    O, _ = model_forward(X, trained_parameters, apply_bias, hidden_activation, TANH)
    P = 2 * (O > 0.5) - 1
    return O, P

def predict(X, trained_parameters, apply_bias, hidden_activation, output_activation):
    if (output_activation == SIGMOID):
        return logistic_prediction(X, trained_parameters, hidden_activation, apply_bias)
    elif (output_activation == RELU):
        return linear_prediction(X, trained_parameters, hidden_activation, apply_bias)
    return tanh_prediction(X, trained_parameters, hidden_activation, apply_bias)

def predict_decision_boundary(X, trained_parameters, apply_bias, hidden_activation):
    O, _ = model_forward(X, trained_parameters, apply_bias, hidden_activation, SIGMOID)
    P = (O > 0.5)
    return P

def predict_multiclass(X, trained_parameters, hidden_activation, output_activation, apply_bias):
    O, _ = model_forward(X, trained_parameters, apply_bias, hidden_activation, output_activation)
    P = []
    for output in O:
        max_val = np.max(output)
        P.append(1 * (output == max_val))
    return O, P