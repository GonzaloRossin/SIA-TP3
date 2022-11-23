import numpy as np

# Receives an array containing the dimensions of each layer l
# Returns a dictionary containing W_i and b_i for each layer l
def init_parameters(layer_dims, apply_bias):
    #np.random.seed(1)
    parameters = {}
    num_layers = len(layer_dims)
    for l in range(1, num_layers):
        parameters['W'+str(l)] = 2 * np.random.rand(layer_dims[l], layer_dims[l-1]) - 1
        if(apply_bias):
            parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

# GRADIENT DESCENT #########################################################################################
## Reach the local or global minimum of the cost function by taking tiny steps downhill (hopefully).

# Use gradient descent to update the parameters
def update_params_gd(params, gradients, learning_rate, num_layers, apply_bias):
    parameters = params.copy()
    for l in range(1, num_layers):
        parameters['W'+str(l)] = parameters['W'+str(l)] - learning_rate * gradients['dW'+str(l)]
        if (apply_bias):
            parameters['b'+str(l)] = parameters['b'+str(l)] - learning_rate * gradients['db'+str(l)]
    return parameters

# MOMENTUM #################################################################################################
## A ball rolling down the cost function surface.
## Alpha resembles the friction, dW and db the acceleration, then vel[dW] and vel[b] the velocity.

def init_velocity(parameters, num_layers, apply_bias):
    vel = {}
    for l in range(1, num_layers):
        vel['dW' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        if (apply_bias):
            vel['db' + str(l)] = np.zeros(parameters['b' + str(l)].shape)
    return vel

# Use gradient descent with momentum to update the parameters
def update_params_momentum(params, gradients, learning_rate, vel, alpha, num_layers, apply_bias):
    parameters = params.copy()
    for l in range(1, num_layers):
        vel['dW' + str(l)] = alpha * vel['dW' + str(l)] + (1-alpha) * gradients['dW' + str(l)]
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * vel['dW' + str(l)]
        if (apply_bias):
            vel['db' + str(l)] = alpha * vel['db' + str(l)] + (1-alpha) * gradients['db' + str(l)]
            parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * vel['db' + str(l)]
    return parameters, vel

# ADAM #####################################################################################################
## Adaptive Moment Estimation: Combine the Root Mean Square Propagation with Momentum.

def init_adam(parameters, num_layers, apply_bias):
    M = {}  # first moment
    V = {}  # second moment
    for l in range(1, num_layers):
        M['dW' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        V['dW' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        if (apply_bias):
            M['db' + str(l)] = np.zeros(parameters['b' + str(l)].shape)
            V['db' + str(l)] = np.zeros(parameters['b' + str(l)].shape)
    return M, V

def update_params_adam(params, gradients, learning_rate, M, V, t, beta1, beta2, epsilon, num_layers, apply_bias):
    parameters = params.copy()
    M_hat = {}
    V_hat = {}
    for l in range(1, num_layers):
        M['dW' + str(l)] = beta1 * M['dW' + str(l)] + (1-beta1) * gradients['dW' + str(l)]    # 1st moment estimate
        V['dW' + str(l)] = beta2 * V['dW' + str(l)] + (1-beta2) * np.power(gradients['dW' + str(l)], 2)   # 2nd moment estimate
        M_hat['dW' + str(l)] = M['dW' + str(l)] / (1 - np.power(beta1, t))  # bias-corrected 1st moment estimate
        V_hat['dW' + str(l)] = V['dW' + str(l)] / (1 - np.power(beta2, t))  # bias-corrected 2nd moment estimate
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * M_hat['dW' + str(l)] / (np.sqrt(V_hat['dW' + str(l)]) + epsilon)  # update parameters
        if (apply_bias):    # idem for bias
            M['db' + str(l)] = beta1 * M['db' + str(l)] + (1-beta1) * gradients['db' + str(l)]    # 1st moment estimate
            V['db' + str(l)] = beta2 * V['db' + str(l)] + (1-beta2) * np.power(gradients['db' + str(l)], 2)   # 2nd moment estimate
            M_hat['db' + str(l)] = M['db' + str(l)] / (1 - np.power(beta1, t))  # bias-corrected 1st moment estimate
            V_hat['db' + str(l)] = V['db' + str(l)] / (1 - np.power(beta2, t))  # bias-corrected 2nd moment estimate
            parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * M_hat['db' + str(l)] / (np.sqrt(V_hat['db' + str(l)]) + epsilon)  # update parameters
    return parameters, M, V