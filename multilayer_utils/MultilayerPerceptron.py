import numpy as np
from utils.constants import *
import matplotlib.pyplot as plt
from utils.InputHandler import InputHandler
from multilayer_utils.Errors import compute_error
from multilayer_utils.PerceptronParameters import *
from multilayer_utils.Propagation import model_forward, model_backward

def random_mini_batches(X, Y, mini_batch_size, seed):
    #np.random.seed(seed)
    num_examples = X.shape[1]  # number of training examples
    c = Y.shape[0]  # number of classes
    mini_batches = []
    shuffle_idxs = list(np.random.permutation(num_examples))
    shuffled_X = X[:, shuffle_idxs]
    shuffled_Y = Y[:, shuffle_idxs].reshape((c, num_examples))
    sample_size = mini_batch_size
    complete_minibatches = num_examples // mini_batch_size
    for k in range(0, complete_minibatches):
        i = k * mini_batch_size
        j = (k+1) * mini_batch_size
        mini_batch_X = shuffled_X[:,i:j]
        mini_batch_Y = shuffled_Y[:,i:j]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if num_examples % mini_batch_size != 0:    # last minibatch is smaller than the desired size
        mini_batch_X = shuffled_X[:, sample_size*mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, sample_size*mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def multilayer_perceptron(X, Y, input_handler: InputHandler):
    
    errors = []
    parameters = init_parameters(input_handler.layer_dims, input_handler.apply_bias)    
    
    num_layers = len(input_handler.layer_dims)
    if (input_handler.optimizer == MOMENTUM):
        vel = init_velocity(parameters, num_layers, input_handler.apply_bias)
    elif (input_handler.optimizer == ADAM):
        M, V = init_adam(parameters, num_layers, input_handler.apply_bias)

    t = 0
    for epoch in range(1, input_handler.num_epochs+1):
        if (epoch % 1000 == 0):
            print(f"Epoch #{epoch}")
        minibatches = random_mini_batches(X, Y, input_handler.batch_size, epoch)
        total_error = []
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            O, caches = model_forward(minibatch_X, parameters, input_handler.apply_bias, input_handler.hidden_activation, input_handler.output_activation)
            total_error.append(compute_error(O, minibatch_Y, input_handler.output_activation))
            gradients = model_backward(O, minibatch_Y, caches, input_handler.hidden_activation, input_handler.output_activation, input_handler.apply_bias)
            if (input_handler.optimizer == MOMENTUM):
                parameters, vel = update_params_momentum(parameters, gradients, input_handler.learning_rate, vel, input_handler.momentum_alpha, num_layers, input_handler.apply_bias)
            elif (input_handler.optimizer == ADAM):
                t = t + 1
                parameters, M, V = update_params_adam(parameters, gradients, input_handler.learning_rate, M, V, t, input_handler.beta1, input_handler.beta2, input_handler.epsilon, num_layers, input_handler.apply_bias)
            else:
                parameters = update_params_gd(parameters, gradients, input_handler.learning_rate, num_layers, input_handler.apply_bias)
        errors.append(np.mean(total_error))
        if (input_handler.use_adaptive_etha and epoch >= input_handler.adaptive_etha['after']):
            n = input_handler.adaptive_etha['after']
            delta_e = np.mean(errors[-n:])
            if (delta_e < 0):
                delta_etha = input_handler.adaptive_etha['a']
            elif (delta_e > 0):
                delta_etha = - input_handler.adaptive_etha['b'] * input_handler.learning_rate
            else:
                delta_etha = 0
            input_handler.learning_rate += delta_etha
    return parameters, errors