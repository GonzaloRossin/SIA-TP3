import numpy as np
from utils.constants import *
from multilayer_utils.Normalization import normalize

def random_layer_dims(feature_num, output_num, num_hidden_layers, max_dim):
    layer_dims = []
    layer_dims.append(feature_num)
    for i in range(num_hidden_layers):
        layer_dims.append(np.random.random_integers(2, max_dim))
    layer_dims.append(output_num)
    return layer_dims

def read_input(input_filepath, num_features):
    X = []
    Y = []
    with open(input_filepath) as input_file:
        input_file.readline()   # skip header
        lines = input_file.readlines()
        for line in lines:
            XY = line.split(",")
            XY = list(map(lambda z: int(z), XY))
            X.append(XY[:num_features])
            Y.append(XY[num_features:])
    return X, Y

class InputHandler:
    
    def __init__(self, input, ratio=-1):

        self.apply_bias = (input['apply_bias']==1)
        self.num_layers = input['hidden_layers']['num_layers']
        self.learning_rate = input['learning_rate']

        self.normalize = (input['normalize']==1)
        self.hidden_activation = input['hidden_activation']
        self.output_activation = input['output_activation']

        num_features = input['num_features']
        num_outputs = input['num_outputs']

        X, Y = read_input(input['input_file'], num_features)
        if (self.normalize):
            Y, self.min_y, self.max_y = normalize(Y, self.output_activation)

        if (ratio < 0):
            self.ratio = input['training_set_ratio']
        else:
            self.ratio = ratio

        training_idx = len(X) * self.ratio // 100
        train_X = X[:training_idx]
        train_Y = Y[:training_idx]
        test_X = X[training_idx:]
        test_Y = Y[training_idx:]

        self.training_set_X = np.array(train_X).T
        self.test_set_X = np.array(test_X).T
        self.training_set_Y = np.array(train_Y).T
        self.test_set_Y = np.array(test_Y).T
        
        if (input['hidden_layers']['use_num']):
            self.layer_dims = random_layer_dims(num_features, num_outputs, input['hidden_layers']['num_layers'], input['hidden_layers']['max_dim'])
        else:
            self.layer_dims = [num_features] + input['hidden_layers']['layer_dims'] + [num_outputs]
        #print(f"\nLayers dim = {self.layer_dims}\n")

        self.num_epochs = input['num_epochs']
        
        if (input['use_mini_batches']==1):
            self.batch_size = min(input['mini_batch_size'], self.training_set_X.shape[1])
        else:
            self.batch_size = self.training_set_X.shape[1]

        self.optimizer = input['optimizer']['method']
        #if (self.optimizer == MOMENTUM):
        self.momentum_alpha = input['optimizer']['momentum_alpha']
        #elif (self.optimizer == ADAM):
        self.beta1 = input['optimizer']['adam']['beta1']
        self.beta2 = input['optimizer']['adam']['beta2']
        self.epsilon = input['optimizer']['adam']['epsilon']
        
        if (input['adaptive_etha']['after'] > 0):
            self.use_adaptive_etha = True
            self.adaptive_etha = {}
            self.adaptive_etha['after'] = input['adaptive_etha']['after']
            self.adaptive_etha['a'] = input['adaptive_etha']['a']
            self.adaptive_etha['b'] = input['adaptive_etha']['b']
        else:
            self.use_adaptive_etha = False