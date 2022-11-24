import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from multilayer_utils.Prediction import predict_latent, predict_generated
from matplotlib.pyplot import matshow

from autoencoder import Autoencoder
from inputs.fonts import formatFontList, bitmap
from utils.constants import SIGMOID
from utils.constants import SIGMOID, RELU

charset = ['`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~','DEL']

def plotError(autoencoder):
    x = []
    for i in range(len(autoencoder.getErrors())):
        x.append(i)

    plt.plot(x, autoencoder.getErrors())
    plt.show()

autoencoder = Autoencoder()
nlayers = autoencoder.inputHandler.num_layers
idx = autoencoder.inputHandler.latent_layer + 1
print(f"idx = {idx}")
latent_layer_dim = autoencoder.inputHandler.layer_dims[idx]
print(f"dim = {latent_layer_dim}")
if len(sys.argv)-1 < latent_layer_dim:
    print(f"Wrong quantity of layers. {latent_layer_dim} are needed")
else:
    autoencoder.trainNetwork()
    plotError(autoencoder)
    prediction, _ = predict_latent(autoencoder.getFontMap(), idx, autoencoder.getParameters(), SIGMOID, False)
    prediction = np.array(prediction).T
    x = prediction.T[0]
    y = prediction.T[1]
    fig, ax = plt.subplots()
    plt.scatter(x, y, c='blue')
    new_x = [float(sys.argv[1])]
    new_y = [float(sys.argv[2])]
    plt.plot(new_x, new_y, marker='*', c='red')
    for i, txt in enumerate(charset):
        plt.annotate(txt, (x[i] + 0.01, y[i] + 0.01))
    plt.show()
    new_input = [float(sys.argv[1]), float(sys.argv[2])]
    V = np.array(new_input).T
    prediction, _ = predict_generated(autoencoder.getFontMap(), idx, V, autoencoder.getParameters(), SIGMOID, False)
    prediction = np.array(prediction).T
    output = np.reshape(prediction, (7, 5))
    plt.matshow(output, vmin=0, vmax=1)
    plt.colorbar()
    plt.show()