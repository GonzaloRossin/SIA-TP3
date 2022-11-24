import matplotlib.pyplot as plt
import numpy as np
from multilayer_utils.Prediction import predict_latent
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


def plotResultComparison(encoder: Autoencoder):
    nlayers = encoder.inputHandler.num_layers
    idx = nlayers // 2 + 2
    prediction, _ = predict_latent(encoder.getFontMap(), idx, encoder.getParameters(), SIGMOID, False)
    prediction = np.array(prediction).T
    x = prediction.T[0]
    y = prediction.T[1]
    plt.scatter(x, y)
    for i, txt in enumerate(charset):
        plt.annotate(txt, (x[i] + 0.01, y[i] + 0.01))
    plt.show()

autoencoder = Autoencoder()
autoencoder.trainNetwork()
plotResultComparison(autoencoder)
'''
plotError(autoencoder)
'''
