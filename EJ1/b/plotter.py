import matplotlib.pyplot as plt
import numpy as np
from multilayer_utils.Prediction import predict

from EJ1.b.add_noise import *
from EJ1.b.autoencoder import Autoencoder
from inputs.fonts import formatFontList, bitmap
from utils.constants import SIGMOID
from utils.constants import SIGMOID, RELU


def plotError(autoencoder):
    x = []
    for i in range(len(autoencoder.getErrors())):
        x.append(i)

    plt.plot(x, autoencoder.getErrors())
    plt.show()


def plotResultComparison(encoder):
    #testSet = formatFontList(bitmap).T
    prediction, _ = predict(encoder.getFontMap(), encoder.getParameters(), False, SIGMOID, SIGMOID)
    prediction = np.array(prediction).T

    '''for i in range(len(bitmap)):
        matrix = np.reshape(testSet[i], (7, 5))
        plt.matshow(matrix)
        plt.colorbar()
        plt.show()'''''
    for i in range(32):
        output = np.reshape(prediction[i], (7, 5))
        plt.matshow(output, vmin=0, vmax=1)
        plt.colorbar()
        plt.show()


def plotError(encoder):
    x = []
    for i in range(len(encoder.getErrors())):
        x.append(i)
    plt.plot(x, encoder.getErrors())
    plt.show()


autoencoder = Autoencoder()
#autoencoder.fontMap = binary_noise(0.1, autoencoder.fontMap)
autoencoder.fontMap = distribution_noise(0.1, autoencoder.fontMap)
autoencoder.trainNetwork()
plotError(autoencoder)
plotResultComparison(autoencoder)
