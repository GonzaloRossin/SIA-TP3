import matplotlib.pyplot as plt
import numpy as np
from multilayer_utils.Prediction import predict

from EJ1.a.autoencoder import Autoencoder
from inputs.fonts import formatFontList, bitmap
from utils import constants
from utils.constants import SIGMOID, RELU


def plotError(autoencoder):
    x = []
    for i in range(len(autoencoder.getErrors())):
        x.append(i)

    plt.plot(x, autoencoder.getErrors())
    plt.show()


def plotResultComparison(autoencoder):
    testSet = formatFontList(bitmap).T
    prediction = np.array(predict(autoencoder.getFontMap(), autoencoder.getParameters(), False, SIGMOID, RELU)).T

    '''for i in range(len(bitmap)):
        matrix = np.reshape(testSet[i], (7, 5))
        plt.matshow(matrix)
        plt.colorbar()
        plt.show()'''''
    for i in range(len(bitmap)):
        output = np.reshape(prediction[i], (7, 5))
        plt.matshow(output)
        plt.colorbar()
        plt.show()

autoencoder = Autoencoder()
autoencoder.trainNetwork()
plotResultComparison(autoencoder)
