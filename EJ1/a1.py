import matplotlib.pyplot as plt
import numpy as np
from multilayer_utils.Prediction import predict
from matplotlib.pyplot import matshow

from autoencoder import Autoencoder
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
    
    #'''
    
    plt.figure()
    plt.axis('off')
    for i in range(1,33) :
        plt.subplot(6, 6, i)
        output = np.reshape(prediction[i-1], (7, 5))
        matshow(output, vmin=0, vmax=1, fignum=False)
        i += 1
    
    #plt.colorbar()
    plt.show()
    #'''


def plotError(encoder):
    x = []
    for i in range(len(encoder.getErrors())):
        x.append(i)
    plt.plot(x, encoder.getErrors())
    plt.show()


autoencoder = Autoencoder()
autoencoder.trainNetwork()
plotError(autoencoder)
plotResultComparison(autoencoder)
