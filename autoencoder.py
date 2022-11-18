import json
from multilayer_utils.MultilayerPerceptron import multilayer_perceptron
from inputs.fonts import *
from utils.InputHandler import InputHandler
import matplotlib.pyplot as plt

class Autoencoder:
    def __init__(self):
        with open('config.json', 'r') as f:
            nnConfig = json.load(f)
            self.inputHandler = InputHandler(nnConfig, -1)
        self.fontMap = formatFontList(bitmap)
        self.parameters = None
        self.errors = None
        self.latent_code = None
    def trainNetwork(self):
        self.parameters, self.errors, self.latent_code = multilayer_perceptron(self.fontMap, self.fontMap
                                                                               , self.inputHandler)
    def getParameters(self):
        return self.parameters
    def getErrors(self):
        return self.errors
    def getLatentCode(self):
        return self.latent_code


autoencoder = Autoencoder()
autoencoder.trainNetwork()
x = []
for i in range(len(autoencoder.getErrors())):
    x.append(i)

plt.plot(x, autoencoder.getErrors())
plt.show()