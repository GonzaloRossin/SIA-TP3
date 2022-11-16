import json
from multilayer_utils.MultilayerPerceptron import multilayer_perceptron
from inputs.fonts import *
from utils.InputHandler import InputHandler


class Autoencoder:
    def __init__(self):
        with open('config.json', 'r') as f:
            nnConfig = json.load(f)
            inputHandler = InputHandler(nnConfig, -1)
        fontMap = formatFontList(bitmap)
        self.neuralNetwork = multilayer_perceptron(fontMap, fontMap, inputHandler)


Autoencoder()