import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

img_width = x_train.shape[1]
img_height = x_train.shape[2]
num_channels = 1 #greyscale
x_train = x_train.reshape(x_train.shape[0], img_height, img_width, num_channels)
x_test = x_test.reshape(x_test.shape[0], img_height, img_width, num_channels)
input_shape = (img_height, img_width, num_channels)

#view a few figures
plt.figure(1)
plt.subplot(221)
plt.imshow(x_train[42][:, :, 0])
plt.subplot(222)
plt.imshow(x_train[420][:, :, 0])
plt.subplot(223)
plt.imshow(x_train[4200][:, :, 0])
plt.subplot(224)
plt.imshow(x_train[42000][:, :, 0])
plt.show()

latent_dim = 2
input_img = Input(shape=input_shape, name='encoder_input')
x = Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
x = Conv2D(64, 3, padding='same', activation='relu',)(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)