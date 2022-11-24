import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics

from PIL import Image
import os

from tensorflow.python.framework.ops import disable_eager_execution


class VAE:
    def __init__(self, epochs, image_width, image_height):
        disable_eager_execution()
        self.image_folder = 'images'
        self.image_shape = (image_width, image_height)
        self.channels = 3
        self.original_dim = image_width * image_height * self.channels
        self.latent_dim = 2
        self.intermediate_dim = 256
        self.batch_size = 10
        self.epochs = epochs
        self.epsilon_std = 1.0
        self.encoder = None
        self.decoder = None
        self.input = None
        self.z_mean = None
        self.z_log_var = None

    def setEncoder(self):
        self.input = Input(shape=(self.original_dim,), name="input")
        # intermediate layer
        h = Dense(self.intermediate_dim, activation='relu', name="encoding")(self.input)
        # defining the mean of the latent space
        self.z_mean = Dense(self.latent_dim, name="mean")(h)
        # defining the log variance of the latent space
        self.z_log_var = Dense(self.latent_dim, name="log-variance")(h)
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])
        # defining the encoder as a keras model
        encoder = Model(self.input, [self.z_mean, self.z_log_var, z], name="encoder")
        # print out summary of what we just did
        encoder.summary()
        self.encoder = encoder

    def setDecoder(self):
        # Input to the decoder
        input_decoder = Input(shape=(self.latent_dim,), name="decoder_input")
        # taking the latent space to intermediate dimension
        decoder_h = Dense(self.intermediate_dim, activation='relu', name="decoder_h")(input_decoder)
        # getting the mean from the original dimension
        x_decoded = Dense(self.original_dim, activation='sigmoid', name="flat_decoded")(decoder_h)
        # defining the decoder as a keras model
        decoder = Model(input_decoder, x_decoded, name="decoder")
        decoder.summary()
        self.decoder = decoder

    def sampling(self, args: tuple):
        z_mean, z_log_var = args
        print(z_mean)
        print(z_log_var)
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0.,
                                  stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def getOutput(self):
        output_combined = self.decoder(self.encoder(self.input)[2])
        # link the input and the overall output
        vae = Model(self.input, output_combined)
        vae.summary()
        self.vae = vae

    def vae_loss(self, x: tf.Tensor, x_decoded_mean: tf.Tensor):
        # Aca se computa la cross entropy entre los "labels" x que son los valores 0/1 de los pixeles,
        # y lo que salió al final del Decoder.
        xent_loss = self.original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)
        return vae_loss


    def setUpVae(self):
        self.setEncoder()
        self.setDecoder()
        self.getOutput()
        self.vae.compile(loss=self.vae_loss, experimental_run_tf_function=False)
        self.vae.summary()
        if not os.path.exists(self.image_folder) or not os.path.isdir(self.image_folder):
            print(f'ERROR: Missing images folder')

        images = [file for file in os.listdir(self.image_folder) if file.endswith(('jpeg', 'png', 'jpg'))]

        x = []

        processed = 0
        for image in images:
            full_path = os.path.join(self.image_folder, image)

            img = Image.open(full_path)

            if len(img.split()) >= 3:
                img = img.convert("RGB")
                img = img.resize(self.image_shape)
                img = np.asarray(img, dtype=np.float32) / 255
                img = img[:, :, :3]

                x.append(img)

                processed += 1

        print(f"Loaded {processed} out of {len(images)} images with shape {self.image_shape}")

        x = np.array(x)

        (x_train, y_train) = (x, x)

        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

        self.vae.fit(x_train, x_train, shuffle=True, epochs=self.epochs, batch_size=self.batch_size)
        
    def plotResults(self):
        n = 15  # figure with 15x15 digits
        digit_size = self.image_shape[0]
        figure = np.zeros((self.image_shape[0] * n, self.image_shape[1] * n, self.channels))
        # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
        # to produce values of the latent variables z, since the prior of the latent space is Gaussian
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(self.image_shape[0], self.image_shape[1], self.channels)
                figure[i * self.image_shape[0]: (i + 1) * self.image_shape[0],
                j * self.image_shape[1]: (j + 1) * self.image_shape[1]] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.show()
