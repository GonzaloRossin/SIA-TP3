from VAE import VAE
dummyVae = VAE(epochs=10000, image_width=16, image_height=16)
dummyVae.setUpVae()
dummyVae.plotResults()
