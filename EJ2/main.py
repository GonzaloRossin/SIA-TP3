from VAE import VAE
dummyVae = VAE(epochs=5000, image_width=16, image_height=16)
dummyVae.setUpVae()
dummyVae.plotResults()
