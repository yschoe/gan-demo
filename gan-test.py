'''

  Mostly done by ChatGPT

'''
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import keras
import matplotlib.pyplot as plt
from gan import generator, discriminator, generate_noise, latent_dim
# from gan import discriminator 
# from gan import generate_noise

# Load the models

generator = keras.models.load_model('generator.h5')
discriminator = keras.models.load_model('discriminator.h5')

# Generate a new image from a random noise vector
noise = generate_noise(1, latent_dim)
generated_image = generator(noise, training=False)

# Denormalize the pixel values
generated_image = (generated_image + 1) / 2.0

# Plot the generated image
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()
