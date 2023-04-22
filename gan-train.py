'''

Created (mostly) by ChatGPT


Prompt: Write python code for GAN, using the mnist data set. Include code to show generated image, based on a white-noise image.

'''

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import gan

# Set the dimensionality of the noise vector
latent_dim = 100

# Set the batch size and number of epochs
batch_size = 128
epochs = 50

# Create a TensorFlow dataset object
#
# - "import gan" above loads data to gan.x_train
#
print("\n\n***  loading data")
dataset = tf.data.Dataset.from_tensor_slices(gan.x_train).shuffle(60000).batch(batch_size)

# Train the GAN
print("\n\n*** start training")
gan.train(dataset, epochs)

# Save the models
print("\n\n*** saving and exiting")
gan.generator.save('generator.h5')
gan.discriminator.save('discriminator.h5')

#print("Press any key to exit")
#input()
