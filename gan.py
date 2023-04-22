'''

Created (mostly) by ChatGPT


Prompt: Write python code for GAN, using the mnist data set. Include code to show generated image, based on a white-noise image.

'''

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

'''

  CONFIG 

'''

# Set the dimensionality of the noise vector
latent_dim = 100

# Set the batch size and number of epochs
batch_size = 128
epochs = 100

'''

  Load data : check if local file exists first

'''

if os.path.exists("mnist_data.npz"):
  # Load local file
  print("\n\n*** Local MNIST file exists. Using it")
  data = np.load('mnist_data.npz')
  x_train = data['x_train']
  y_train = data['y_train']
  x_test = data['x_test']
  y_test = data['y_test']
else:
  # Load the MNIST dataset
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  # save
  print("\n\n*** Saving to Local MNIST file")
  np.savez('mnist_data.npz', x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

# Normalize the pixel values between -1 and 1
x_train = x_train / 127.5 - 1
x_train = np.expand_dims(x_train, axis=3)

# Build the generator
generator = tf.keras.Sequential()
generator.add(layers.Dense(7*7*256, use_bias=False, input_shape=(latent_dim,)))
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU())

generator.add(layers.Reshape((7, 7, 256)))
generator.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU())

generator.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU())

generator.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

# Build the discriminator
discriminator = tf.keras.Sequential()
discriminator.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
discriminator.add(layers.LeakyReLU())
discriminator.add(layers.Dropout(0.3))

discriminator.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
discriminator.add(layers.LeakyReLU())
discriminator.add(layers.Dropout(0.3))

discriminator.add(layers.Flatten())
discriminator.add(layers.Dense(1))

# Define the loss functions for the generator and discriminator
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Define the optimizers for the generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Generate random noise as input for the generator
def generate_noise(batch_size, latent_dim):
    return tf.random.normal([batch_size, latent_dim])

# Define the training loop
@tf.function
def train_step(images):
    noise = generate_noise(batch_size, latent_dim)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Train the GAN
def train(dataset, epochs):

    for epoch in range(epochs):

        print("epoch ",epoch)

        for image_batch in dataset:
            train_step(image_batch)

        # Generate images from the generator
        if epoch % 5 == 0:
            noise = generate_noise(1, latent_dim)
            generated_image = generator(noise, training=False)

            # Denormalize the pixel values
            generated_image = (generated_image + 1) / 2.0

            # Plot the generated image
            f = plt.figure()
            plt.imshow(generated_image[0, :, :, 0], cmap='gray')
            plt.savefig(f'gan-{epoch :04}.pdf')
            print(f'  saved gan-{epoch :04}.pdf')

'''
original code by ChatGPT: This part has been moved to gan-train.py

# Create a TensorFlow dataset object
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size)

# Train the GAN
train(dataset, epochs)

# Save the models
generator.save('generator.h5')
discriminator.save('discriminator.h5')
'''
