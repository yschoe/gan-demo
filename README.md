# gan-demo

A simple demo for Generative Adversarial Network, using the MNIST data set

The code was mostly generated using ChatGPT. I separated the code into the gan module and train and test scripts. 

### Training

1. In `gan.py` edit these parameters
	```
	# Set the dimensionality of the noise vector
	latent_dim = 100

	# Set the batch size and number of epochs
	batch_size = 128
	epochs = 100
	```
2. Run `python gan-train.py`
   - This will result in multiple `gan-XXXX.pdf` files, which are intermediate results.
   - The weight files are saved to `generator.h5` and `discriminator.h5`.
   - Downloaded MNIST data will be stored locally for reuse `mnist_data.mpz`.

### Testing

1. After training, simply run `python gan-test.py`
   - This will load the saved weight files `generator.h5` and `discriminator.h5`.

### Testing with pre-trained weight files

1. Copy `saved/generator.h5` and `saved/discriminator.h5` to the current directory, and run
`pytho gan-test.py`


