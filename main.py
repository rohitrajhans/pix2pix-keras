import numpy as np
import cv2
import keras
import tensorflow as tf
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.models import load_model
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import ReLU
from keras import Sequential
from matplotlib import pyplot as plt
from numpy import vstack

from networks.generator import generator
from networks.discriminator import discriminator
from networks.GAN import GAN

def generate_random_training_samples(data, n_samples, n_patch):

    # data : input dataset in .npz format
    # n_samples : number of samples required
    # n_patch : output feature map size (16*16 in our case)

    # this function generates a batch of random samples and returns source images and target

    train_A, train_B = data
    n = randint(0, train_A.shape[0], n_samples)
    X1, X2 = train_A[n], train_B[n]

    # generate the target array of ones
    y = ones((n_samples, n_patch, n_patch, 1))

    return [X1, X2], y

def generate_fake_samples(generator_model, samples, n_patch):
   
    # generator_model : input the generator model
    # input sample for prediction
    # n_patch : output feature map size (16*16 in our case)
    
    #  generates a batch of fake images through the generator model and the associated target

    print(samples.shape)
    X = generator_model.predict(samples)

    # generate the target array of zeros
    y = zeros((len(X), n_patch, n_patch, 1))
    return X, y

def load_real_samples(filename):
	
    # filename : input .npz filename

    # function loads and preprocesses image array before training
    
    # load compressed numpy arrays (.npz)
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	
    # scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5

	return [X1, X2]

def train(discriminator_model, generator_model, gan_model, data, model_dest, n_epochs=200, n_batch=1, n_patch=16, random_jitter=False, current_step=0):

    # discriminator_model : input discriminator model
    # generator_model : input generator model
    # gan_model : input composite gan model
    # data : input dataset as an array of images
    # model_dest : destination for saving model
    # n_epochs : number of epochs
    # n_batch : batch size
    # n_patch : output feature map size
    # random_jitter : boolean value that determines whether to apply random jitter to an image before training
    # current_step : in case of resuming training from a checkpoint, current_step indicates the point from where to restart the training

    train_A, train_B = data

    # calculating total number of steps required in training
    batches_per_epoch = int((len(train_A)) / n_batch)
    n_steps = batches_per_epoch*n_epochs

    print(n_steps, batches_per_epoch)

    # Looping over all the steps
    for i in range(current_step, n_steps):

        # Get a batch of real images
        [X_real_A, X_real_B], y_real = generate_random_training_samples(data, n_batch, n_patch)

        # Adding random jitter
        if random_jitter==True:

            # Upsample input images from 256*256 to 286*286
            input_image = tf.image.resize(X_real_A, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            real_image = tf.image.resize(X_real_B, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            stacked_image = tf.stack([input_image, real_image], axis=0)

            # Randomly crop the images back to 256*256
            cropped_image = tf.image.random_crop( stacked_image, size=[2, 1, 256, 256, 3])

            X_real_A, X_real_B = cropped_image[0], cropped_image[1]
            
            # convert from tensor to numpy
            X_real_A = keras.backend.eval(X_real_A)
            X_real_B = keras.backend.eval(X_real_B)

        # Generate a batch of fake images
        X_fake, y_fake = generate_fake_samples(generator_model, X_real_A, n_patch)

        # Calculate the discriminator losses
        discriminator_loss_real = discriminator_model.train_on_batch([X_real_A, X_real_B], y_real)
        discriminator_loss_generated = discriminator_model.train_on_batch([X_real_A, X_fake], y_fake)

        # Calculate the generator loss
        generator_loss, a, b = gan_model.train_on_batch(X_real_A, [y_real, X_real_B])

        print('%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, discriminator_loss_real, discriminator_loss_generated, generator_loss))

        # Save model state every 10 epochs
        if (i+1) % (batches_per_epoch * 10) == 0:
            save_model(i, generator_model, discriminator_model, gan_model, model_dest)

def save_model(step, g_model, d_model, gan_model, model_dest):

    # step : step at which model is being saved
    # g_model, d_model, gan_model : models
    # model_dest : destination to save the models

    # function saves the models at the given step for further training later

	filename1 = model_dest + ('model_g_%06d.h5' % (step+1))
	g_model.save(filename1)
 
	filename2 = model_dest + ('model_d_%06d.h5' % (step+1))
	d_model.save(filename2)
 
	filename3 = model_dest + ('model_gan_%06d.h5' % (step+1))
	gan_model.save(filename3)
	print('Models successfully saved at step: %d' % (step))

def start_training(dataset_url, model_dest):

    # dataset_url : path to compressed dataset
    # model_dest : destination path to save models

    # this function loads the dataset and starts the training
    
    train_generator = load_real_samples(dataset_url)
    print('Dataset Loaded', train_generator[0].shape, train_generator[1].shape)
    # define input shape based on the loaded dataset
    image_shape = train_generator[0].shape[1:]
    # define the models
    d_model = discriminator(image_shape)
    g_model = generator(image_shape)
    # define the composite GAN model
    gan_model = GAN(g_model, d_model, image_shape)
    # train model
    train(d_model, g_model, gan_model, train_generator, model_dest) 

def resume_training(step, dataset_url, d_model_src, g_model_src, gan_model_src, model_dest):

    # step : step from which training has to be resumed
    # d_model_src, g_model_src, gan_model_src : path where models have been saved

    # this function resumes the training from the mentioned step and the already saved models

    d_model = load_model(d_model_src)
    g_model = load_model(g_model_src)
    gan_model = load_model(gan_model_src)

    dataset = load_real_samples(dataset_url)
    train(d_model, g_model, gan_model, dataset, model_dest, current_step=step)

# Call to start training
dataset_url = './assets/datasets/maps/compressed/maps_256.npz'
model_dest = './models/maps/'
start_training(dataset_url, model_dest)

# Call to resume training from step 153400 ( 140 epochs on the maps dataset)
# step = 153440
# d_model_src = model_dest + 'model_d_' + str(step) + '.h5'
# g_model_src = model_dest + 'model_g_' + str(step) + '.h5'
# gan_model_src = model_dest + 'model_gan_' + str(step) + '.h5'
# resume_training(step, dataset_url, d_model_src, g_model_src, gan_model_src, model_dest)