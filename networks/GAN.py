from networks.generator import generator
from networks.discriminator import discriminator
import numpy as np
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

def GAN(g_model, d_model, input_shape=(256, 256, 3)):

    # g_model : input generator model
    # d_model : input discriminator model
    # input_shape : input image shape
    
    # This is a composite model to connect the output of the generator model to the input of the discrimintor model

    ip_image = Input(shape=input_shape)

    # Discriminator trainable attribute set to false to train the generator.
    # Discriminator state remains the same while the generator trains. 
    d_model.trainable = False
    g_output = g_model(ip_image)
    d_output = d_model([ip_image, g_output])

    GAN_model = Model(ip_image, [d_output, g_output])
    # Defining loss function
    # loss = Adversarial_loss + lambda * L1_loss
    loss = ['binary_crossentropy', 'mae']
    # weighting the loss contributions of the different model outputs
    loss_weights = [1, 100]

    # optimizer parameters specified in  3.3.
    optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)
    GAN_model.compile(loss=loss, optimizer=optimizer, loss_weights=loss_weights)

    return GAN_model

g_model = generator()
d_model = discriminator()
GAN_model = GAN(g_model, d_model)
GAN_model.summary()