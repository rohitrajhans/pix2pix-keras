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

def encoder(ip_layer, n_filters, batchNorm=True):
    
    # batchNorm : boolean value that determines whether to apply batch normalization
    # ip_layer : input feature vector
    # n_filters : number of filters after concatenation

    # Encoder architecture : C64-C128-C256-C512-C512-C512-C512-C512
    # batchNorm is only applied to first layer of the encoder i.e. C64

    # Weights initialized from a Gaussian distribution with mean 0 and standard deviation 0.02
    initialized_weights = RandomNormal(stddev=0.02)

    # All layers have kernel size 4*4 and stride 2*2
    kernel_size = (4, 4)
    stride = (2, 2)

    layer = Conv2D(n_filters, kernel_size, strides=stride, padding='same', kernel_initializer=initialized_weights)(ip_layer)

    if batchNorm:
        layer = BatchNormalization()(layer, training=True)

    layer = LeakyReLU(alpha=0.2)(layer)

    return layer

def decoder(ip_layer, skip_connection, n_filters, dropout=True):

    # ip_layer : input feature vector
    # n_filters : number of filters after concatenation
    # dropout : boolean value that determines whether to apply dropout to the layer
    # skip_connection : the encoder layer from which skip connections will be applied

    # Decoder architecture : C512-C512-C512-C512-C256-C128-C64

    # Weights initialized from a Gaussian distribution with mean 0 and standard deviation 0.02
    initialized_weights = RandomNormal(stddev=0.02)

    # All layers have kernel size 4*4 and stride 2*2
    kernel_size = (4, 4)
    stride = (2, 2)

    layer = Conv2DTranspose(n_filters, kernel_size, strides=stride, padding='same', kernel_initializer=initialized_weights)(ip_layer)
    layer = BatchNormalization()(layer, training=True)
    if dropout:
        layer = Dropout(0.5)(layer, training=True)

    # Applying the skip connection
    layer = Concatenate()([layer, skip_connection])
    # ReLUs in the decoder are not leaky - Ref. 6.1.1.
    layer = Activation('relu')(layer)

    return layer

def generator(image_size = (256, 256, 3)):

    # image_size : shape of input image

    # Generator Architecture : CCD512-CD1024-CD1024-CD1024-CD1024-CD512-CD256-CD128
    # U-Net architecture with skip connections between each layer i in the encoder and the layer n-i in the decoder

    # Weights initialized from a Gaussian distribution with mean 0 and standard deviation 0.02
    kernel_init = RandomNormal(stddev = 0.02)

    ip_image = Input(shape = image_size)
    
    # All layers have kernel size 4*4 and stride 2*2
    kernel_size = (4,4)
    stride = (2,2)

    e1 = encoder(ip_image, 64, batchNorm=False) # No batchNorm for first encoder layer
    e2 = encoder(e1, 128)
    e3 = encoder(e2, 256)
    e4 = encoder(e3, 512)
    e5 = encoder(e4, 512)
    e6 = encoder(e5, 512)
    e7 = encoder(e6, 512)

    # Bottleneck layer, connecting encoder and decoder
    bottle_neck = Conv2D(512, kernel_size, strides=stride, padding='same', kernel_initializer=kernel_init, activation='relu')(e7)

    d1 = decoder(bottle_neck, e7, 512)
    d2 = decoder(d1, e6, 512)
    d3 = decoder(d2, e5, 512)
    d4 = decoder(d3, e4, 512, dropout=False)
    d5 = decoder(d4, e3, 256, dropout=False)
    d6 = decoder(d5, e2, 128, dropout=False)
    d7 = decoder(d6, e1, 64, dropout=False)

    # Convolution is applied to map to the number of output channels, followed by a tanh function - Ref. 6.1.1.
    op_image = Conv2DTranspose(3, kernel_size, strides=stride, padding='same', kernel_initializer=kernel_init, activation='tanh')(d7)

    # compile model
    model = Model(ip_image, op_image)

    return model

model = generator()
model.summary()

