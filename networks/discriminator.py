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

def discriminator(image_shape=(256, 256, 3)):

    # image_shape : specifies the shape of the input image. By default- 256*256
    
    # Discriminator architecture details:
        # Ck denotes a Convulution-BatchNorm-LeakyReLU layer with k filters (Ref 6.1.)
        # Discriminator architecture: C64-C128-C256-C512 (Ref 6.1.2.)
        # Receptive field size for above architecture : 70*70

        # The depth of the discriminator architecture is responsible for the receptive field size

    # Weights initialized from a Gaussian distribution with mean 0 and standard deviation 0.02 (Ref 6.2.)
    curv_val = RandomNormal(stddev=0.02)

    # Discriminator uses two inputs, the source input image and the target input image
    # source image input
    # For maps and cityscapes dataset, input image shape is 256*256
    src_image_inp = Input(shape=image_shape)
    # target image input
    target_image_inp = Input(shape=image_shape)

    # concatenate images channel-wise
    merged_input = Concatenate()([src_image_inp, target_image_inp])

    kernel_size = (4,4)
    stride = (2,2)

    # First layer : C64
    layer = Conv2D( 64, kernel_size, strides = stride, padding ='same', kernel_initializer=curv_val)(merged_input)
    # No batch-normalization for the first layer
    layer = LeakyReLU(alpha=0.2)(layer)

    # Subsequent layers : C128-C256-C512
    filter_size = [128, 256, 512]
    for i in range(len(filter_size)):
        layer = Conv2D( filter_size[i], kernel_size, strides = stride, padding ='same', kernel_initializer=curv_val)(layer)
        layer = BatchNormalization()(layer)
        # Applying Leaky ReLU activation with slope 0.2
        layer = LeakyReLU(alpha=0.2)(layer)
        
    layer = Conv2D(512, (4,4), padding='same', kernel_initializer=curv_val)(layer)
    layer = BatchNormalization()(layer)
    layer= LeakyReLU(alpha=0.2)(layer)

    # After last layer a convolution is applied to map to a 1-dimensional output
    layer = Conv2D(1, (4,4), padding='same', kernel_initializer=curv_val)(layer)
    patch_out = Activation('sigmoid')(layer)

    # define model
    model = Model([src_image_inp, target_image_inp], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

model = discriminator()
print(model.summary())
