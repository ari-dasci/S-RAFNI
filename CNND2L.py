
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense
from tensorflow.keras.regularizers import l2
import numpy as np

# def eight_layer(input_tensor = None, input_shape = None, num_classes = 10):
def eight_layer(num_classes):
    # if input_tensor is None:
    #     img_input = tf.keras.Input(shape = input_shape)
    # else:
    #     if not K.is_keras_tensor(input_shape):
    #         img_input = tf.keras.Input(tensor = input_tensor, shape = input_shape)
    #     else:
    #         img_input = input_tensor
    img_input = tf.keras.Input(shape = (32, 32, 3))

    # Block 1
    x = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal',
                name = 'block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding = 'same', kernel_initializer = 'he_normal',
                name = 'block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal',
                name = 'block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding = 'same', kernel_initializer = 'he_normal',
                name = 'block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block2_pool')(x)

    # Block 3
    x = Conv2D(196, (3, 3), padding = 'same', kernel_initializer = 'he_normal',
                name = 'block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(196, (3, 3), padding = 'same', kernel_initializer = 'he_normal',
                name = 'block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block3_pool')(x)

    x = Flatten(name = 'flatten')(x)

    x = Dense(256, kernel_initializer = 'he_normal', kernel_regularizer = l2(0.01),
                bias_regularizer = l2(0.01), name = 'fc1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name = 'lid')(x)

    x = Dense(num_classes, kernel_initializer = 'he_normal')(x)
    x = Activation('softmax')(x)

    return tf.keras.Model(inputs = img_input, outputs = x)
