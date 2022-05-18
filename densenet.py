import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D, Dropout
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.regularizers import l2
import numpy as np

decay = 0.0005

def conv(input, filters, stride):
    return Conv2D(filters, (3, 3), stride, padding = 'same', use_bias = False,
                    kernel_initializer = tf.keras.initializers.RandomNormal(
                        stddev = np.sqrt(2.0 / 9 / filters)),
                    kernel_regularizer = l2(decay))(input)

def add_layer(input, growthRate):
    x = BatchNormalization(momentum = 0.9, epsilon = 1e-05)(input)
    #x = BatchNormalization(axis = 3)(input)
    x = Activation('relu')(x)
    x = conv(x, growthRate, (1, 1))
    x = Dropout(rate = 0.2)(x)
    return tf.concat([x, input], 3)

def add_transition(input):
    filters = input.shape[3]
    x = BatchNormalization(momentum = 0.9, epsilon = 1e-05)(input)
    #x = BatchNormalization(axis = 3)(input)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), strides = (1, 1), use_bias = False, padding = 'same',
                kernel_regularizer = l2(decay))(x)
    x = Activation('relu')(x)
    x = Dropout(rate = 0.2)(x)
    x = AveragePooling2D((2, 2), strides = (2, 2))(x)
    return x

def densenet(depth, growthRate, num_classes):
    N = int((depth-4)/3)

    input = tf.keras.Input(shape = (32, 32, 3))
    x = conv(input, 16, (1, 1))

    # Block 1
    for i in range(N):
        x = add_layer(x, growthRate)
    x = add_transition(x)

    # Block 2
    for i in range(N):
        x = add_layer(x, growthRate)
    x = add_transition(x)

    # Block 3
    for i in range(N):
        x = add_layer(x, growthRate)

    x = BatchNormalization(momentum = 0.9, epsilon = 1e-05)(x)
    #x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_classes, activation = 'softmax',
            kernel_initializer = tf.keras.initializers.VarianceScaling(2.0),
            kernel_regularizer = l2(decay), bias_regularizer = l2(0))(x)

    return tf.keras.Model(inputs = input, outputs = output)
