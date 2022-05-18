import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, add
from tensorflow.keras.regularizers import l2
import numpy as np
import sys

sys.setrecursionlimit(10000)

BN_AXIS = 3

def residual_unit(filters, decay, more_filters = False, first = False):
    def f(input):

        if more_filters and not first:
            stride = 2
        else:
            stride = 1

        if not first:
            b = BatchNormalization(axis = BN_AXIS)(input)
            b = Activation('relu')(b)
        else:
            b = input

        b = Conv2D(filters = filters, kernel_size = (3, 3),
                    strides = (stride, stride), kernel_initializer = 'he_normal',
                    padding = 'same', kernel_regularizer = l2(decay),
                    bias_regularizer = l2(0))(b)
        b = BatchNormalization(axis = BN_AXIS)(b)
        b = Activation('relu')(b)
        res = Conv2D(filters = filters, kernel_size = (3, 3),
                    kernel_initializer = 'he_normal', padding = 'same',
                    kernel_regularizer = l2(decay), bias_regularizer = l2(0))(b)

        # check and match number of filters for the shorcut
        input_shape = K.int_shape(input)
        residual_shape = K.int_shape(res)
        if not input_shape[3] == residual_shape[3]:

            stride_width = int(round(input_shape[1] / residual_shape[1]))
            stride_height = int(round(input_shape[2] / residual_shape[2]))

            input = Conv2D(filters = residual_shape[3], kernel_size = (1,1),
                            strides = (stride_width, stride_height),
                            kernel_initializer = 'he_normal', padding = 'valid',
                            kernel_regularizer = l2(decay))(input)

        return add([input, res])

    return f

def resnet_preact(depth, decay, num_classes):
    # This creates 2 + 6 * depth layers
    input = tf.keras.Input(shape = (32, 32, 3))

    # 1 conv + BN + relu
    filters = 16
    b = Conv2D(filters = filters, kernel_size = (3, 3),
                kernel_initializer = 'he_normal', padding = 'same',
                kernel_regularizer = l2(decay), bias_regularizer = l2(0))(input)
    b = BatchNormalization(axis = BN_AXIS)(b)
    b = Activation('relu')(b)

    # 1 res, no striding
    b = residual_unit(filters, decay, first = True)(b) # 2 layers inside
    for _ in np.arange(1, depth):  # start from 1 => 2 * depth in total
        b = residual_unit(filters, decay)(b)

    filters *= 2

    # 2 res, with striding
    b = residual_unit(filters, decay, more_filters = True)(b)
    for _ in np.arange(1, depth):
        b = residual_unit(filters, decay)(b)

    filters *= 2

    # 3 res, with striding
    b = residual_unit(filters, decay, more_filters = True)(b)
    for _ in np.arange(1, depth):
        b = residual_unit(filters, decay)(b)

    b = BatchNormalization(axis = BN_AXIS)(b)
    b = Activation('relu')(b)

    b = AveragePooling2D(pool_size = (8, 8), strides = (1, 1),
                            padding = 'valid')(b)

    out = Flatten()(b)
    dense = Dense(units = num_classes, kernel_initializer = 'he_normal',
                    activation = 'softmax', kernel_regularizer = l2(decay),
                    bias_regularizer =  l2(0))(out)

    return tf.keras.Model(inputs = input, outputs = dense)
