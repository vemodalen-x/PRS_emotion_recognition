import keras.backend as K
from keras.layers.convolutional import Conv3D
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import (AveragePooling3D, GlobalAveragePooling3D, AveragePooling2D)
from keras.regularizers import l2
from keras.layers import Input, Flatten
from keras.models import Model


def __create_Conv3D_net(simple,
                        width,
                        height,
                        frames,
                        nb_classes,
                        weight_decay=1e-4,
                        activation='softmax',
                        # attention=True,
                        ):
    # 如果channel 在第一维的话 那此值为1 否则(tensorflow默认)为-1

    DInput = Input([height, width, frames, 1])
    x = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same',
               strides=(1, 1, 1), kernel_regularizer=l2(weight_decay), data_format='channels_last',
               input_shape=(simple, frames, width, height, 1))(DInput)
    x = Activation('relu')(x)
    x = AveragePooling3D()(x)
    #
    x = Conv3D(16, (3, 3, 3), padding='same',
               strides=(1, 1, 1), kernel_regularizer=l2(weight_decay), data_format='channels_last')(x)
    x = Activation('relu')(x)
    x = AveragePooling3D()(x)

    x = Conv3D(32, (3, 3, 3), padding='same',
               strides=(1, 1, 1), kernel_regularizer=l2(weight_decay), data_format='channels_last')(x)
    x = Activation('relu')(x)
    x = AveragePooling3D()(x)

    x = Conv3D(2, (3, 3, 3), padding='same',
               strides=(1, 1, 1), kernel_regularizer=l2(weight_decay), data_format='channels_last')(x)
    x = Activation('relu')(x)
    x = AveragePooling3D()(x)

    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(nb_classes, activation=activation)(x)

    model = Model(DInput, x)
    return model
