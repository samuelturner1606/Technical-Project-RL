import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, initializers, optimizers
from logic3 import Game, State, Random

def basic(x, filters=256, kernel_size=3, strides=1, is_input: bool = False):
    'Convolutional layer then BatchNormalization and ReLU activation.'
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="valid",
        data_format='channels_first',
        use_bias=False,
        kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None)
    )(x)
    if is_input:
        x.input_shape=(2, 8, 8)
    x = layers.BatchNormalization(axis=1, momentum=0.999, epsilon=0.001, center=True, scale=False)(x)
    x = layers.ReLU()(x)
    return x

def bottleneck(tensor, filters=128):
    'Creates residual block with bottleneck and skip connection.'
    x = basic(tensor, filters=filters, kernel_size=1, strides=1)
    x = basic(x, filters=filters, kernel_size=3, strides=1)
    x = basic(x, filters=filters, kernel_size=3, strides=1)
    x = layers.Conv2D(
        filters=2*filters,
        kernel_size=1,
        strides=1,
        padding="valid",
        data_format='channels_first',
        use_bias=False,
        kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None)
    )(x)
    x = layers.BatchNormalization(axis=1, momentum=0.999, epsilon=0.001, center=True, scale=False)(x)
    x = layers.Add()([tensor,x]) # skip connection
    x = layers.ReLU()(x)
    return x

def make_network():
    input = Input(shape=(2,8,8), dtype=tf.float32)
    x = basic(input, kernel_size=3, is_input=True)
    x = bottleneck(x)
    x = bottleneck(x)
    x = bottleneck(x)
    x = bottleneck(x)
    x = bottleneck(x)
    x = bottleneck(x)
    x = layers.Flatten(data_format='channel_first')(x)
    actor = layers.Dense(16384, activation='softmax', name='actor')(x)
    critic = layers.Dense(1, activation='tanh', name='critic')(x)  
    model = Model(inputs=input, outputs=[actor, critic])
    model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=optimizers.Adam()) 
    model.summary()
    return model

def process_state(state: State):
    'Makes 2x8x8 ndarray as input to neural network with the index (0,...) always corresponding to current player.'
    if state.current_player == 1:
        return np.array([state.boards[1],state.boards[0]], state.boards.dtype)
    else:
        return state.boards