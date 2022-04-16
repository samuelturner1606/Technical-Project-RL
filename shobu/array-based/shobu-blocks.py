import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, initializers, optimizers, utils, losses, models
from logic3 import Game, State, Random

def basic(x, filters=256, kernel_size=3, strides=1):
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
    state = Input(shape=(2,8,8), dtype='float32', name='state')
    x = basic(input, kernel_size=3)
    x = bottleneck(x)
    x = bottleneck(x)
    x = bottleneck(x)
    x = bottleneck(x)
    x = bottleneck(x)
    x = bottleneck(x)
    x = layers.Flatten(data_format='channels_first')(x)
    actor = layers.Dense(16384, activation='softmax', name='actor')(x)
    critic = layers.Dense(1, activation='tanh', name='critic')(x)  
    model = Model(inputs=state, outputs=[actor, critic])
    model.compile(
        optimizer = 'sgd', # optimizers.Adam()
        loss = {
            'actor': losses.SparseCategoricalCrossentropy(from_logits=True),
            'critic': losses.MeanSquaredError()
            }, 
        loss_weights = {'actor': 1, 'critc': 1},
        metrics=["acc"]
    )
    model.summary()
    #utils.plot_model(model, "model_diagram.png", show_shapes=True)
    return model

def process_state(state: State):
    'Makes 2x8x8 ndarray as input to neural network with the index (0,...) always corresponding to current player.'
    if state.current_player == 1:
        return np.array([state.boards[1],state.boards[0]], state.boards.dtype)
    else:
        return state.boards

def train(model: Model, states: list[State], actor_targets, critic_targets):
    'This function trains the neural network with examples obtained from self-play.'
    return model.fit(
        x = {'state': [process_state(s) for s in states]}, # might need to change to ndarray
        y = {'actor': actor_targets, 'critic': critic_targets},
        epochs=10,
        batch_size=64,
        )

def save_checkpoint(model: Model):
    'functions for saving and loading.'
    model.save('saved_models')
    del model
    # Recreate the exact same model purely from the file:
    return models.load_model('saved_models')

'''
# Simple training loop
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# Iterate over the batches of a dataset.
for x, y in dataset:
    with tf.GradientTape() as tape:
        logits = model(x)
        # Compute the loss value for this batch.
        loss_value = loss_fn(y, logits)

    # Update the weights of the model to minimize the loss value.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

'''
