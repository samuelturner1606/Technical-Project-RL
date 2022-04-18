import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, initializers, optimizers, utils, losses, models, callbacks, metrics
from logic3 import Game, State, Random

def basic_block(x, filters=256, kernel_size=3, strides=1):
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

def bottleneck_block(x1, filters=128):
    'Creates residual block with bottleneck and skip connection.'
    x = basic_block(x1, filters=filters, kernel_size=1, strides=1)
    x = basic_block(x, filters=filters, kernel_size=3, strides=1)
    x = basic_block(x, filters=filters, kernel_size=3, strides=1)
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
    x = layers.Add()([x1,x]) # skip connection
    x = layers.ReLU()(x)
    return x

# Create combined actor-critic network.
states = Input(shape=(2,8,8), dtype='float32', name='states')
x = basic_block(input, kernel_size=3)
x = bottleneck_block(x)
x = bottleneck_block(x)
#x = bottleneck_block(x)
#x = bottleneck_block(x)
#x = bottleneck_block(x)
#x = bottleneck_block(x)
x = layers.Flatten(data_format='channels_first')(x)
actor = layers.Dense(4*8*2*4*4*4*4, activation=None, name='actor')(x)
critic = layers.Dense(1, activation='sigmoid', name='critic')(x)
    
class Network:
    'Class containing all methods and variables that interact with the neural network.'
    ### Self-Play
    num_explorative_moves = 30
    max_plies = 150
    num_simulations = 400
    
    # Root prior exploration noise.
    root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
    root_exploration_fraction = 0.25

    # UCB formula
    pb_c_base = 19652
    pb_c_init = 1.25

    ### Training
    training_steps = int(10e3)
    batch_size = 50
    checkpoint_interval = 20*batch_size
    weight_decay = 1e-4
    momentum = 0.9

    learning_rate_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=2e-1,
        decay_steps=checkpoint_interval//2,
        decay_rate=0.99,
        staircase=False )
    
    model_callbacks = [
        callbacks.EarlyStopping(
        monitor='accuracy',
        verbose=1,
        patience=checkpoint_interval//2),
        callbacks.ModelCheckpoint(
        filepath='./weights.{epoch:02d}-{accuracy:.2f}.hdf5',
        monitor='accuracy',
        verbose=1, # 0 for silent callback display
        save_best_only=True,
        save_weights_only=True,
        save_freq=checkpoint_interval), # saves model after N batches (if best)
        callbacks.TensorBoard(
        log_dir='./logs', 
        write_graph=True,
        update_freq='batch') ]
    
    # model is defined outside of __init__() so that all Network instances share the same model
    model = Model(inputs=[states], outputs=[actor, critic])
    model.compile(
        optimizer = optimizers.SGD(learning_rate=learning_rate_schedule, momentum=momentum),
        loss = {
            'actor': losses.SparseCategoricalCrossentropy(from_logits=True),
            'critic': losses.MeanSquaredError()
        }, 
        metrics={'actor': 'accuracy', 'critic': 'accuracy'},
        loss_weights = {'actor': 1, 'critic': 1}
    )
    model.summary()

    def inference(self, board: np.ndarray):
        '@return (masked_policy, value)'
        policy, value = self.model(board, training=False)
        NotImplemented
        masked_policy = policy[0]
        return masked_policy, value[0]
    
    def train(self, states: list[np.ndarray], actor_targets: np.ndarray, critic_targets: np.ndarray):
        '''trains the neural network with examples obtained from self-play.
        Model weights are saved at the end of every epoch, if it's the best seen so far.
        '''
        Network.model.fit(
            x = {'states': states},
            y = {'actor': actor_targets, 'critic': critic_targets},
            batch_size=Network.batch_size,
            callbacks=[Network.model_callbacks],
            workers=1)
