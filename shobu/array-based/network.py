import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, initializers, optimizers, utils, losses, models, callbacks, metrics
from logic3 import Game, State, Random

class Network:
    'Class containing all methods and variables that interact with the neural network.'
    # Configs
    action_space_shape = (4,8,2,4,4,4,4)
    num_actions = 4*8*2*4*4*4*4
    batch_size = 50
    batch_save_freq = 20*batch_size
    momentum = 0.9
    workers = 1
    num_explorative_moves = 20
    max_plies = 150
    num_simulations = 100
    c_visit = 50
    c_scale = 1
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=2e-1,
        decay_steps=batch_save_freq//2,
        decay_rate=0.99,
        staircase=False)
    training_steps = int(10e3)
    model_callbacks = [
        callbacks.EarlyStopping(
            monitor='accuracy',
            verbose=1,
            patience=batch_save_freq//2),
        callbacks.ModelCheckpoint(
            filepath='./weights.{epoch:02d}-{accuracy:.2f}.hdf5',
            monitor='accuracy',
            verbose=1, # 0 for silent callback display
            save_best_only=True,
            save_weights_only=True,
            save_freq=batch_save_freq), # saves model after N batches (if best)
        callbacks.TensorBoard(
            log_dir='./logs', 
            write_graph=True,
            update_freq='batch')
    ]

    def __init__(self) -> None:
        'Creates combined actor-critic network.'
        states = Input(shape=(2,8,8), dtype='float32', name='states')
        x = self.basic_block(input, kernel_size=3)
        x = self.bottleneck_block(x)
        x = self.bottleneck_block(x)
        #x = self.bottleneck_block(x)
        #x = self.bottleneck_block(x)
        #x = self.bottleneck_block(x)
        #x = self.bottleneck_block(x)
        critic = layers.Dense(1, activation='tanh', name='critic')(x)
        x = layers.Flatten(data_format='channels_first')(x)
        actor = layers.Dense(Network.num_actions, activation=None, name='actor')(x)
        self.model = Model(inputs=[states], outputs=[actor, critic])
        self.model.compile(
            optimizer = optimizers.SGD(learning_rate=Network.lr_schedule, momentum=Network.momentum),
            loss = {
                'actor': losses.SparseCategoricalCrossentropy(from_logits=True),
                'critic': losses.MeanSquaredError()
            }, 
            metrics={'actor': 'accuracy', 'critic': 'accuracy'},
            loss_weights = {'actor': 1, 'critic': 1}
        )
        self.model.summary()
        #utils.plot_model(model, "model_diagram.png", show_shapes=True)

    @staticmethod
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
    
    def bottleneck_block(self, tensor, filters=128):
        'Creates residual block with bottleneck and skip connection.'
        x = self.basic_block(tensor, filters=filters, kernel_size=1, strides=1)
        x = self.basic_block(x, filters=filters, kernel_size=3, strides=1)
        x = self.basic_block(x, filters=filters, kernel_size=3, strides=1)
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

    def train(self, states: list[np.ndarray], legal_actions: list[np.ndarray], actor_targets: np.ndarray, critic_targets: np.ndarray):
        '''trains the neural network with examples obtained from self-play.
        Model weights are saved at the end of every epoch, if it's the best seen so far.
        '''
        history = self.model.fit(
            x = {'states':states} ,
            y = {'actor': actor_targets, 'critic': critic_targets},
            batch_size=Network.batch_size,
            callbacks=[Network.model_callbacks],
            workers=Network.workers
            )
    
    def inference(self, state: State):
        if state.current_player == 1:
            input = np.array([state.boards[1],state.boards[0]], state.boards.dtype)
        else:
            input = state.boards
        policy, value = self.model(input, training=False)
        raise NotImplementedError
        masked_policy = policy[0]
        return masked_policy, value[0]



