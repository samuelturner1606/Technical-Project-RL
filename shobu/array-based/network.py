import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, layers, Model, optimizers, losses, callbacks

def bottleneck_block(x):
    'Creates residual block with bottleneck and skip connection.'
    m = layers.Conv2D(64, (1,1), padding='same', data_format='channels_first', use_bias=False, kernel_initializer='truncated_normal')(x)
    m = layers.BatchNormalization(axis=1, momentum=0.999, scale=False)(m)
    m = layers.ReLU(6)(m)
    m = layers.DepthwiseConv2D((3,3), padding='same', data_format='channels_first', use_bias=False, depthwise_initializer='truncated_normal')(m)
    m = layers.BatchNormalization(axis=1, momentum=0.999, scale=False)(m)
    m = layers.ReLU(6)(m)
    m = layers.DepthwiseConv2D((3,3), padding='same', data_format='channels_first', use_bias=False, depthwise_initializer='truncated_normal')(m)
    m = layers.BatchNormalization(axis=1, momentum=0.999, scale=False)(m)
    m = layers.ReLU(6)(m)
    m = layers.Conv2D(16, (1,1), padding='same', data_format='channels_first', use_bias=False, kernel_initializer='truncated_normal')(m)
    m = layers.BatchNormalization(axis=1, momentum=0.999, scale=False)(m)
    m = layers.Add()([m, x]) # skip connection
    return layers.ReLU(6)(m)

# Create combined actor-critic network
states = Input(shape=(2,8,8), dtype='float32', name='states')
x = layers.Conv2D(16, (3,3), padding='same', data_format='channels_first', use_bias=False, kernel_initializer='truncated_normal')(states)
x = layers.BatchNormalization(axis=1, momentum=0.999, scale=False)(x)
x = layers.ReLU(6)(x)
x = bottleneck_block(x)
x = bottleneck_block(x)
x = layers.Flatten(data_format='channels_first')(x)
actor_logits = layers.Dense(4*8*2*4*4*4*4, activation=None, name='actor_logits')(x)
critic = layers.Dense(1, activation='sigmoid', name='critic')(x)
    
class Network:
    'Class containing all methods and variables that interact with the neural network.'
    ### Self-Play
    num_explorative_moves = 30
    num_simulations = 400
    
    # Root prior exploration noise.
    root_dirichlet_alpha = 0.1
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
    model = Model(inputs=[states], outputs=[actor_logits, critic])
    model.compile(
        optimizer = optimizers.SGD(learning_rate=learning_rate_schedule, momentum=momentum),
        loss = {
            'actor_logits': losses.SparseCategoricalCrossentropy(from_logits=True),
            'critic': losses.MeanSquaredError()
        }, 
        metrics={'actor_logits': 'accuracy', 'critic': 'accuracy'},
        loss_weights = {'actor_logits': 1, 'critic': 1}
    )
    model.summary()

    @staticmethod
    def inference(board: np.ndarray):
        '@return (masked_policy, value)'
        policy, value = Network.model(board, training=False)
        NotImplemented
        masked_policy = policy[0]
        return masked_policy, value[0]
    
    @staticmethod
    def train(states: list[np.ndarray], actor_targets: np.ndarray, critic_targets: np.ndarray):
        '''trains the neural network with examples obtained from self-play.
        Model weights are saved at the end of every epoch, if it's the best seen so far.
        '''
        Network.model.fit(
            x = {'states': states},
            y = {'actor_logits': actor_targets, 'critic': critic_targets},
            batch_size=Network.batch_size,
            callbacks=[Network.model_callbacks],
            workers=1)
