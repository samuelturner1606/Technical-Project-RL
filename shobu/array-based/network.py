import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, layers, Model, optimizers, losses, callbacks, metrics, utils, models
import os

device_name = tf.test.gpu_device_name()
if not device_name:
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

def bottleneck_block(x):
    'Creates residual block with bottleneck and skip connection.'
    m = layers.Conv2D(64, (1,1), padding='same', data_format='channels_first', use_bias=False, kernel_initializer='truncated_normal')(x)
    m = layers.BatchNormalization(axis=1, momentum=0.999, scale=False)(m)
    m = layers.ReLU()(m)
    m = layers.DepthwiseConv2D((3,3), padding='same', data_format='channels_first', use_bias=False, depthwise_initializer='truncated_normal')(m)
    m = layers.BatchNormalization(axis=1, momentum=0.999, scale=False)(m)
    m = layers.ReLU()(m)
    m = layers.DepthwiseConv2D((3,3), padding='same', data_format='channels_first', use_bias=False, depthwise_initializer='truncated_normal')(m)
    m = layers.BatchNormalization(axis=1, momentum=0.999, scale=False)(m)
    m = layers.ReLU()(m)
    m = layers.Conv2D(16, (1,1), padding='same', data_format='channels_first', use_bias=False, kernel_initializer='truncated_normal')(m)
    m = layers.BatchNormalization(axis=1, momentum=0.999, scale=False)(m)
    m = layers.Add()([m, x]) # skip connection
    return layers.ReLU()(m)

# Create combined actor-critic network
states = Input(shape=(2,8,8), dtype='float32', name='states')
x = layers.Conv2D(16, (3,3), padding='same', data_format='channels_first', use_bias=False, kernel_initializer='truncated_normal')(states)
x = layers.BatchNormalization(axis=1, momentum=0.999, scale=False)(x)
x = layers.ReLU()(x)
x = bottleneck_block(x)
x = bottleneck_block(x)
x = layers.Flatten(data_format='channels_first')(x)
actor_logits = layers.Dense(4*8*2*4*4*4*4, activation=None, name='actor_logits')(x)
critic = layers.Dense(1, activation='sigmoid', name='critic')(x)
    
class Network:
    'Class containing all methods and variables that interact with the neural network.'
    ### Self-Play
    num_simulations = 50
    
    # Root prior exploration noise.
    root_dirichlet_alpha = 0.1
    root_exploration_fraction = 0.25

    # UCB formula
    pb_c_base = 19652
    pb_c_init = 1.25

    ### Training
    training_games = int(15e3)
    weight_decay = 1e-4
    momentum = 0.9
    exporative_moves = 20
    checkpoint_freq = 20

    learning_rate_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=2e-1,
        decay_steps=500,
        decay_rate=0.99,
        staircase=False )
    
    checkpoint_dir = './checkpoints'
    log_dir = './logs'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model_callbacks = [
        callbacks.EarlyStopping(
            monitor='loss',
            verbose=0,
            patience=500),
        callbacks.ModelCheckpoint(
            filepath=f'{checkpoint_dir}/weights-best.hdf5',
            monitor='loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            save_freq='epoch'),
        callbacks.TensorBoard(
            log_dir=log_dir,
            update_freq=100)  
    ]
    model = Model(inputs=[states], outputs=[actor_logits, critic])
    #utils.plot_model(model, "model_diagram.png", show_shapes=True)
    checkpoints = [f'./checkpoints/{name}' for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        model.load_weights(latest_checkpoint)
 
    model.compile(
        optimizer = optimizers.SGD(learning_rate=learning_rate_schedule, momentum=momentum),
        loss = {
            'actor_logits': losses.CategoricalCrossentropy(from_logits=True),
            'critic': losses.MeanSquaredError()
        }, 
        metrics={
            'actor_logits': metrics.CategoricalAccuracy(dtype=tf.float32), 
            'critic': metrics.Accuracy(dtype=tf.float32)},
        #loss_weights = {'actor_logits': 1, 'critic': 1}
    )

    @staticmethod
    def inference(board: np.ndarray, legal_actions: np.ndarray) -> tuple[np.ndarray, float]:
        '''@return softmax_policy: probability of actions, illegal actions have probability of 0
        @return value: float between 0 and 1'''
        assert legal_actions.ndim == 1, 'legal actions not flat'
        policy_logits, value = Network.model(board[None,...], training=False)
        masked_policy = policy_logits[0].numpy()
        masked_policy[legal_actions==0] = -np.inf
        def softmax(x):
            f = np.exp(x - np.max(x))  # shift values
            return f / f.sum(axis=0)
        return softmax(masked_policy), value.numpy()[0,0]
    
    @staticmethod
    def train(state_history: np.ndarray, actor_targets: np.ndarray, critic_targets: np.ndarray):
        '''Trains the neural network from a game.
        Model weights are saved at the end of every epoch, if it's the best seen so far.
        '''
        Network.model.fit(
            x = {'states': state_history},
            y = {'actor_logits': actor_targets, 'critic': critic_targets},
            batch_size=critic_targets.shape[0],
            callbacks=[Network.model_callbacks])