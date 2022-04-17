import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, initializers, optimizers, utils, losses, models, callbacks
from logic3 import Game, State, Random

class NNet:
    num_actions = 4*8*2*4*4*4*4
    EPOCHS = 10
    checkpoint_filepath = 'weights.{epoch:02d}-{accuracy:.2f}.hdf5'
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='accuracy',
        verbose=1, # 0 for silent callback display
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        save_freq='epoch')

    def __init__(self) -> None:
        states = Input(shape=(2,8,8), dtype='float32', name='states')
        x = self.basic_block(input, kernel_size=3)
        x = self.bottleneck_block(x)
        x = self.bottleneck_block(x)
        #x = self.bottleneck_block(x)
        #x = self.bottleneck_block(x)
        #x = self.bottleneck_block(x)
        #x = self.bottleneck_block(x)
        x = layers.Flatten(data_format='channels_first')(x)
        actor = layers.Dense(16384, activation='softmax', name='actor')(x)
        critic = layers.Dense(1, activation='tanh', name='critic')(x)  
        self.model = Model(inputs=states, outputs=[actor, critic])
        self.model.compile(
            optimizer = 'sgd', # optimizers.Adam()
            loss = {
                'actor': losses.SparseCategoricalCrossentropy(from_logits=True),
                'critic': losses.MeanSquaredError()
                }, 
            loss_weights = {'actor': 1, 'critc': 1},
            metrics=['accuracy']
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

    def train(self, states: np.ndarray, actor_targets: np.ndarray, critic_targets: np.ndarray):
        '''trains the neural network with examples obtained from self-play.
        Model weights are saved at the end of every epoch, if it's the best seen so far.
        '''
        history = self.model.fit(
            x = {'state': states},
            y = {'actor': actor_targets, 'critic': critic_targets},
            epochs=NNet.EPOCHS,
            batch_size=states.shape[0]/2, # need to check
            callbacks=[NNet.model_checkpoint_callback]
            )
    
    def predict(self, state: State):
        'Makes 2x8x8 ndarray as input to neural network with the index (0,...) always corresponding to current player.'
        # preparing input
        if state.current_player == 1:
            input = np.array([state.boards[1],state.boards[0]], state.boards.dtype)
        else:
            input = state.boards
        pi, v = self.model(input, training=False)
        return pi[0], v[0]

    def load_checkpoint(self):
        'The model weights (that are considered the best) are loaded into the model.'
        self.model.load_weights(NNet.checkpoint_filepath)


######### Example actor-critc from keras docs

# Configuration parameters for the whole setup

max_plies = 150
net = NNet()

def play_episode():
    action_history = []
    critic_value_history = []
    reward_history = []
    for timestep in range(1, max_plies):
        # env.render(); Adding this line would show the attempts
        # of the agent in a pop up window.

        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        # Predict action probabilities and estimated future rewards
        # from environment state
        action_probs, critic_value = net.model(state)
        critic_value_history.append(critic_value[0, 0])

        # Sample action from action probability distribution
        action = np.random.choice(NNet.num_actions, p=np.squeeze(action_probs))
        action_history.append(tf.math.log(action_probs[0, action]))

        # Apply the sampled action in our environment
        state, reward, done, _ = env.step(action)
        reward_history.append(reward)

        if done:
            return action_history, critic_value_history, reward_history
    
while True:  # Run until solved
    state = env.reset()
    action_history, critic_value_history, rewards_history = play_episode()
    net.train()