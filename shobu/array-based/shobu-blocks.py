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


######### Example actor-critc from keras docs

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
env = object  # Create the environment
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


num_inputs = 4
num_actions = 2
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = Model(inputs=inputs, outputs=[action, critic])

optimizer = optimizers.Adam(learning_rate=0.01)
huber_loss = losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))

            # Apply the sampled action in our environment
            state, reward, done, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
