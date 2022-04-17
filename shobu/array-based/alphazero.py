"""Pseudocode description of the AlphaZero algorithm."""

import math
import numpy
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, initializers, optimizers, utils, losses, models, callbacks, metrics

##########################
####### Helpers ##########

class Network:
  ### Self-Play
  workers = 1

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
  checkpoint_interval = int(1e3)
  window_size = int(1e6)
  batch_size = 50
  batch_save_freq = 20*batch_size
  weight_decay = 1e-4
  momentum = 0.9

  learning_rate_schedule = optimizers.schedules.ExponentialDecay(
      initial_learning_rate=2e-1,
      decay_steps=batch_save_freq//2,
      decay_rate=0.99,
      staircase=False)

  def inference(self, image):
    return (-1, {})  # Value, Policy

  def get_weights(self):
    # Returns the weights of this network.
    return []

class Node:
  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}

  def expanded(self):
    return len(self.children) > 0

  def value(self):
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count


class Game:

  def __init__(self, history=None):
    self.history = history or []
    self.child_visits = []
    self.num_actions = 4672  # action space size for chess; 11259 for shogi, 362 for Go

  def terminal(self):
    # Game specific termination rules.
    pass

  def terminal_value(self, to_play):
    # Game specific value.
    pass

  def legal_actions(self):
    # Game specific calculation of legal actions.
    return []

  def clone(self):
    return Game(list(self.history))

  def apply(self, action):
    self.history.append(action)

  def store_search_statistics(self, root):
    sum_visits = sum(child.visit_count for child in root.children.itervalues())
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in range(self.num_actions)
    ])

  def make_image(self, state_index: int):
    # Game specific feature planes.
    return []

  def make_target(self, state_index: int):
    return (self.terminal_value(state_index % 2),
            self.child_visits[state_index])

  def to_play(self):
    return len(self.history) % 2


class ReplayBuffer:
  def __init__(self):
    self.window_size = Network.window_size
    self.batch_size = Network.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self):
    # Sample uniformly across positions.
    move_sum = float(sum(len(g.history) for g in self.buffer))
    games = numpy.random.choice(
        self.buffer,
        size=self.batch_size,
        p=[len(g.history) / move_sum for g in self.buffer])
    game_pos = [(g, numpy.random.randint(len(g.history))) for g in games]
    return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]


class SharedStorage:

  def __init__(self):
    self._networks = {}

  def latest_network(self) -> Network:
    if self._networks:
      return self._networks[max(self._networks.iterkeys())]
    else:
      return Network()  # policy -> uniform, value -> 0.5

  def save_network(self, step: int, network: Network):
    self._networks[step] = network


##### End Helpers ########
##########################


# AlphaZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def alphazero():
  storage = SharedStorage()
  replay_buffer = ReplayBuffer()

  for i in range(Network.workers):
    run_selfplay(storage, replay_buffer)

  train_network(storage, replay_buffer)

  return storage.latest_network()


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(storage: SharedStorage, replay_buffer: ReplayBuffer):
  while True:
    network = storage.latest_network()
    game = play_game(network)
    replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(network: Network):
  game = Game()
  while not game.terminal() and len(game.history) < Network.max_plies:
    action, root = run_mcts(game, network)
    game.apply(action)
    game.store_search_statistics(root)
  return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(game: Game, network: Network):
  root = Node(0)
  evaluate(root, game, network)
  add_exploration_noise(root)

  for _ in range(Network.num_simulations):
    node = root
    scratch_game = game.clone()
    search_path = [node]

    while node.expanded():
      action, node = select_child(node)
      scratch_game.apply(action)
      search_path.append(node)

    value = evaluate(node, scratch_game, network)
    backpropagate(search_path, value, scratch_game.to_play())
  return select_action(game, root), root


def select_action(game: Game, root: Node):
  visit_counts = [(child.visit_count, action) for action, child in root.children.iteritems()]
  if len(game.history) < Network.num_explorative_moves:
    _, action = softmax_sample(visit_counts)
  else:
    _, action = max(visit_counts)
  return action


# Select the child with the highest UCB score.
def select_child(node: Node):
  _, action, child = max((ucb_score(node, child), action, child) for action, child in node.children.iteritems())
  return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(parent: Node, child: Node):
  pb_c = math.log((parent.visit_count + Network.pb_c_base + 1) / Network.pb_c_base) + Network.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  value_score = child.value()
  return prior_score + value_score


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: Game, network: Network):
  value, policy_logits = network.inference(game.make_image(-1))

  # Expand the node.
  node.to_play = game.to_play()
  policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
  policy_sum = sum(policy.itervalues())
  for action, p in policy.iteritems():
    node.children[action] = Node(p / policy_sum)
  return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: list[Node], value: float, to_play):
  for node in search_path:
    node.value_sum += value if node.to_play == to_play else (1 - value)
    node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(node: Node):
  actions = node.children.keys()
  noise = numpy.random.gamma(Network.root_dirichlet_alpha, 1, len(actions))
  frac = Network.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


def train_network(storage: SharedStorage, replay_buffer: ReplayBuffer):
  network = Network()
  optimizer = tf.train.MomentumOptimizer(Network.learning_rate_schedule, Network.momentum)
  for i in range(Network.training_steps):
    if i % Network.checkpoint_interval == 0:
      storage.save_network(i, network)
    batch = replay_buffer.sample_batch()
    update_weights(optimizer, network, batch, Network.weight_decay)
  storage.save_network(Network.training_steps, network)


def update_weights(optimizer: tf.train.Optimizer, network: Network, batch, weight_decay: float):
  loss = 0
  for image, (target_value, target_policy) in batch:
    value, policy_logits = network.inference(image)
    loss += (tf.losses.mean_squared_error(value, target_value) + tf.nn.softmax_cross_entropy_with_logits(logits=policy_logits, labels=target_policy))

  for weights in network.get_weights():
    loss += weight_decay * tf.nn.l2_loss(weights)

  optimizer.minimize(loss)


######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################