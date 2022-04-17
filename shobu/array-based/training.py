"""Gumbel AlphaZero algorithm for Shōbu."""

import math
import numpy as np
import tensorflow as tf
from network import Network

class Node:
  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.q_sum = 0
    self.children = {}
    self.reward = 0

  def expanded(self) -> bool:
    return len(self.children) > 0

  def mean_q(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.q_sum / self.visit_count

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
        root.children[a].visit_count / sum_visits if a in root.children else 0 for a in range(self.num_actions)
    ])

  def make_image(self, state_index: int):
    # Game specific feature planes.
    return []

  def make_target(self, state_index: int):
    return (self.terminal_value(state_index % 2),
            self.child_visits[state_index])

  def to_play(self):
    return len(self.history) % 2

def train_cycle():
  net = Network()
  for i in range(Network.training_steps):
    game = play_self()
    net.train()

# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_self():
  game = Game()
  while not game.terminal() and len(game.history) < Network.max_plies:
    action, root = run_mcts()
    game.apply(action)
    game.store_search_statistics(root)
  return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts():
  root = Node(0)
  evaluate(root)
  '''
  add_exploration_noise(config, root)
  '''
  for _ in range(Network.num_simulations):
    node = root
    scratch_game = game.clone()
    search_path = [node]

    while node.expanded():
      action, node = select_child(node)
      scratch_game.apply(action)
      search_path.append(node)

    value = evaluate(node, scratch_game)
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
    node.q_sum += value if node.to_play == to_play else (1 - value)
    node.visit_count += 1