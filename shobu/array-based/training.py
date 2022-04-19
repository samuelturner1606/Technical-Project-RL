'''
AlphaZero algorithm applied to Shōbu. 
AlphaZero cycles between: self-play data generation and network training. 
network checkpoints are handled by keras callback functions.
'''

import math
import numpy as np
import tensorflow as tf
from network import Network
from logic3 import Game

class Node:
  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}
  def expanded(self) -> bool: 
    'Whether the node has children'
    return len(self.children) > 0
  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

class MCTS:
  def policy(self, legal_actions: np.ndarray, game: Game) -> tuple[int]:
    '''Core Monte Carlo Tree Search algorithm. 
    To decide on an action, we run N simulations:
    - always starting at the root of the search tree
    - traversing the tree according to the UCB formula
    - stopping when we reach a leaf node and backpropagating up.'''
    root = Node(0)
    evaluate(root, game, legal_actions)
    add_exploration_noise(root)
    for _ in range(Network.num_simulations):
      node = root
      scratch_game = Game(action_history=game.action_history)
      search_path = [node]
      while node.expanded():
        action, node = select_child(node)
        scratch_game.action_history.append(action)
        search_path.append(node)
      value = evaluate(node, scratch_game, legal_actions)
      backpropagate(search_path, value, scratch_game.current_player)
    game.store_search_statistics(root)
    return select_action(game, root), root

def select_action(game: Game, root: Node):
  visit_counts = [(child.visit_count, action) for action, child in root.children.iteritems()]
  if len(game.action_history) < Network.num_explorative_moves:
    _, action =  NotImplemented # softmax_sample(visit_counts)
  else:
    _, action = max(visit_counts)
  return action

def select_child(node: Node):
  'Select the child with the highest UCB score.'
  _, action, child = max((ucb_score(node, child), action, child) for action, child in node.children.items())
  return action, child

def ucb_score(parent: Node, child: Node):
  'The score for a node is based on its value, plus an exploration bonus based on the prior.'
  pb_c = math.log((parent.visit_count + Network.pb_c_base + 1) / Network.pb_c_base) + Network.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
  prior_score = pb_c * child.prior
  value_score = child.value()
  return prior_score + value_score

def evaluate(node: Node, game: Game, legal_actions: np.ndarray):
  'We use the neural network to obtain a value and policy prediction.'
  assert legal_actions.shape == (4,8,2,4,4,4,4), 'legal actions not the right shape'
  softmax_policy, value = Network.inference(game.state.boards, legal_actions.flatten())
  # Expand the node.
  node.to_play = game.current_player
  for a in np.argwhere(legal_actions):
    prior = softmax_policy[ np.ravel_multi_index(a, legal_actions.shape) ]
    node.children[tuple(a)] = Node(prior)
  return value

def backpropagate(search_path: list[Node], value: float, to_play):
  'At the end of a simulation, we propagate the evaluation all the way up the tree to the root.'
  for node in search_path:
    node.value_sum += value if node.to_play == to_play else (1 - value)
    node.visit_count += 1

def add_exploration_noise(node: Node):
  'At the start of each search, we add dirichlet noise to the prior of the root to encourage the search to explore new actions.'
  actions = node.children.keys()
  noise = np.random.gamma(Network.root_dirichlet_alpha, 1, len(actions))
  frac = Network.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

if __name__ == '__main__':
  # Each game is produced by starting at the initial board position, then
  # repeatedly executing a Monte Carlo Tree Search to generate moves until the end
  # of the game is reached.
  game = Game(MCTS(), MCTS(), True)
  s, a, r = game.play()
  pass