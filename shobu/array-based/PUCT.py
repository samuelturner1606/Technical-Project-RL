from random import choice
import math
import numpy as np
from logic3 import State
from network import Network
from logic3 import Game

class Node(State):
    def __init__(self, to_play: bool, boards: np.ndarray = None, parent: object = None, prior: float = 0) -> None:
        super().__init__(boards)
        self.parent = parent
        if parent:
            parent.children.append(self)
        self.children = {}
        self.value_sum = 0 
        self.visit_count = 0
        self.prior = prior
        self.to_play = to_play

    def is_leaf(self) -> bool: 
        return len(self.children) == 0

    def value(self) -> float:
        'Normalised Q value.'
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def policy(legal_actions: np.ndarray, game: Game) -> Node:
        '''Run N simulations of the Core Monte Carlo Tree Search algorithm and return a good action.
        - build a game tree starting from the `root`
        - traverse down according to the UCB formula until a leaf node is found
        - backpropagate statistics upto the root
        '''
        root = Node(to_play=game.current_player, boards=game.state.boards, )
        for _ in range(Network.num_simulations):
            leaf = tree_policy(root)
            reward = default_policy(leaf)
            backup(leaf, reward)
        return max(root.children, key = lambda child: child.visit_count) # select(root, 0)



def mcts(self, legal_actions: np.ndarray, game: Game):
    root = Node(0)
    evaluate()
    add_exploration_noise(root)
    for _ in range(Network.num_simulations):
        node = root
        scratch_game = Game(action_history=game.action_history)
        search_path = [node]
        while not node.is_leaf():
            action, node = select_child(node)
            scratch_game.action_history.append(action)
            search_path.append(node)
        value = evaluate(node, scratch_game, legal_actions)
        backpropagate(search_path, value, scratch_game.current_player)
    store_search_statistics(root)
    return # select_action





def evaluate(node: Node, legal_actions: np.ndarray) -> float:
  '''Use the neural network to update the prior of children nodes. 
  @return value: predicted value of node board state
  '''
  assert legal_actions.shape == (4,8,2,4,4,4,4), 'legal actions not the right shape'
  softmax_policy, value = Network.inference(node.boards, legal_actions.flatten())
  for a in np.argwhere(legal_actions):
    prior = softmax_policy[ np.ravel_multi_index(a, legal_actions.shape) ]
    node.children[tuple(a)] = Node(prior=prior)
  return value


def add_exploration_noise(node: Node):
  'At the start of each search, we add dirichlet noise to the prior of the root to encourage the search to explore new actions.'
  actions = node.children.keys()
  noise = np.random.gamma(Network.root_dirichlet_alpha, 1, len(actions))
  frac = Network.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac





def tree_policy(node: Node) -> Node:
    'Traverse down the game tree from the root node until a leaf node is found.'
    while not node.is_terminal():
        if not node.children:
            return expand(node)
        else:
            node = select(node, C)
    return node

def expand(node: Node) -> Node:
    'Add unexplored children nodes to the tree and return one such child at random.'
    children = node.find_children()
    return choice(children)

def select(node: Node, c: float) -> Node:
    'Select the best child `node` using UCB1, to balance exploration and exploitation.'
    if node.player == 1:
        best_child = max(node.children, key = lambda child: child.value_sum / child.visit_count+ c * math.sqrt(2 * math.log(node.N) / child.N) if child.visit_count!= 0 else float('inf'))
    else:
        best_child = min(node.children, key = lambda child: child.value_sum / child.visit_count- c * math.sqrt(2 * math.log(node.N) / child.N) if child.visit_count!= 0 else float('-inf'))
    return best_child

def select2(node: Node, c: float) -> Node:
    'Alternative to `select` function in which ties between the best children are broken randomly.'
    v = []
    for child in node.children:
        if node.player == 1:
            child.v = child.value_sum / child.visit_count+ c * math.sqrt(2 * math.log(node.N) / child.N) if child.visit_count!= 0 else float('inf')
        else:
            child.v = child.value_sum / child.visit_count- c * math.sqrt(2 * math.log(node.N) / child.N) if child.visit_count!= 0 else float('-inf')
        v.append(child.v)
    if node.player == 1:
        best_children = [child for child in node.children if child.v == max(v)]
    else:
        best_children = [child for child in node.children if child.v == min(v)]
    random_best = choice(best_children)
    return random_best


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
















def default_policy(node: Node) -> int:
    'Randomly play from the leaf `node` until a terminal node is reached and return the reward.'
    while not node.is_terminal():
        node = node.find_random_child()
    return node.reward()











def backup(node: Node, reward: int) -> None:
    'Update the node statistics starting from the leaf node up to the root node.'
    while node:
        node.visit_count+= 1
        node.value_sum += reward
        node = node.parent
    return

def backpropagate(search_path: list[Node], value: float, to_play):
  'At the end of a simulation, we propagate the evaluation all the way up the tree to the root.'
  for node in search_path:
    node.value_sum += value if node.to_play == to_play else (1 - value)
    node.visit_count += 1







def store_search_statistics(self, root):
    sum_visits = sum(child.visit_count for child in root.children.itervalues())
    self.child_visits.append([root.children[a].visit_count / sum_visits if a in root.children else 0 for a in range(self.num_actions)])
