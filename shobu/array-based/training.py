'''
AlphaZero algorithm applied to Shōbu. 
AlphaZero cycles between: self-play data generation and network training. 
network checkpoints are handled by keras callback functions.
'''

import math
import numpy as np
from network import Network
from logic3 import Game, State, Random, Human

class Node(State):
    'Extends the State class with statistics used in MCTS.'
    def __init__(self, current_player: bool, boards: np.ndarray = None, parent: object = None, prior: float = 0) -> None:
        if boards is None:
            self.boards = boards
        else:
            super().__init__(boards)
        self.parent = parent
        self.children = {}
        self.value_sum = 0 
        self.visit_count = 0
        self.prior = prior
        self.current_player = current_player

class MCTS:
    'Player that performs a Monte Carlo Tree search with neural network priors to select moves each turn.'
    def policy(self, root_legal_actions: np.ndarray, game: Game) -> Node:
        '''Run N simulations of the Core MCTS algorithm and return a good action.
        - build a game tree starting from the `root`
        - traverse down according to the UCB formula until a leaf node is found
        - backpropagate statistics upto the root
        '''
        root = Node(current_player=game.current_player, boards=game.state.boards)
        self.evaluate(root, root_legal_actions)
        self.add_exploration_noise(root)
        for _ in range(Network.num_simulations):
            leaf = self.tree_policy(root)
            leaf_legal_actions = leaf.legal_actions() 
            leaf_value = self.evaluate(leaf, leaf_legal_actions)
            self.backpropagate(leaf, leaf_value)
            if not np.any(leaf_legal_actions): # pseudo inexpensive termination condition
                break
        # produce policy target logits from child visit counts and select best action
        policy_target = root_legal_actions.flatten().astype(np.float32)
        max_visits = 0
        best_action = None
        for a in np.argwhere(root_legal_actions):
            flat_a =  np.ravel_multi_index(a, root_legal_actions.shape)
            count = root.children[tuple(a)].visit_count
            policy_target[flat_a] = count
            if count > max_visits:
                max_visits = count
                best_action = tuple(a)
        game.policy_targets.append(policy_target)
        return best_action

    @staticmethod
    def evaluate(node: Node, legal_actions: np.ndarray) -> float:
        '''Use the neural network to evaluate the `node` board state and create child nodes with priors. 
        @return value: predicted value of node board state
        '''
        assert legal_actions.shape == (4,8,2,4,4,4,4), 'legal actions not the right shape'
        softmax_policy, value = Network.inference(node.boards, legal_actions.flatten())
        for a in np.argwhere(legal_actions):
            prior = softmax_policy[ np.ravel_multi_index(a, legal_actions.shape) ]
            node.children[tuple(a)] = Node(current_player=1-node.current_player, parent=node, prior=prior)
        return value

    @staticmethod
    def add_exploration_noise(node: Node):
        'At the start of each search, we add dirichlet noise to the prior of the root to encourage the search to explore new actions.'
        actions = node.children.keys()
        noise = np.random.gamma(Network.root_dirichlet_alpha, 1, len(actions))
        frac = Network.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    def tree_policy(self, node: Node) -> Node:
        'Traverse down the game tree from the root node until a leaf node is found.'
        while len(node.children) > 0: # while not a leaf node
            action, node = self.select_child(node)
            node.boards = node.parent.boards
            node.boards = node.apply(action)
        return node

    def select_child(self, node: Node):
        'Select the child with the highest UCB score.'
        _, action, child = max( (self.ucb_score(node, child), action, child) for action, child in node.children.items() )
        return action, child

    @staticmethod
    def ucb_score(parent: Node, child: Node):
        'The score for a node is based on its value, plus an exploration bonus based on the prior.'
        pb_c = math.log((parent.visit_count + Network.pb_c_base + 1) / Network.pb_c_base) + Network.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior
        if child.visit_count == 0:
            value_score = 0
        else:
            value_score = child.value_sum / child.visit_count # normalised Q value
        return prior_score + value_score

    @staticmethod
    def backpropagate(leaf: Node, leaf_value: float) -> None:
        '''Update the node statistics starting from the leaf node up to the root node.
        @param leaf_value: neural networks prediction of the probability of the leaf node's `current player` winning'''
        node = leaf
        while node:
            node.visit_count += 1
            node.value_sum += ( leaf_value if node.current_player == leaf.current_player else (1 - leaf_value) )
            node = node.parent
        return

if __name__ == '__main__':
    for _ in range(Network.training_steps):
        game = Game(MCTS(), MCTS())
        state_history, policy_targets, critic_targets = game.play()
    pass