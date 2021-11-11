from gym import Env, spaces
import numpy as np
import random

class TicTacToe(Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(3**9)
        self.action_space = spaces.Discrete(9)
        self.epsilon = 0.1  # exploration rate
        self.alpha = 0.1  # learning rate
        self.gamma = 1  # disount factor
        self.q_table = np.zeros([self.observation_space.n, self.action_space.n])

    def step(self, action):
        self.state[action] = self.player # applies action
        reward, done = self._is_game_over()
        info = {} # placeholder
        self.player *= -1 # changes player
        return self.state, reward, done, info

    def reset(self):
        self.state = np.zeros(9, dtype=int) # [0 0 0 0 0 0 0 0 0]
        self.player = 1 # 'player X' = 1, 'player O' = -1
        return self.state

    def render(self, reward):
        print('\nstate:\n'+str(self.state.reshape([3,3]))+'\nreward: '+str(reward))
        return
    
    def _choose_action(self):
        possible_actions = np.nonzero(self.state == 0)[0]
        if random.random() < self.epsilon:
            action = random.choice(possible_actions)
        else:
            action, max_value = None, None
            for possible_action in possible_actions: # could rewrite following code to use argmax?
                value = self.q_table[(self.state, possible_action)]
                if max_value is None or max_value < value:
                    max_value = value
                    action = possible_action
        return action

    def _is_game_over(self):
        state = self.state.reshape([3,3])
        S = []
        for i in range(3):
            S.append( sum(state[i,:]) )  # rows
        for j in range(3):
            S.append( sum(state[:,j]) ) # columns
        S.append( np.trace(state) ) # left diagonal  
        S.append( np.trace(np.fliplr(state)) ) # right diagonal  
        if 3 in S: # X wins
            done = True
            reward = 1
        elif -3 in S: # O wins
            done = True
            reward = -1
        elif 0 not in self.state: # draw
            done = True
            reward = 0
        else: # not ended
            done = False
            reward = None
        return reward, done
    
    def _q_update(self, state, action, reward, new_state, done):
        #new_q_val = q_val + self.alpha * (reward + self.gamma * max_next_q_val - q_val)
        return

if __name__ == "__main__":
    env = TicTacToe()
    epochs = 10
    for episode in range(epochs):
        state = env.reset()
        while True:
            action = env._choose_action()
            new_state, reward, done, info = env.step(action)
            env._q_update(state, action, reward, new_state, done)
            state = new_state
            if done:
                env.render(reward)
                break
    print('\nq table: '+str(env.q_table))