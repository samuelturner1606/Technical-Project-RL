import numpy as np
import random

class TicTacToe:
    def __init__(self):
        self.epsilon = 0.1  # exploration rate
        self.alpha = 0.9  # learning rate
        self.gamma = 1  # disount factor
        self.Q = {(0,0,0,0,0,0,0,0,0): {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}} # { states : {actions : values }
        
    def step(self,action):
        self.state[action] = self.player # applies action
        reward, done = self.is_game_over(self.state)
        info = None
        return self.state, reward, done, info

    def reset(self):
        self.state = np.zeros(9, dtype=int) # [0 0 0 0 0 0 0 0 0]
        self.player = 1 # 'player X' = 1, 'player O' = -1
        return np.copy(self.state)

    def render(self, state, reward):
        print('\nState:\n'+str(np.reshape(state,[3,3]))+'\nreward: '+str(reward))
        print('\nQ table: '+str(self.Q))
        return
    
    def is_game_over(self, state):
        state = np.reshape(state, [3,3])
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
            reward = 0
        return reward, done
    
    def bellman_update(self, old_state, action, reward, state, done):
        q = self.Q[tuple(old_state)][action]
        if done:
            td = 0
        else:
            best_next_action, best_next_q = self.best_action(state)
            td = self.gamma*best_next_q - q # temporal difference
        # Bellman optimality equation
        self.Q[tuple(old_state)][action] = q + self.alpha * (reward + td)
        return

    def best_action(self, state):
        best_action, best_q = None, None
        for action in self.possible_actions(state):
            q = self.Q[tuple(state)][action]
            if self.player == 1: # maximise q
                if best_q is None or best_q < q:
                    best_q = q
                    best_action = action
            else: # minimise q
                if best_q is None or best_q > q:
                    best_q = q
                    best_action = action
        return best_action, best_q
    
    def possible_actions(self, state):
        return np.nonzero(state == 0)[0]
    
    def greedy_policy(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.possible_actions(state))
        else:
            action, best_q = self.best_action(state)
        return action


if __name__ == "__main__":
    env = TicTacToe()
    for episode in range(20):
        old_state = env.reset()
        for action in range(9):
            state, reward, done, info = env.step(action)
            if tuple(state) not in env.Q and not done:
                    env.Q[tuple(state)] = dict.fromkeys(env.possible_actions(state), 0)
            env.bellman_update(old_state, action, reward, state, done)
            old_state = np.copy(state)
            env.player *= -1 # change player
            if done:
                env.render(state, reward)
                break