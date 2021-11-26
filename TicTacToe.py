import numpy as np
import random
import matplotlib.pyplot as plt

class TicTacToe:
    def __init__(self):
        self.epsilon = 1  # exploration rate
        self.alpha = 0.5  # learning rate
        self.gamma = 0.9  # disount factor
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
        new_q = q + self.alpha * (reward + td)
        self.Q[tuple(old_state)][action] = round(new_q, 2)
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

    def grow_Q(self, state, done):
        if not done and tuple(state) not in self.Q:
                        self.Q[tuple(state)] = dict.fromkeys(self.possible_actions(state), 0)
        return

    def train(self,epochs):
        for i in range(epochs):
            old_state = self.reset()
            while True:
                action = self.greedy_policy(old_state)
                state, reward, done, info = self.step(action)
                self.grow_Q(state, done)
                self.bellman_update(old_state, action, reward, state, done)
                old_state = np.copy(state)
                self.player *= -1 # change player
                if done:
                    if self.epsilon > 0.1:
                        self.epsilon = round(self.epsilon*0.999,2)
                    # self.render(state, reward)
                    break
        return
    
    def test(self,epochs):
        results = []
        for i in range(epochs):
            state = self.reset()
            while True:
                if self.player == 1:
                    action, best_q = self.best_action(state)
                else:
                    action = random.choice(self.possible_actions(state))
                state, reward, done, info = self.step(action)
                self.grow_Q(state, done)
                self.player *= -1
                if done:
                    results.append(reward)
                    break
        win_rate = len([x for x in results if x == 1])/epochs
        draw_rate = len([x for x in results if x == 0])/epochs
        loss_rate = len([x for x in results if x == -1])/epochs
        return win_rate, draw_rate, loss_rate
    
    def average_q(self):
        q_values = []    
        for action_value_pair in env.Q.values():
            for value in action_value_pair.values():
                q_values.append(value)
        avg = round(np.mean(q_values),5)
        return avg

if __name__ == "__main__":
    env = TicTacToe()
    avg_q_values, win_rates, draw_rates, loss_rates = [], [], [], []
    train_n, test_n, cycles = 1000, 100, 1000

    for i in range(cycles):
        env.train(train_n)
        avg_q_values.append(env.average_q())
        win_rate, draw_rate, loss_rate = env.test(test_n)
        win_rates.append(win_rate)
        draw_rates.append(draw_rate)
        loss_rates.append(loss_rate)

    x = np.arange(train_n, train_n*cycles+1, train_n)

    fig, ax = plt.subplots()
    ax.plot(x, avg_q_values, 'b')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Average Q(s,a) value')
    ax.set_title('Performance during training')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, win_rates, 'g', label='Win rate')
    ax.plot(x, draw_rates, 'b', label='Draw rate')
    ax.plot(x, loss_rates, 'r', label='Loss rate')
    plt.ylim([0, 1])
    ax.set_xlabel('Training epochs')
    ax.set_ylabel('Percentage')
    ax.set_title('Performance against random opponent')
    ax.legend()
    plt.show()