import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class TicTacToe:
    def __init__(self):
        self.epsilon = 1  # exploration rate
        self.alpha = 0.2  # learning rate
        self.gamma = 0.9  # disount factor
        self.Q = {} # { states : {actions : values }
        
    def step(self,action):
        self.state[action] = self.player # applies action
        reward, done = self.is_game_over(self.state)
        info = None
        return self.state, reward, done, info

    def reset(self):
        self.state = np.zeros(9, dtype=int) # [0 0 0 0 0 0 0 0 0]
        self.player = 1 # player 1 = 1, player 2 = -1
        self.grow_Q(self.state, False)
        return np.copy(self.state)

    def render(self, df):
        with open("Q table.txt", "w") as f:
            for state in self.Q.keys():
                f.write('\n'+str(state)+':  '+str(self.Q[state]))

        fig1, ax1 = plt.subplots()
        ax1 = sns.regplot(x='Training episodes', y='Average Q(s,a) value', data=df, color="purple", scatter_kws={"s": 10}, order=2)
        ax1.set_title('Performance during training')
        plt.savefig('plot1.png')

        fig2, ax2 = plt.subplots()
        ax2 = sns.regplot(x='Training episodes', y='Win rate', data=df, color="green", label='Win rate', scatter_kws={"s": 10}, order=2)
        ax2 = sns.regplot(x='Training episodes', y='Draw rate', data=df, color="blue", label='Draw rate', scatter_kws={"s": 10}, order=2)
        ax2 = sns.regplot(x='Training episodes', y='Loss rate', data=df, color="red", label='Loss rate', scatter_kws={"s": 10}, order=2)
        ax2.set_title('Performance against random opponent')
        ax2.set_ylabel('Percentage')
        ax2.set_ylim([0,1])
        ax2.legend()
        plt.savefig('plot2.png')
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
        if 3 in S: # player 1 wins
            done = True
            reward = 1
        elif -3 in S: # player 1 loses
            done = True
            reward = -1
        elif 0 not in self.state: # draw
            done = True
            reward = 0
        else: # not ended
            done = False
            reward = 0
        return reward, done
    
    def possible_actions(self, state):
        return np.nonzero(state == 0)[0]
    
    def greedy_policy(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.possible_actions(state))
        else:
            action, best_q = self.best_action(state,self.player)
        return action

    def best_action(self, state, player):
        best_action, best_q = None, None
        for action in self.possible_actions(state):
            q = self.Q[tuple(state)][action]
            if player == 1: # maximise q of player 1
                if best_q is None or best_q < q:
                    best_q = q
                    best_action = action
            else: # minimise q of player 1
                if best_q is None or best_q > q:
                    best_q = q
                    best_action = action
        return best_action, best_q
    
    def grow_Q(self, state, done):
        if tuple(state) not in self.Q:
            possible_actions = self.possible_actions(state)
            n = len(possible_actions)
            action_value_pair = {}
            for action in possible_actions:
                if done:
                    init_val = 0
                else:
                    init_val = round(random.uniform(-1,1), 2)
                action_value_pair[action]=init_val  
            self.Q[tuple(state)] = action_value_pair
        return

    def bellman_update(self, old_state, action, reward, state, done):
        q = self.Q[tuple(old_state)][action]
        if done:
            best_next_q = 0
        else:
            best_next_action, best_next_q = self.best_action(state, -1*self.player) # 2-player game so opponents best action!
        # Bellman optimality equation
        new_q = q + self.alpha * (reward + self.gamma*best_next_q - q)
        self.Q[tuple(old_state)][action] = round(new_q, 2)
        return
    
    def train(self,episodes):
        for episode in range(episodes):
            old_state = self.reset()
            while True:
                action = self.greedy_policy(old_state)
                state, reward, done, info = self.step(action)
                env.grow_Q(state, done)
                self.bellman_update(old_state, action, reward, state, done)
                old_state = np.copy(state)
                self.player *= -1 # change player
                if done:
                    if self.epsilon > 0.1:
                        self.epsilon = round(self.epsilon*0.999,5)
                    break
        return
    
    def test(self,episodes):
        results = []
        for episode in range(episodes):
            state = self.reset()
            while True:
                if self.player == 1:
                    action, best_q = self.best_action(state, 1)
                else:
                    action = random.choice(self.possible_actions(state))
                state, reward, done, info = self.step(action)
                self.grow_Q(state, done)
                self.player *= -1
                if done:
                    results.append(reward)
                    break
        win_rate = len([x for x in results if x == 1])/episodes
        draw_rate = len([x for x in results if x == 0])/episodes
        loss_rate = len([x for x in results if x == -1])/episodes
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
    train_n, test_n, cycles = 200, 1000, 100

    for cycle in range(cycles):
        env.train(train_n)
        avg_q_values.append(env.average_q())
        win_rate, draw_rate, loss_rate = env.test(test_n)
        win_rates.append(win_rate)
        draw_rates.append(draw_rate)
        loss_rates.append(loss_rate)
    
    x = np.arange(train_n, train_n*cycles+1, train_n)
    data = {'Training episodes':x, 'Win rate':win_rates, 'Loss rate':loss_rates, 'Draw rate':draw_rates, 'Average Q(s,a) value':avg_q_values}
    df = pd.DataFrame(data)
    env.render(df)