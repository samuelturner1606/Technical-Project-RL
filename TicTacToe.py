from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

class Game(Env):
    def __init__(self):
        self.action_space = Discrete(9) # [0 1 2 3 4 5 6 7 8]
        self.observation_space = Discrete(9)
        self.reset()

    def step(self, action):
        self.state[action] = self.player # applies action
        self.isGameOver() # calculates reward
        info = {} # placeholder
        self.player *= -1 # changes player
        return self.state, self.reward, self.done, info

    def reset(self):
        self.state = np.zeros(9, dtype=int) # [0 0 0 0 0 0 0 0 0]
        self.player = 1 # 'player X' = 1, 'player O' = -1
        self.done = False
        self.reward = None
        return self.state

    def render(self):
        print('\nstate:\n'+str(self.state.reshape([3,3]))+'\nreward: '+str(self.reward))
        return
    
    def randomAction(self):
        emptySpots = np.nonzero(self.state == 0)[0]
        randomAction = random.choice(emptySpots)
        self.step(randomAction)
        return

    def isGameOver(self):
        state = self.state.reshape([3,3])
        S = []
        for i in range(3):
            S.append( sum(state[i,:]) )  # rows
        for j in range(3):
            S.append( sum(state[:,j]) ) # columns
        S.append( np.trace(state) ) # left diagonal  
        S.append( np.trace(np.fliplr(state)) ) # right diagonal  
        if 3 in S: # X wins
            self.done = True
            self.reward = 1
        elif -3 in S: # O wins
            self.done = True
            self.reward = -1
        elif 0 not in self.state: # draw
            self.done = True
            self.reward = 0
        return 

env = Game()
for episode in range(10):
    while not env.done:
        env.randomAction()
    env.render()
    env.reset()