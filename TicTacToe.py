from gym import Env, spaces
import numpy as np
import random

class Game(Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(3**9)
        self.action_space = spaces.MultiDiscrete([2, 9])

    def step(self, action):
        self.state[action] = self.player # applies action
        reward, done = self.isGameOver()
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
    
    def getRandomAction(self):
        emptySpots = np.nonzero(self.state == 0)[0]
        randomAction = random.choice(emptySpots)
        return randomAction

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

if __name__ == "__main__":
    env = Game()
    for episode in range(10):
        obs = env.reset()
        while True:
            action = env.getRandomAction()
            obs, reward, done, info = env.step(action)
            if done:
                env.render(reward)
                break