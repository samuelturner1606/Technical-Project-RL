import numpy as np
import random

class Game:
    
    def __init__(self):
        self.playing = True
        self.player = 'x'
        self.state = np.zeros([3,3])
        self.reward = None
    
    def action(self, i, j):
        if self.playing:
            if self.player == 'x':
                marker = 1
                self.player = 'o'
            else:
                marker = -1
                self.player = 'x'
            self.state[i][j] = marker
            self.isGameOver()
        return
    
    def isGameOver(self):
        sums = []
        for i in range(3):
            sums.append( sum(self.state[i,:]) )  # rows
        for j in range(3):
            sums.append( sum(self.state[:,j]) ) # columns
        sums.append( np.trace(self.state) ) # left diagonal  
        sums.append( np.trace(np.fliplr(self.state)) ) # right diagonal  
        if 3 in sums: # x wins
            self.playing = False
            self.reward = 1
        elif -3 in sums: # o wins
            self.playing = False
            self.reward = -1
        elif 0 not in self.state: # draw
            self.playing = False
            self.reward = 0
        return
    
    def randomAction(self):
        if self.playing and 0 in self.state:
            empty_spots = []
            for i in range(3):
                for j in range(3):
                    if self.state[i][j] == 0:
                        empty_spots.append((i,j))    
            x, y = random.choice(empty_spots)
            self.action(x, y)
        return

games = [Game() for i in range(20)]
for game in games:
    while game.playing:
        game.randomAction()
    print("\n state: \n"+str(game.state)+", reward: "+str(game.reward))