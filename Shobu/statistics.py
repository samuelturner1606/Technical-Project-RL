'''Generating statistics from random games.'''
from time import sleep
import shobu as s

if __name__ == '__main__':
    game = s.State()
    num_actions = []
    while True:
        sleep(0.1)
        game.render()
        num_actions.append(s.count_actions(game))
        game.random_ply()
        if game.is_terminal():
            break
    game.render()
    print(f'Player {game.reward % 3} won')
    print(f'Game length: {game.plies}')
    print(f'Min number of actions: {min(num_actions)}')
    print(f'Max number of actions: {max(num_actions)}')
    print(f'Mean number of actions: {sum(num_actions)/len(num_actions)}')



