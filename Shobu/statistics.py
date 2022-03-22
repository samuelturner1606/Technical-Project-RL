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
        game = game.random_ply()
        reward = game.is_terminal()
        if reward:
            break
    game.render()
    print(f'Player {reward % 3} won')



