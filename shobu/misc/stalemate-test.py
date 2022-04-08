'''Confirming existance of a stalemate scenario from which it seems to be impossible to ever reach a terminal state.
Warranting a max game length to be imposed.'''

from time import sleep
import shobu as s

if __name__ == '__main__':
    game = s.State(boards=[[[528384,8],[520,262144]],[[8,8192],[262144,520]]])
    while True:
        sleep(0.1)
        game.render()
        game.random_ply()
        if game.is_terminal():
            break
    game.render()
    print(f'Player {game.reward % 3} won')