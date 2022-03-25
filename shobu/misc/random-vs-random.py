from time import sleep
import shobu as s

if __name__ == '__main__':
    game = s.State()
    while True:
        sleep(0.1)
        game.render()
        game.random_ply()
        if game.is_terminal():
            break
    game.render()
    print(f'Player {game.reward % 3} won')