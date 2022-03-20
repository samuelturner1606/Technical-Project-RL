from time import sleep
import shobu as s

if __name__ == '__main__':
    game = s.State()
    while True:
        sleep(0.1)
        game.render()
        game = game.random_child()
        reward = game.is_terminal()
        if reward:
            break
    game.render()
    print(f'Player {reward % 3} won')