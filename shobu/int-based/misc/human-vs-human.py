import shobu as s

if __name__ == '__main__':
    print('\nThe four boards are labelled as: \n0 1\n2 3\n\nEach board square is labelled as: \n0  1  2  3\n6  7  8  9\n12 13 14 15\n18 19 20 21\n')
    game = s.State()
    while True:
        game.render(f" Player {2 - game.player}'s turn")
        game.human_ply()
        if game.is_terminal():
            break
    game.render()
    print(f'Player {game.reward % 3} won')