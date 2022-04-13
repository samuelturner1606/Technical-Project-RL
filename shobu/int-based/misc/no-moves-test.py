'''Confirming existence of states without legal moves and viewing max number of legal moves.'''
import shobu as s
  
if __name__ == '__main__':
    game = s.State(boards=[[[4,266240],[794624,8]],[[256,526],[8328,786432]]])
    game.render()
    game.all_legals()
    game.random_ply()
    print(s.count_actions(game))

    game = s.State()
    game.boards = [[15,565376],[15,565376]],[[15,565376],[15,565376]]
    game.render()
    print(s.count_actions(game))