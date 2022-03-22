'''
A small version of the board game Shōbu played on only two boards.
'''
from time import sleep
import shobu as s
from random import choice, shuffle

ALT_COMBOS: tuple[tuple[tuple[tuple[int]]]] = ( # board indices
    (((0,0),(1,0))),
    (((1,0),(0,0))) )

class Small_state(s.State):
    'Object containing all information required to uniquely define a Shōbu game state.'
    def __init__(self) -> None:
        super().__init__()
        self.boards = [[[15,3932160]],[[15,3932160]]]

    def render(self, message: str = '') -> None:
        'Print the current game state.'
        print(f'---- {str(message)}')
        for i in 0,1:
            side = s.mailbox(self.boards[i][0], i^0)
            side[-1] += '\033[39m'
            for row in range(4):
                print(side[row])
        return

    def random_ply(self):
        'Randomly return a legal successor state. Faster than choosing from all legal moves.'
        #boards = self.boards.copy()
        p, a = ALT_COMBOS[self.player]
        p_board = self.boards[p[0]][p[1]]
        a_board = self.boards[a[0]][a[1]]
        directions = list(s.DIRECTIONS.values())
        while directions:
            shuffle(directions)
            direc = directions.pop()
            p1, p2 = self.legal_passives(p_board, direc)
            a1, a2 = self.legal_aggros(a_board, direc)
            distances = [1,2]
            while distances:
                shuffle(distances)
                dist = distances.pop()
                if dist == 1 and p1 and a1:
                    random_p = choice(s.split(p1))
                    random_a = choice(s.split(a1))
                    self.make_move(p_board, a_board, random_p, random_a, direc, dist)
                    return
                elif dist == 2 and p2 and a2:
                    random_p = choice(s.split(p2))
                    random_a = choice(s.split(a2))
                    self.make_move(p_board, a_board, random_p, random_a, direc, dist)
                    return
        self.render('No legal random moves')
        self.done = True
        if self.player == 1: 
            self.reward = -1 # player 1 loses since they cannot move
        else:
            self.reward = 1
        self.player=not self.player
        return
    
    def human_ply(self) -> None:
        'Ask the user for a legal move and update the state with it.'
        legals = self.all_legals()
        if legals:
            p, a = ALT_COMBOS[self.player]
            p_board = self.boards[p[0]][p[1]]
            a_board = self.boards[a[0]][a[1]]
            direc = s.get_choice(list(legals), 'direction')
            dist = int(s.get_choice([ str(i) for i in list(legals[direc]) ], 'distance'))
            passives, aggros = legals[direc][dist]
            end1 = 1 << int(s.get_choice([str(n.bit_length()-1) for n in s.split(passives)],'passive move ending square'))
            end2 = 1 << int(s.get_choice([str(n.bit_length()-1) for n in s.split(aggros)],'aggressive move ending square'))
            self.make_move(p_board, a_board, end1, end2, s.DIRECTIONS[direc], dist)
            self.player = not self.player
        return
    
    def all_legals(self) -> dict[tuple[tuple[int]],dict[str,dict[int,tuple]]]:
        '''Find all legal moves for all board combinations.\n
        Returns { direction : { distance : (passive moves, aggro moves) } } \n
        Where boards are the indice tuples and moves are ending positions'''
        b = self.board_legals()
        p, a = ALT_COMBOS[self.player]
        passive_board = self.boards[p[0]][p[1]]
        aggro_board = self.boards[a[0]][a[1]]
        directions = {}
        for key in s.DIRECTIONS.keys():
            distances = {}
            passives = b[tuple(passive_board)][key][0]
            aggros = b[tuple(aggro_board)][key][1]
            if passives[0] and aggros[0]: # has legal moves at distance 1
                distances[1] = (passives[0], aggros[0])
                if passives[1] and aggros[1]: # has legal moves at distance 2
                    distances[2] = (passives[1], aggros[1])
                directions[key] = distances
        if not directions: # if no legal moves
            self.render('No legal moves')
            self.done = True
            if self.player == 1: 
                self.reward = -1 # player 1 loses since they cannot move
            else:
                self.reward = 1
        return directions

def count_actions(state: Small_state):
    'Count the number of legal actions in the current state.'
    count = 0
    legals = state.all_legals()
    for direction in legals:
        for distance in legals[direction]:
            p, a = legals[direction][distance]
            p_count = len(s.split(p))
            a_count = len(s.split(a))
            count += (p_count * a_count)
    return count


if __name__ == '__main__':
    print('random vs random:')
    game = Small_state()
    print(f'Number of starting action: {count_actions(game)}')
    while True:
        #sleep(0.1)
        game.render()
        game.random_ply()
        if game.is_terminal():
            break
    game.render()
    print(f'Player {game.reward % 3} won in {game.plies} plies.')

    print('human vs human')
    game = Small_state()
    while True:
        sleep(0.1)
        game.render()
        game.human_ply()
        if game.is_terminal():
            break
    game.render()
    print(f'Player {game.reward % 3} won in {game.plies} plies.')