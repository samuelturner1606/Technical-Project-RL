'''
A fast and lightweight bitboard board game engine implementation of Shōbu.
'''
from random import choice, shuffle

DIRECTIONS = {'N':6, 'E':-1, 'S':-6, 'W':1, 'NE':5, 'NW':7, 'SE':-7, 'SW':-5}
BITMASK = 0b1111001111001111001111 # bitmask deletes bits that move off the board
COMBOS: tuple[tuple[tuple[tuple[int]]]] = ( # board indices: passive board (p,i) -> aggro boards (p,~i) or (~p,i)
    (((0,0),(0,1)), ((0,0),(1,0)), ((0,1),(0,0)), ((0,1),(1,1))),
    (((1,0),(0,0)), ((1,0),(1,1)), ((1,1),(0,1)), ((1,1),(1,0))) )
COMBOS2: tuple[dict[str,tuple[tuple[int]]]] = ( # more user friendly mapping
    {'01':((0,0),(0,1)), '02':((0,0),(1,0)), '10':((0,1),(0,0)), '13':((0,1),(1,1))},
    {'20':((1,0),(0,0)), '23':((1,0),(1,1)), '31':((1,1),(0,1)), '32':((1,1),(1,0))} )

def bitshift(bits: int, direction: int, distance: int) -> int:
    shift = direction * distance
    if shift > 0:
        return (bits >> shift) & BITMASK
    else:
        return (bits << -shift) & BITMASK

def split(legal_moves: int) -> list[int]:
    'Split `legal moves` into invidual moves by the on-bits.'
    seperate_moves = []
    while legal_moves:
        seperate_moves.append(legal_moves & -legal_moves) # get least significant on-bit
        legal_moves &= legal_moves-1 # remove least significant on-bit
    return seperate_moves

def view(bits: int, message: str = '') -> None:
    'For debugging.'
    mailbox = [ bits & (1 << n) and 1  for n in range(24) ]
    mailbox_2D = [ mailbox[ 6*row : 6*row+6 ] for row in range(4) ]
    print(*mailbox_2D, sep='\n', end=f'\n{message}\n\n')
    return

def mailbox(bitboards: list[int], type: bool) -> list[list[int]]:
    'Transform the bitboards into a 2D mailbox representation. Type refers to whether the board is light = 0 or dark = 1.'
    if type:
        colour = '\033[34m' # blue
    else:
        colour = '\033[39m' # white
    mailbox_2D = 4 * [colour]
    for row in range(4):
        for n in range(6*row, 6*row+4):
            mailbox_2D[row] += str( (bitboards[1] & (1 << n) and 1) + 2 * (bitboards[0] & (1 << n) and 1) )
    return mailbox_2D

def get_choice(choices: list[str], prompt: str = ''):
    'General user input handling function.'
    choice = ''
    while choice not in choices:
        choice = input(f"Choose one {prompt} from [{', '.join(choices)}]:\n")
    return choice

class State:
    'Object containing all information required to uniquely define a Shōbu game state.'
    def __init__(self, player: bool = True, boards: list[list[list[int]]] = [[[15,3932160], [15,3932160]],[[15,3932160], [15,3932160]]]) -> None:
        self.player = player # player 1 = True, player 2 = False
        self.boards = boards # index 1 corresponds to player 1
        self.reward: int = 0

    def render(self, message: str = '') -> None:
        'Print the current game state.'
        print(f'--------{str(message)}')
        for i in 0,1:
            left = mailbox(self.boards[i][0], i^0)
            right = mailbox(self.boards[i][1], i^1)
            for row in range(4):
                print(left[row]+right[row])
        return
    
    def is_terminal(self) -> int:
        '''The first player to lose all their stones on any board loses the game. 
        Return values 1 = player 1 win, -1 = player 2 win and 0 = game not over.'''
        if not self.reward: # if not already terminal from there being no legal moves
            for side in self.boards:
                for board in side:
                    if not board[0]:
                        self.reward = 1 # player 1 wins
                        break 
                    elif not board[1]:
                        self.reward = -1 # player 2 wins
                        break 
        return self.reward 
    
    def make_move(self, passive_board: list[int], aggro_board: list[int], passive_end: int, aggro_end: int, direction: int, distance: int) -> None:
        'Update the two boards inplace with the legal passive and aggro moves.'
        passive_start = bitshift(passive_end, -direction, distance)
        passive_board[self.player] ^= (passive_start | passive_end)
        aggro_start = bitshift(aggro_end, -direction, distance)
        aggro_board[self.player] ^= (aggro_start | aggro_end)
        path = aggro_end | bitshift(aggro_end, -direction, 1)
        collision = path & aggro_board[not self.player]
        if collision: 
            landing = bitshift(aggro_end, direction, 1)
            aggro_board[not self.player] ^= (collision | landing)
        return
    
    def all_legals(self) -> dict[tuple[tuple[int]],dict[str,dict[int,tuple]]]:
        '''Find all legal moves for all board combinations.\n
        Returns { boards: { direction : { distance : (passive moves, aggro moves) } } }\n
        Where boards are the indice tuples and moves are ending positions'''
        b = self.board_legals()
        output = {}
        for p, a in list(COMBOS[self.player]):
            passive_board = self.boards[p[0]][p[1]]
            aggro_board = self.boards[a[0]][a[1]]
            directions = {}
            for key in DIRECTIONS.keys():
                distances = {}
                passives = b[tuple(passive_board)][key][0]
                aggros = b[tuple(aggro_board)][key][1]
                if passives[0] and aggros[0]: # has legal moves at distance 1
                    distances[1] = (passives[0], aggros[0])
                    if passives[1] and aggros[1]: # has legal moves at distance 2
                        distances[2] = (passives[1], aggros[1])
                    directions[key] = distances
            if directions:
                output[(p, a)] = directions
        if not output: # if no legal moves
            self.render(' No legal moves')
            if self.player == 1: 
                self.reward = -1 # player 1 loses since they cannot move
            else:
                self.reward = 1
        return output
    
    def board_legals(self) -> dict[tuple,dict[int,tuple]]:
        '''Find all legal moves, in all directions, on all boards not including board combos.\n
        Returns { board : { direction : (passive moves, aggro moves) } }'''
        output = {}
        for side in self.boards:
            for board in side:
                if tuple(board) not in output:
                    temp = {}
                    for key, d in DIRECTIONS.items():
                        temp[key] = (self.legal_passives(board,d), self.legal_aggros(board,d))
                    output[tuple(board)] = temp
        return output

    def legal_passives(self, passive_board: list[int], direction: int) -> tuple[int]:
        'Find the ending squares of all legal passive moves, for all distances, in a direction, on a given board.'
        empty = ~(passive_board[self.player] | passive_board[not self.player])
        passive1 = bitshift(passive_board[self.player], direction, 1) & empty
        passive2 = bitshift(passive1, direction, 1) & empty
        return passive1, passive2
    
    def legal_aggros(self, aggro_board: list[int], direction: int) -> tuple[int]:
        'Find the ending squares of all legal aggro moves, for all distances, in a direction, on a given board.'
        p2 = aggro_board[self.player]
        p3 = bitshift(p2, direction, 1)
        x2 = p2 | aggro_board[not self.player]
        x1 = bitshift(x2, -direction, 1)
        x3 = bitshift(x2, direction, 1)
        # not pushing more than one stone or your own stones
        legal_aggro1 = ~( (x1 & x2) | p2 )
        aggro1 = bitshift(p2, direction, 1) & legal_aggro1 
        legal_aggro2 = ~( (x2|x3) & (x1|x2|p3) & (x1|x3|p2) )
        aggro2 = bitshift(p2, direction, 2) & legal_aggro2
        return aggro1, aggro2

    def random_ply(self) -> object:
        'Randomly return a legal successor state. Faster than choosing from all legal moves.'
        boards = self.boards.copy()
        combos = list(COMBOS[self.player])
        while combos:
            shuffle(combos)
            p, a = combos.pop()
            p_board = boards[p[0]][p[1]]
            a_board = boards[a[0]][a[1]]
            directions = list(DIRECTIONS.values())
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
                        random_p = choice(split(p1))
                        random_a = choice(split(a1))
                        self.make_move(p_board, a_board, random_p, random_a, direc, dist)
                        return State(not self.player, boards)
                    elif dist == 2 and p2 and a2:
                        random_p = choice(split(p2))
                        random_a = choice(split(a2))
                        self.make_move(p_board, a_board, random_p, random_a, direc, dist)
                        return State(not self.player, boards)
        self.render(' No legal random moves')
        if self.player == 1: 
            self.reward = -1 # player 1 loses since they cannot move
        else:
            self.reward = 1
        return
    
    def human_ply(self) -> None:
        'Ask the user for a legal move and update the state with it.'
        legals = self.all_legals()
        if legals:
            combos = COMBOS2[self.player]
            combo = combos[get_choice(list(combos), 'passive and aggro board combo')]
            p_board = self.boards[combo[0][0]][combo[0][1]]
            a_board = self.boards[combo[1][0]][combo[1][1]]
            direc = get_choice(list(legals[combo]), 'direction')
            dist = int(get_choice([ str(i) for i in list(legals[combo][direc]) ], 'distance'))
            passives, aggros = legals[combo][direc][dist]
            end1 = 1 << int(get_choice([str(n.bit_length()-1) for n in split(passives)],'passive move ending square'))
            end2 = 1 << int(get_choice([str(n.bit_length()-1) for n in split(aggros)],'aggressive move ending square'))
            self.make_move(p_board, a_board, end1, end2, DIRECTIONS[direc], dist)
            self.player = not self.player
        return

def count_actions(state: State):
    'Count the number of legal actions in the current state.'
    count = 0
    legals = state.all_legals()
    for combo in legals:
        for direction in legals[combo]:
            for distance in legals[combo][direction]:
                p, a = legals[combo][direction][distance]
                p_count = len(split(p))
                a_count = len(split(a))
                count += (p_count * a_count)
    return count

if __name__ == '__main__':
    game = State(boards=[[[4,266240],[794624,8]],[[256,526],[8328,786432]]])
    game.render()
    game.all_legals()
    game.random_ply()
    pass