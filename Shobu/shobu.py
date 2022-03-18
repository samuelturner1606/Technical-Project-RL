'''
A bitboard implementation of the board game 'Shōbu'.
'''
from random import choice

DIRECTIONS = {'N':6, 'E':-1, 'S':-6, 'W':1, 'NE':5, 'NW': 7, 'SE': -7, 'SW': -5}
BITMASK = 0b1111001111001111001111 # bitmask deletes bits that move off the board
MAP = { # (p,i): (p, ~i), (~p, i)
    (0,0) : ( (0,1), (1,0) ),
    (0,1) : ( (0,0), (1,1) ),
    (1,0) : ( (0,0), (1,1) ),
    (1,1) : ( (0,1), (1,0) )
    }
START_BOARD = [0b1111, 0b1111000000000000000000] # index 1 corresponds to player 1

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

def mailbox(bits: list[int], type: bool) -> list[list[int]]:
    'Transform the bitboards into a 2D mailbox representation. Type refers to whether the board is light = 0 or dark = 1.'
    if type:
        colour = '\033[34m' # blue
    else:
        colour = '\033[39m' # white
    mailbox_2D = 4 * [colour]
    for row in range(4):
        for n in range(6*row, 6*row+4):
            mailbox_2D[row] += str( (bits[1] & (1 << n) and 1) + 2 * (bits[0] & (1 << n) and 1) )
    return mailbox_2D

def _f(a:int, b:int, c:int):
    'Testing legal aggro permutations'
    p2 = b & 1
    p3 = c & 1
    # not pushing more than one stone or your own stones
    F1 = not( (a and b) or p2 ) # b can be replaced with o2 here
    F2 = not( (b or c) and (a or b or p3) and (a or c or p2) )
    return F1, F2

def rand_moves(all_legals: dict):
    'Return legal passive and aggro moves in a random board combo, direction and distance.'
    boards = choice(list(all_legals))
    direction = choice(list(all_legals[boards]))
    dist = choice(list(all_legals[boards][direction]))
    passives, aggros = all_legals[boards][direction][dist]
    return passives, aggros

class State:
    'Object containing all information required to uniquely define a Shōbu game state.'
    def __init__(self, player: bool = True, side1: list[list[int]] = [START_BOARD, START_BOARD], side2: list[list[int]] = [START_BOARD, START_BOARD]) -> None:
        self.player = player # player 1 = True, player 2 = False
        self.boards = [side2, side1] # index 1 corresponds to player 1

    def render(self, message: str = '') -> None:
        'Print the current game state.'
        print(f'--------{message}')
        for i in 0,1:
            left = mailbox(self.boards[i][0], i^0)
            right = mailbox(self.boards[i][1], i^1)
            for row in range(4):
                print(left[row]+right[row])
        return
    
    def is_game_over(self) -> int:
        '''The first player to lose all their stones on any board loses the game. 
        Return values 1 = player 1 win, -1 = player 2 win and 0 = game not over.'''
        if not self.boards[0][0][0] or not self.boards[0][1][0]:
            return 1 # player 1 wins
        elif not self.boards[1][0][1] or not self.boards[1][1][1]: 
            return -1 # player 2 wins
        return 0 # game not over
    
    def move(self, board1: list[int], board2: list[int], end1: int, end2: int, direction: int, distance: int):
        'Update the two boards with the legal moves'
        # passive move
        start1 = bitshift(end1, -direction, distance)
        view(start1, 'start1')
        view(board1[self.player], 'before1')
        board1[self.player] ^= (start1 | end1)
        view(board1[self.player], 'after1')
        # aggressive move
        start2 = bitshift(end2, -direction, distance)
        view(start2, 'start2')
        view(board2[self.player], 'before2')
        board2[self.player] ^= (start2 | end2)
        view(board2[self.player], 'after2')
        path = end2 | bitshift(end2, -direction, 1)
        view(path, 'path')
        collision = path & board2[not self.player]
        if collision: 
            view(collision, 'collision')
            landing = bitshift(end2, direction, 1)
            view(board2[not self.player], 'before3')
            board2[not self.player] ^= (collision | landing)
            view(board2[not self.player], 'after3')
        return board1, board2
    
    def all_legals(self) -> dict[tuple,dict[str,dict[int,tuple]]]:
        '''Find all legal moves for all board combinations.\n
        Returns { boards: { direction : { distance : (passive moves, aggro moves) } } }\n
        Where boards are the indice tuples and moves are ending positions'''
        b = self.board_moves()
        output = {}
        for i in False,True:
            passive_board = self.boards[self.player][i]
            for j in (self.player, not i), (not self.player, i) :
                aggro_board = self.boards[j[0]][j[1]]
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
                    output[((self.player,i), j)] = directions
        return output
    
    def board_moves(self) -> dict[tuple,dict[int,tuple]]:
        '''Find all legal moves, in all directions, on all boards.\n
        Returns { board : { direction : (passive moves, aggro moves) } }'''
        output = {}
        for side in self.boards:
            for board in side:
                temp = {}
                for key, d in DIRECTIONS.items():
                    temp[key] = (self.legal_passives(board,d), self.legal_aggros(board,d))
                if temp:
                    output[tuple(board)] = temp
        return output

    def legal_passives(self, passive_board: list[int], direction: int) -> tuple[int]:
        'Find the ending squares of all legal passive moves, for all distances, in a direction, on a given board'
        empty = ~(passive_board[self.player] | passive_board[not self.player])
        passive1 = bitshift(passive_board[self.player], direction, 1) & empty
        passive2 = bitshift(passive1, direction, 1) & empty
        return passive1, passive2
    
    def legal_aggros(self, aggro_board: list[int], direction: int) -> tuple[int]:
        'Find the ending squares of all legal aggro moves, for all distances, in a direction, on a given board'
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

if __name__ == '__main__':
    game = State()
    game.boards = [ [ [905,45124], [3,128] ] , [ [2130440,16518] , [2130440,1585473] ] ]
    game.render()
