'''
A bitboard implementation of the board game 'Shōbu'.
'''
from random import choice

DIRECTIONS = {'N':5, 'E':-1, 'S':-5, 'W':1, 'NE':4, 'NW': 6, 'SE': -6, 'SW': -4}
BITMASK = 0b01111011110111101111 # bitmask deletes bits that move off the board
MAP = { # (p,i): (p, ~i), (~p, i)
    (0,0) : ( (0,1), (1,0) ),
    (0,1) : ( (0,0), (1,1) ),
    (1,0) : ( (0,0), (1,1) ),
    (1,1) : ( (0,1), (1,0) )
    }
START_BOARD = [0b1111, 0b1111000000000000000] # index 1 corresponds to player 1

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
    mailbox = [ bits & (1 << n) and 1  for n in range(20) ]
    mailbox_2D = [ mailbox[ 5*row : 5*row+5 ] for row in range(4) ]
    print(*mailbox_2D, sep='\n', end=f'\n{message}\n\n')
    return

def dilate(x, direction, distance):
    for _ in range(distance):
        x |= bitshift(x,direction,1)
    return x

def erode(x, direction, distance):
    for _ in range(distance):
        x &= bitshift(x,direction,1)
    return x

def f(a:int, b:int, c:int):
    'Testing legal aggro permutations'
    p2 = b & 1
    p3 = c & 1
    # not pushing more than one stone or your own stones
    F1 = not( (a and b) or p2 ) # b can be replaced with o2 here
    F2 = not( (b or c) and (a or b or p3) and (a or c or p2) )
    return F1, F2

class State:
    'Object containing all information required to uniquely define a Shōbu game state.'
    def __init__(self, player: bool = True, side1: list[list[int]] = [START_BOARD, START_BOARD], side2: list[list[int]] = [START_BOARD, START_BOARD]) -> None:
        self.player = player # player 1 = True, player 2 = False
        self.boards = [side2, side1] # index 1 corresponds to player 1

    def render(self, message: str = '') -> None:
        'Print the current game state.'
        print(f'----------{message}')
        for i in 0,1:
            left = self.mailbox(self.boards[i][0], i^0)
            right = self.mailbox(self.boards[i][1], i^1)
            for row in range(4):
                print(left[row]+right[row])
        return
    
    def mailbox(self, bits: list[int], type: bool):
        'Transform the bitboards into a 2D mailbox representation. Type refers to whether the board is light = 0 or dark = 1.'
        if type:
            colour = '\033[34m' # blue
        else:
            colour = '\033[39m' # white
        mailbox_2D = 4 * [colour]
        for row in range(4):
            for n in range(5*row, 5*row+4):
                mailbox_2D[row] += str( (bits[1] & (1 << n) and 1) + 2 * (bits[0] & (1 << n) and 1) )
        return mailbox_2D
    
    def is_game_over(self) -> int:
        '''The first player to lose all their stones on any board loses the game. 
        Return values 1 = player 1 win, -1 = player 2 win and 0 = game not over.'''
        if not self.boards[0][0][0] or not self.boards[0][1][0]:
            return 1 # player 1 wins
        elif not self.boards[1][0][1] or not self.boards[1][1][1]: 
            return -1 # player 2 wins
        return 0 # game not over
    
    def move(self, board1: list[int], board2: list[int], end1: int, end2: int, direction: int, distance: int):
        'Return the boards after a legal move has been applied to them'
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
        path = dilate(start2, direction, distance)
        view(path, 'path')
        collision = path & board2[not self.player]
        if collision: 
            view(collision, 'collision')
            landing = bitshift(end2, direction, 1)
            view(board2[not self.player], 'before3')
            board2[not self.player] ^= (collision | landing)
            view(board2[not self.player], 'after3')
        return board1, board2
    
    def input_move(self) -> None:
        self.render()
        return

    @staticmethod
    def random_move(all_legal_moves: dict[ str, dict[int,int] ]) -> int:
        direction = choice( list(all_legal_moves) )
        distance = choice( list(all_legal_moves[direction]) )
        random_moves = split( all_legal_moves[direction][distance] )
        return choice(random_moves)
    
    def all_legal_moves(self) -> dict:
        'Returns { boards? : { direction : { distance : moves } } }'
        all_legals = {}
        for i in False,True:
            passive_board = self.boards[self.player][i]
            for j in (self.player, not i), (not self.player, i) :
                aggro_board = self.boards[j[0]][j[1]]
                print((self.player, i), j)
                all_moves = self.legal_moves(passive_board, aggro_board)
                if all_moves:
                    all_legals[((self.player, i), j)] = all_moves
        return all_legals

    def legal_moves(self, passive_board: list[int], aggro_board: list[int]) -> dict[str, dict[int, int]]:
        'Returns { direction : { distance : moves } } '
        moves = {}
        for key, direction in DIRECTIONS.items():
            legals = self.legals(direction, passive_board, aggro_board)
            if legals:
                moves[key] = legals
        return moves

    def legals(self, direction: int, passive_board: list[int], aggro_board: list[int]) -> dict[int,int]:
        'Returns { distance : moves } where moves are ending positions'
        legals = {}
        passive1, passive2 = self.legal_passives(passive_board, direction)
        aggro1, aggro2 = self.legal_aggros(aggro_board, direction)
        if passive1 and aggro1: # has legal moves at distance 1
            legals[1] = (passive1, aggro1)
            if passive2 and aggro2:
                legals[2] = (passive2, aggro2)
        return legals
    
    def legal_aggros(self, aggro_board: list[int], direction: int) -> tuple[int]:
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

    def legal_passives(self, passive_board: list[int], direction: int) -> tuple[int]:
        empty = ~(passive_board[self.player] | passive_board[not self.player])
        passive1 = bitshift(passive_board[self.player], direction, 1) & empty
        passive2 = bitshift(passive1, direction, 1) & empty
        return passive1, passive2

if __name__ == '__main__':
    game = State()
    game.boards = [ [ [457,11300], [3,80] ] , [ [270600,4166] , [270600,232609] ] ]
    game.render()
    game.player = not game.player
    direction = DIRECTIONS['SW']

    b1 = [270600,4166]
    b2 = [457,11300]

    a1, a2 = game.legal_aggros( b2 , direction)
    p1, p2 = game.legal_passives( b1 , direction)

    rand_a2 = choice(split(a2))
    rand_p2 = choice(split(p2))

    new_b1, new_b2 = game.move(b1, b2, rand_p2, rand_a2, direction, 2)
    game.boards = [ [ new_b2, [0,0] ] , [ new_b1 , [0,0] ] ]
    game.render()
    pass