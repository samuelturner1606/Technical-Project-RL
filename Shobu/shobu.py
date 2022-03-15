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
            for n in range(5*row, 5*row+5):
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
    
    def move(self, board1: list[int], board2: list[int], end_square: int, direction: int, distance: int):
        start_square = bitshift(end_square, -direction, distance)
        board1[self.player] ^= start_square|end_square
        board2[self.player] ^= start_square|end_square
        path = dilate(start_square,direction,distance)
        collision = path & board2[not self.player]
        if collision: 
            landing = bitshift(end_square, direction, 1)
            board2[not self.player] ^= collision|landing
        return
    
    def input_move(self) -> None:
        self.render()
        return

    def board_combos(self):
        for i in 0,1:
            passive_board = self.boards[self.player][i]
            for j in (self.player,not i), (not self.player,i) :
                aggro_board = self.boards[j[0]][j[1]]
        return

    def all_legal_moves(self, passive_board: list[int], aggro_board: list[int]) -> dict[str, dict[int, int]]:
        moves = {}
        for key, direction in DIRECTIONS.items():
            legals = self.legal_moves(direction, passive_board, aggro_board)
            if legals:
                moves[key] = legals
        return moves

    def legal_moves(self, direction: int, passive_board: list[int], aggro_board: list[int]) -> dict[int,int]:
        legals = {}
        legal_passive = ~(passive_board[self.player] | passive_board[not self.player])
        passive1 = bitshift(passive_board[self.player], direction, 1) & legal_passive
        if passive1: # can play a passive move at distance 1
            legal_aggro = ~erode(aggro_board[self.player] | aggro_board[not self.player], -direction, 1) & ~aggro_board[self.player]
            aggro1 = bitshift(aggro_board[self.player], direction, 1) & legal_aggro
            legal1 = passive1 & aggro1 
            if legal1: # has legal moves at distance 1
                legals[1] = legal1
                passive2 = bitshift(passive1, direction, 1) & legal_passive
                if passive2: # can play a passive move at distance 2
                    aggro2 = bitshift(aggro1, direction, 1) & legal_aggro
                    legal2 = passive2 & aggro2
                    if legal2:
                        legals[2] = legal2
        return legals
    
    @staticmethod
    def random_move(all_legal_moves: dict[ str, dict[int,int] ]) -> int:
        direction = choice( list(all_legal_moves) )
        distance = choice( list(all_legal_moves[direction]) )
        random_moves = split( all_legal_moves[direction][distance] )
        return choice(random_moves)

if __name__ == '__main__':
    game = State()
    '''
    while True:
        game.input_move()
        reward = game.is_game_over()
        if reward:
            print(f'Game over! reward = {reward}')
            break
        else:
            game.player = not game.player
    '''
