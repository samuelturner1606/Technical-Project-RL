'''
Board game engine implementation of Shōbu using bitwise operators.
### Details
 - In many places 1-x (or 1^x) is used instead of ~x or not x in order to get the desired output.
'''
from random import choice, shuffle
import numpy as np

COMBOS = ( # mapping of passive to aggro board combinations
    { 0: (1, 2), 1: (0, 3)}, # player 2
    { 2: (0, 3), 3: (1, 2)} )# player 1
SHIFT: dict[str,tuple[int]] = {'N':(-1,0),'NE':(-1,1),'E':(0,1),'SE':(1,1),'S':(1,0),'SW':(1,-1),'W':(0,-1),'NW':(-1,-1)}
QUADRANT = ( # Get an object that will slice an nx8x8 ndarray into nx4x4, via the indices 0 to 3.
    (slice(None), slice(0,4), slice(0,4)),
    (slice(None), slice(0,4), slice(4,8)),
    (slice(None), slice(4,8), slice(0,4)),
    (slice(None), slice(4,8), slice(4,8)) )

def negate(offset: tuple[int]) -> tuple[int]:
    'Returns the shift `offset` in the opposite direction.'
    return tuple(map((-1).__mul__, offset))

def get_choice(choices: list[str], prompt: str = ''):
    'General user input handling function.'
    choice = ''
    while choice not in choices:
        choice = input(f"Choose one {prompt} from [{', '.join(choices)}]:\n")
    return choice

def count_actions(boards):
    'Count the number of legal actions in the current boards.'
    pass

class State2:
    'Object containing all information required to uniquely define a Shōbu game boards.'
    def __init__(self) -> None:
        self.done:bool = False
        self.reward:int = 0
        self.plies:int = 0

    def is_terminal(self) -> bool:
        return

    def random_ply(self) -> None:
        'Randomly play a legal move. Faster than choosing from all legal moves.'
        return
    
    def human_ply(self) -> None:
        'Ask the user for a legal move and update the boards with it.'
        return

class State:
    '''4 Shōbu 4x4 boards.
    @param boards: 2x8x8 np.ndarray or None
    @param player: 1 for player1, 0 for player2.'''
    def __init__(self, boards: np.ndarray = None, player: int = 1) -> None:
        assert player in (0,1), 'Player parameter incorrect.'
        self.player = player
        if boards is None: # creates default starting state
            self.boards: np.ndarray = np.zeros(shape=(2,8,8), dtype=np.uint8)
            self.boards[1, -1:0:-4, :] = 1 # player 1 pieces
            self.boards[0, 0:-1:4, :] = 1 # player 2 pieces
        else:
            self.boards: np.ndarray = np.asarray(boards, dtype=np.uint8)
            assert self.boards.shape == (2,8,8), 'Board not the right shape.'
            
    def render(self) -> None:
        'Prints the current game boards.'
        m = (2*self.boards[0] + self.boards[1]).astype(str)
        m = np.insert(m, 4, 8*['|'], 1)
        m = np.insert(m, 4, 9*['-'], 0)
        txt: str = '\n ' + np.array_str(m)
        txt = txt.replace('\'','')
        txt = txt.replace('[','')
        txt = txt.replace(']','')
        out = txt.splitlines(True)
        return print(*out) 

    def shift(self, array: np.ndarray, offset: tuple) -> np.ndarray:
        'Returns a copy of an `array` shifted (without rollover) by `offset`.'
        assert len(offset) == array.ndim, 'Number of dimensions do not match.'
        new_array = np.empty_like(array)
        def slice1(o):
            return slice(o, None) if o >= 0 else slice(0, o)
        new_array[tuple(slice1(o) for o in offset)] = (array[tuple(slice1(-o) for o in offset)])
        for axis, o in enumerate(offset):
            new_array[(slice(None),) * axis + (slice(0, o) if o >= 0 else slice(o, None),)] = 0
        return new_array

    def legal_passives(self, p_board: np.ndarray, direction: str) -> tuple[np.ndarray,np.ndarray]:
        '''Find the ending squares of all legal passive moves, for all distances, in a direction, on a given board.
        @param p_board: A 2x4x4 ndarray representing the board to play a passive move on.
        @param direction: Compass direction.
        @return passive1: A 2x4x4 ndarray with a legal passive move, of distance 1, applied to all pieces of the current player.
        @return passive2: A 2x4x4 ndarray with a legal passive move, of distance 2, applied to all pieces of the current player.'''
        assert p_board.shape == (2,4,4), 'Passive board not the right shape.'
        player = p_board[self.player]
        opponent = p_board[1 - self.player]
        empty = 1 - (player | opponent)
        offset1 = SHIFT[direction]
        passive1 = self.shift(player, offset1) & empty
        passive2 = self.shift(passive1, offset1) & empty
        return passive1, passive2
    
    def legal_aggros(self, a_board: np.ndarray, direction: str) -> tuple[np.ndarray,np.ndarray]:
        '''Find the ending squares of all legal aggressive moves, for all distances, in a direction, on a given board.
        @param a_board: A 2x4x4 ndarray representing the board to play a aggressive move on.
        @param direction: Compass direction.
        @return aggro1: A 2x4x4 ndarray with a legal aggressive move, of distance 1, applied to all pieces of the current player.
        @return aggro2: A 2x4x4 ndarray with a legal aggressive move, of distance 2, applied to all pieces of the current player.
        '''
        assert a_board.shape == (2,4,4), 'Aggressive board not the right shape.'
        offset1 = SHIFT[direction]
        p2 = a_board[self.player]
        p3 = self.shift(p2, offset1)
        x2 = p2 | a_board[1 - self.player]
        x1 = self.shift(x2, negate(offset1))
        x3 = self.shift(x2, offset1)
        # not pushing more than one stone or your own stones
        legal_aggro1 = 1 - ( (x1 & x2) | p2 )
        aggro1 = p3 & legal_aggro1 
        legal_aggro2 = 1 - ( (x2|x3) & (x1|x2|p3) & (x1|x3|p2) )
        aggro2 = self.shift(p3, offset1) & legal_aggro2
        return aggro1, aggro2

    def legals(self) -> np.ndarray:
        '''Find all legal moves on all boards, in all directions and distances.
        @returns output: 32x8x8 ndarray, where for index (i, j, k):
        - i%2 indicates the move distance
        - i//4 indicates the compass direction of the move
        - (j,k) indicates the ending square of a move'''
        output = []
        board = [ self.boards[QUADRANT[i]] for i in range(len(QUADRANT)) ]
        for direction in SHIFT:
            block = np.zeros(shape=(4,8,8), dtype=np.uint8)
            for p in COMBOS[self.player]:
                passives1, passives2 = self.legal_passives(board[p], direction)
                if passives1.any():
                    block[0:2][QUADRANT[p]] = np.stack((passives1,passives2))
                    for a in COMBOS[self.player][p]:
                        aggros1, aggros2 = self.legal_aggros(board[a], direction)
                        if not passives2.any():
                            aggros2 = np.zeros_like(aggros2) # can't play aggro move if couldn't play passive
                        block[2:4][QUADRANT[a]] = np.stack((aggros1,aggros2))
            output.append(block)
        return np.concatenate(output)

    def make_move(self, p_end: tuple[int], a_end: tuple[int], direction: str, distance: int) -> np.ndarray:
        '''Returns a copy of the boards with (an assumed) legal passive and aggressive move having been applied.
        @param p_end: (x_1, y_1) coordinates representing the ending square of the legal passive move
        @param a_end: (x_2, y_2) coordinates representing the ending square of the legal aggressive move
        @returns new_boards: 2x8x8 ndarray'''
        a, b = SHIFT[direction]; offset1 = (0, a, b)
        neg_offset = negate(tuple(map((distance).__mul__, offset1)))
        new_boards: np.ndarray = self.boards.copy()
        # turn on ending coordinates in 2x8x8 zeros ndarray
        endings = np.zeros_like(new_boards)
        x1, y1 = p_end; x2, y2 = a_end
        endings[self.player, y1, x1] = 1
        endings[:, y2, x2] = 1
        startings = self.shift(endings, neg_offset)
        new_boards[self.player] ^= (startings | endings)[self.player] # update player pieces
        # determine if collision with opponent piece
        endings = endings[1 - self.player]
        path = endings | self.shift(endings, negate((a,b)))
        collision: np.ndarray = path & new_boards[1 - self.player]
        if collision.any(): 
            landing = self.shift(endings, (a,b))
            # mask out when landing on other boards
            mask = np.zeros_like(landing)
            x2 = 4*(x2//4); y2 = 4*(y2//4)
            mask[slice(y2,y2+4), slice(x2,x2+4)] = 1
            landing &= mask
            new_boards[1 - self.player] ^= (collision | landing) # update opponent pieces
        assert new_boards.shape == (2,8,8)
        return new_boards

if __name__ == '__main__':
    #a = np.random.randint(2, size=(2,8,8), dtype=np.uint8)
    #b1 = a[1,:,:]
    #b2 = a[0,:,:]
    #a[1,:,:] = ( b2 & b1) ^ b1
    a = [[[1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]],
    [[0, 0, 1, 1, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 1, 0, 0]]]
    a = np.asarray(a, dtype=np.uint8)
    game = State(a)
    game.render()
    #x = game.legals()
    x = game.make_move((1,0), (6,1), 'N', 2)
    game.boards = x
    game.render()

pass