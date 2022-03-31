'''
Board game engine implementation of Shōbu using bitwise operators.
### Details
 - In many places 1-x (or 1^x) is used instead of ~x or not x in order to get the desired output.
'''
from random import choice, shuffle
import numpy as np

DI: dict[str,tuple[int]] = {'N':(-1,0),'NE':(-1,1),'E':(0,1),'SE':(1,1),'S':(1,0),'SW':(1,-1),'W':(0,-1),'NW':(-1,-1) }
SHIFT2 = {
    1: {
        'N':(-1,0),
        'NE':(-1,1),
        'E':(0,1),
        'SE':(1,1),
        'S':(1,0),
        'SW':(1,-1),
        'W':(0,-1),
        'NW':(-1,-1) }, 
    2: {
        'N':(-2,0),
        'NE':(-2,2),
        'E':(0,2),
        'SE':(2,2),
        'S':(2,0),
        'SW':(2,-2),
        'W':(0,-2),
        'NW':(-2,-2) }
}

COMBOS = ( # mapping of passive to aggro board combinations
    { 0: [1, 2], 1: [0, 3]}, # player 2
    { 2: [0, 3], 3: [1, 2]} # player 1
)

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
        '''The first player to lose all their stones on any board loses the game. 
        Reward values 1 = player 1 win, -1 = player 2 win and 0 = game not over.
        Returns self.done'''
        if self.plies >= 1000: # Max game length avoids theoretical stalemates
            self.done = True
            self.reward = 0
            self.render('Max game length reached')
        elif not self.done: # if not already terminal from there being no legal moves or max game length
            for side in self.boards:
                for board in side:
                    if not board[0]:
                        self.done = True
                        self.reward = 1 # player 1 wins
                        break 
                    elif not board[1]:
                        self.done = True
                        self.reward = -1 # player 2 wins
                        break 
        return self.done 
    
    def make_move(self, passive_board: list[int], aggro_board: list[int], passive_end: int, aggro_end: int, direction: int, distance: int) -> None:
        'Update the two boards inplace with the legal passive and aggro moves.'
        passive_start = bitshift(passive_end, -direction, distance)
        p ^= (passive_start | passive_end)
        aggro_start = bitshift(aggro_end, -direction, distance)
        aggro_board[self.player] ^= (aggro_start | aggro_end)
        path = aggro_end | bitshift(aggro_end, -direction, 1)
        collision = path & aggro_board[not self.player]
        if collision: 
            landing = bitshift(aggro_end, direction, 1)
            aggro_board[not self.player] ^= (collision | landing)
        self.plies += 1
        self.player = not self.player
        return
    
    def all_legals(self) -> dict[tuple[tuple[int]],dict[str,dict[int,tuple]]]:
        '''Find all legal moves for all board combinations.
        '''
        pass

    def random_ply(self) -> None:
        'Randomly play a legal move. Faster than choosing from all legal moves.'
        return
    
    def human_ply(self) -> None:
        'Ask the user for a legal move and update the boards with it.'
        return


class State:
    '4 Shōbu 4x4 boards.'
    def __init__(self, boards: np.ndarray = None, player: int = 1) -> None:
        self.player = player # player 1 = index 1, player 2 = index 0
        if boards is None:
            self.boards = np.zeros(shape=(2,8,8), dtype=np.uint8)
            self.boards[1, -1:0:-4, :] = 1 # player 1 pieces
            self.boards[0, 0:-1:4, :] = 1 # player 2 pieces
        else:
            self.boards = np.asarray(boards, dtype=np.uint8)
            assert self.boards.shape == (2,8,8), 'Board not the right shape.'
            
    def render(self) -> None:
        'Prints the current game boards.'
        return print(2*self.boards[0,:,:] + self.boards[1,:,:])
    
    def board(self, num: int) -> np.ndarray:
        'Get a 2x4x4 board from the 2x8x8 `self.boards`, via the numbers 0 to 3.'
        if num == 0:
            return self.boards[:, 0:4, 0:4]
        elif num == 1:
            return self.boards[:, 0:4, 4:8]
        elif num == 2:
            return self.boards[:, 4:8, 0:4]
        else:
            return self.boards[:, 4:8, 4:8]
    
    QUADRANT = ( # Get an object that will slice an nx8x8 array into nx4x4, via the numbers 0 to 3.
        (slice(None), slice(0,4), slice(0,4)),
        (slice(None), slice(0,4), slice(4,8)),
        (slice(None), slice(4,8), slice(0,4)),
        (slice(None), slice(4,8), slice(4,8))
        ) 
    def quadrant(self, array: np.ndarray, num: int):
        
        return

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

    def legal_passives(self, p_board: np.ndarray, direction: str) -> tuple[np.ndarray]:
        '''Find the ending squares of all legal passive moves, for all distances, in a direction, on a given board.
        @param p_board: A 2x4x4 ndarray representing the board to play a passive move on.
        @param direction: Compass direction.
        @return passive1: A 2x4x4 ndarray with a legal passive move, of distance 1, applied to all pieces of the current player.
        @return passive2: A 2x4x4 ndarray with a legal passive move, of distance 2, applied to all pieces of the current player.
        '''
        assert p_board.shape == (2,4,4), 'Passive board not the right shape.'
        player = p_board[self.player, :, :]
        opponent = p_board[1 - self.player, :, :]
        empty = 1 - (player | opponent)
        offset1 = DI[direction]
        passive1 = self.shift(player, offset1) & empty
        passive2 = self.shift(passive1, offset1) & empty
        return passive1, passive2
    
    def legal_aggros(self, a_board: np.ndarray, direction: str) -> tuple[np.ndarray]:
        '''Find the ending squares of all legal aggressive moves, for all distances, in a direction, on a given board.
        @param a_board: A 2x4x4 ndarray representing the board to play a aggressive move on.
        @param direction: Compass direction.
        @return aggro1: A 2x4x4 ndarray with a legal aggressive move, of distance 1, applied to all pieces of the current player.
        @return aggro2: A 2x4x4 ndarray with a legal aggressive move, of distance 2, applied to all pieces of the current player.
        '''
        assert a_board.shape == (2,4,4), 'Aggressive board not the right shape.'
        offset1 = DI[direction]
        p2 = a_board[self.player, :, :]
        p3 = self.shift(p2, offset1)
        x2 = p2 | a_board[1 - self.player, :, :]
        x1 = self.shift(x2, negate(offset1))
        x3 = self.shift(x2, offset1)
        # not pushing more than one stone or your own stones
        legal_aggro1 = 1 - ( (x1 & x2) | p2 )
        aggro1 = p3 & legal_aggro1 
        legal_aggro2 = 1 - ( (x2|x3) & (x1|x2|p3) & (x1|x3|p2) )
        aggro2 = self.shift(p3, offset1) & legal_aggro2
        return aggro1, aggro2

    def legals(self) -> np.ndarray:
        '''Find all legal moves, in all directions, on all boards not including board combos.'''
        output = np.zeros(shape=(32,8,8), dtype=np.uint8)
        boards = [self.board(i).copy() for i in range(4)]
        #for index, direction in enumerate(DI):
            #pass
        direction = 'N'
        index = 0
        for b in COMBOS[self.player]:
            p1, p2 = self.legal_passives(boards[b], direction)
        return output
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
    game.player = 1 - game.player
    game.legals()

pass
