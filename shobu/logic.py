'''
Board game engine implementation of Shōbu using bitwise operators.
### Details
 - In many places 1-x (or 1^x) is used instead of ~x or not x in order to get the desired output.
'''
from random import choice, shuffle
import numpy as np

P2A = ( # mapping of passive to aggro board combinations
    { 0: (1, 2), 1: (0, 3)}, # player 2
    { 2: (0, 3), 3: (1, 2)} )# player 1
A2P = tuple( {aggro:passive for passive in side for aggro in side[passive]} for side in P2A )
DIRECTIONS = ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')
OFFSETS = ((-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)) # indices match directions
QUADRANT = ( # Get an object that will slice an nx8x8 ndarray into nx4x4, via the indices 0 to 3.
    (slice(None), slice(0,4), slice(0,4)),
    (slice(None), slice(0,4), slice(4,8)),
    (slice(None), slice(4,8), slice(0,4)),
    (slice(None), slice(4,8), slice(4,8)) )
AGGRO_MASK = np.zeros((32,8,8), dtype=np.uint8)
AGGRO_MASK[2::4] = 1; AGGRO_MASK[3::4] = 1 # used to mask legal actions


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

    def _shift(self, array: np.ndarray, offset: tuple[int]) -> np.ndarray:
        'Returns a copy of an `array` shifted (without rollover) by `offset`.'
        assert len(offset) == array.ndim, 'Number of dimensions do not match.'
        new_array = np.empty_like(array)
        def slice1(o):
            return slice(o, None) if o >= 0 else slice(0, o)
        new_array[tuple(slice1(o) for o in offset)] = (array[tuple(slice1(-o) for o in offset)])
        for axis, o in enumerate(offset):
            new_array[(slice(None),) * axis + (slice(0, o) if o >= 0 else slice(o, None),)] = 0
        return new_array

    def _legal_passives(self, p_board: np.ndarray, offset: tuple[int,int]) -> tuple[np.ndarray,np.ndarray]:
        '''Find the ending squares of all legal passive moves, for all distances, in a direction, on a given board.
        @param p_board: 2x4x4 ndarray representing the board to play a passive move on.
        @param offset: directional offset
        @return passive1: 2x4x4 ndarray with a legal passive move, of distance 1, applied to all pieces of the current player
        @return passive2: 2x4x4 ndarray with a legal passive move, of distance 2, applied to all pieces of the current player'''
        assert p_board.shape == (2,4,4), 'Passive board not the right shape.'
        player = p_board[self.player]
        opponent = p_board[1 - self.player]
        empty = 1 - (player | opponent)
        passive1 = self._shift(player, offset) & empty
        passive2 = self._shift(passive1, offset) & empty
        return passive1, passive2
    
    def _legal_aggros(self, a_board: np.ndarray, offset: tuple[int,int]) -> tuple[np.ndarray,np.ndarray]:
        '''Find the ending squares of all legal aggressive moves, for all distances, in a direction, on a given board.
        @param a_board: A 2x4x4 ndarray representing the board to play a aggressive move on.
        @param offset: Directional offset
        @return aggro1: A 2x4x4 ndarray with a legal aggressive move, of distance 1, applied to all pieces of the current player
        @return aggro2: A 2x4x4 ndarray with a legal aggressive move, of distance 2, applied to all pieces of the current player'''
        assert a_board.shape == (2,4,4), 'Aggressive board not the right shape.'
        p2 = a_board[self.player]
        p3 = self._shift(p2, offset)
        x2 = p2 | a_board[1 - self.player]
        x1 = self._shift(x2, self._negate(offset))
        x3 = self._shift(x2, offset)
        mask1 = 1 - ( (x1 & x2) | p2 ) 
        aggro1 = p3 & mask1 
        mask2 = 1 - ( (x2|x3) & (x1|x2|p3) & (x1|x3|p2) ) # not pushing more than one stone or your own stones
        aggro2 = self._shift(p3, offset) & mask2
        return aggro1, aggro2

    def legal_actions(self) -> np.ndarray:
        '''Find all legal actions on all boards, in all directions and distances.
        @return output: 32x8x8 ndarray, where for coordinates (z, y, x):
        - z % 2, indicates the move distance
        - z // 4, indicates the compass direction of the move
        - (z // 2) % 2, indicates if the moves was aggressive
        - (y, x) indicates the ending square of a move relative from the top-left corner of the boards

        Legal and illegal actions are given a value of 1 and 0 respectively in the `output`. The `output` can be used to directly mask the policy head of the neural network for legal actions.'''
        output = []
        board = [ self.boards[QUADRANT[i]] for i in range(4) ]
        for offset in OFFSETS:
            block = np.zeros(shape=(4,8,8), dtype=np.uint8)
            for p_board in P2A[self.player]:
                passives1, passives2 = self._legal_passives(board[p_board], offset)
                if passives1.any():
                    block[0:2][QUADRANT[p_board]] = np.stack((passives1,passives2))
                    for a_board in P2A[self.player][p_board]:
                        aggros1, aggros2 = self._legal_aggros(board[a_board], offset)
                        if not passives2.any(): # if couldn't play passive
                            aggros2 = np.zeros(aggros2.shape, aggros2.dtype) # can't play aggro move 
                        block[2:4][QUADRANT[a_board]] = np.stack((aggros1,aggros2))
            output.append(block)
        output = np.concatenate(output)
        assert output.shape == (32,8,8), 'Shape of legal actions is wrong'
        assert output.any(), 'No legal actions'
        return output

    def _action_to_move(self, z1:int, y1:int, x1:int, z2:int, y2:int, x2:int) -> np.ndarray:
        '''Returns a copy of the boards with (an assumed) legal passive and aggressive move having been applied.
        @param (z1, y1, x1): coordinates representing the ending square of the legal passive move
        @param (z2, y2, x2): coordinates representing the ending square of the legal aggressive move
        @return new_boards: 2x8x8 ndarray'''
        offset1, distance1 = z1//4, (z1%2)+1
        offset2, distance2 = z2//4, (z2%2)+1
        neg_offset = self._negate(tuple(map((distance1).__mul__, offset1)))
        new_boards: np.ndarray = self.boards.copy()
        # turn on ending coordinates in 2x8x8 zeros ndarray
        endings = np.zeros(new_boards.shape, new_boards.dtype)
        endings[self.player, y1, x1] = 1
        endings[:, y2, x2] = 1
        startings = self._shift(endings, neg_offset)
        new_boards[self.player] ^= (startings | endings)[self.player] # update player pieces
        # determine if collision with opponent piece
        endings = endings[1 - self.player]
        path = endings | self._shift(endings, self._negate(offset1))
        collision: np.ndarray = path & new_boards[1 - self.player]
        if collision.any(): 
            landing = self._shift(endings, offset1)
            # mask out when landing on other boards
            mask = np.zeros(landing.shape, landing.dtype)
            mask[QUADRANT[self._quadrant(x2,y2)][1:]] = 1 # quadrant of board
            landing &= mask
            new_boards[1 - self.player] ^= (collision | landing) # update opponent pieces
        assert new_boards.shape == (2,8,8)
        return new_boards

    def random_action(self, legal_actions: np.ndarray):
        '''Randomly select a passive and aggressive action from all legal actions.
        @return new_boards: 2x8x8 ndarray from `_action_to_move()`'''
        z2, y2, x2 = choice(np.argwhere(AGGRO_MASK & legal_actions).tolist())
        p_board = A2P[self.player][self._quadrant(x2,y2)]
        z1 = z2 - 2
        p_options = legal_actions[z1]
        p_mask = np.zeros(p_options.shape, p_options.dtype)[None,:]
        p_mask[QUADRANT[p_board]] = 1
        p_options &= p_mask[0]
        y1, x1 = choice(np.argwhere(p_options).tolist())
        return self._action_to_move(z1, y1, x1, z2, y2, x2)

    def human_action(self, legal_actions: np.ndarray):
        '''Select a passive and aggressive action from all legal actions.
        @return new_boards: 2x8x8 ndarray from `_action_to_move()`'''
        def get_choice(choices: list[str], prompt: str = '') -> str:
            'General user input handling function.'
            choice = ''
            while choice not in choices:
                choice = input(f"Choose one {prompt} from [{', '.join(choices)}]:\n")
            return choice
        A = np.split(legal_actions, 8, axis=0) # split legal actions by direction
        q = DIRECTIONS.index( get_choice([DIRECTIONS[i] for i in range(8) if A[i].any()], 'direction') ) # choose a direction
        B = A[q] 
        w = int(get_choice([ str(i-1) for i in [2,3] if B[i].any() ], 'distance') ) + 1 # choose a distance
        z2 = 4*q + w
        C = np.argwhere(B[w])
        a_coord = get_choice([ f'({x} {y})' for y, x in C], 'aggro move (x y) ending square') # choose aggro ending
        x2 = int(a_coord[1]); y2 = int(a_coord[3])
        z1 = z2 - 2
        p_options = legal_actions[z1]
        p_mask = np.zeros(p_options.shape, p_options.dtype)[None,:]
        p_board = A2P[self.player][self._quadrant(x2,y2)]
        p_mask[QUADRANT[p_board]] = 1
        p_options &= p_mask[0]
        D = np.argwhere(p_options)
        p_coord = get_choice([ f'({x} {y})' for y, x in D], 'passive move (x y) ending square') # choose passive ending
        x1 = int(p_coord[1]); y1 = int(p_coord[3])
        return self._action_to_move(z1, y1, x1, z2, y2, x2)

    @staticmethod
    def _negate(offset: tuple[int]) -> tuple[int]:
        'Returns the `offset` in the opposite direction.'
        return tuple(map((-1).__mul__, offset))

    @staticmethod
    def _quadrant(x: int, y: int) -> int:
        'Maps (x, y) coordinates to board quadrants 0 to 3.'
        return x//4 + 2*(y//4)

if __name__ == '__main__':
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
    for _ in range(20):
        game.render()
        l = game.legal_actions()
        game.boards = game.human_action(l)
        game.player = 1 - game.player
    pass
