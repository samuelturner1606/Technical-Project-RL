from dataclasses import dataclass
import mailbox

@dataclass 
class State():
    '''Representation of the board game in binary'''
    bitboard1: int = 0
    bitboard2: int = 0

    def possible_moves(self) -> int:
        '''Find all possible next moves given the current state'''
        overlap = self.bitboard1 | self.bitboard2 # find all piece positions
        shifted_up = (overlap << (COLS+1)) | BOTTOM_BITMASK # bitshift up a row and fill in empty bottom row
        return shifted_up & ~overlap & FULL_BITMASK
    
    def to_mailbox(self) -> str:
        '''Convert the current state into mailbox representation'''
        board1, board2  = self.to_binary(self.bitboard1), self.to_binary(self.bitboard2)
        mailbox = [str( int(board2[bit]+board1[bit],2) ) for bit in range(BYTE_LENGTH)]
        return ''.join(mailbox)
    
    @staticmethod
    def to_binary(bitboard: int) -> str:
        '''Convert a bitboard integer into a binary string'''
        return bin(bitboard)[2:].zfill(BYTE_LENGTH)

class Connect4():
    '''A Connect-4 reinforcement learning environment class'''
    def __init__(self):
        self.state = State()
        self.player = 1 # player 1 = 1, player 2 = -1
        print('\nNew game started')
        self.render()
        
    def step(self, action: int):
        '''Applies an action to the environement and gets the reward'''
        if self.player == 1:
            self.state.bitboard1 |= action
        else:
            self.state.bitboard2 |= action
        self.render()
        reward, done = self.is_game_over()
        self.player *= -1 # change player
        return self.state, reward, done

    def render(self) -> None:
        '''Processes and prints the current state to the terminal'''
        mailbox = self.state.to_mailbox()
        #b = mailbox.replace('0','.').replace('1','x').replace('2','o')
        for row in self.make2D(mailbox):
            print(row)

    def is_game_over(self):
        if self.is_win(self.state.bitboard1): # player 1 wins
            done = True
            reward = 1
            print('Player 1 wins!')
        elif self.is_win(self.state.bitboard2): # player 1 loses
            done = True
            reward = -1
            print('Player 2 wins!')
        elif not self.state.possible_moves(): # draw
            done = True
            reward = 0
            print('Draw!')
        else: # not ended
            done = False
            reward = 0
        return reward, done

    def ask_for_action(self) -> int:
        """Ask user for players next move"""
        while True:
            move_col = input(f'\nPlayer {self.player % 3} enter a move column between 1-{COLS}: ')
            if move_col.isnumeric():
                move_col = int(move_col)
                if move_col in range(1, COLS+1):
                    action = self.state.possible_moves() & self.col_bitmask(move_col)
                    if action == 0:
                        print('\nColumn is full!')
                    else:
                        return action
                else:
                    print('\nColumn out of bounds!')
            else:
                print('\nOnly use the number keys!')
                

    @staticmethod
    def make2D(binary: str) -> list[str]:
        '''Remove bitboard padding and split it by the number of rows'''
        return [binary[(COLS+1)*i:(COLS+1)*i+COLS] for i in range(ROWS)]
    
    @staticmethod
    def col_bitmask(move_col: int) -> int:
        '''Convert a move column integer into a bitmask to be used on a bitboard'''
        a = ['0']*(COLS+1)
        a[move_col-1]='1'
        b = ''.join(a) * ROWS
        return int(b, 2)
    
    @staticmethod
    def is_win(bitboard: int) -> bool:
        '''Check if a bitboard integer has connect four'''
        # Check -
        pairs = bitboard & (bitboard >> 1)
        if (pairs & (pairs >> 2 * 1)):
            return True
        # Check |
        pairs = bitboard & (bitboard >> (COLS+1))
        if (pairs & (pairs >> 2 * (COLS+1))):
            return True
        # Check /
        pairs = bitboard & (bitboard >> COLS)
        if (pairs & (pairs >> 2 * COLS)):
            return True
        # Check \
        pairs = bitboard & (bitboard >> (COLS+2))
        if (pairs & (pairs >> 2 * (COLS+2))):
            return True

def main():
    for episode in range(3):
        env = Connect4()
        old_s = env.state
        while True:
            a = env.ask_for_action()
            s, r, done = env.step(a)
            old_s = s
            if done:
                break

if __name__ == "__main__":
    ROWS, COLS = 6, 7
    BYTE_LENGTH = ROWS * (COLS+1)
    BOTTOM_BITMASK = ( 2**(COLS+1) - 2 ) 
    FULL_BITMASK = ( 2**(BYTE_LENGTH) - 1 ) # used to remove extra bits from a left bitshift
    main()