import random

def to_binary(bitboard):
    return bin(bitboard)[2:].zfill(ROWS*(COLS+1))

def make2D(string):
    output = []
    for i in range(ROWS):
        substring = string[(COLS+1)*i:(COLS+1)*i+COLS]
        output.append(substring)
    return output

def to_mailbox(bitboard1, bitboard2):
    board1, board2  = to_binary(bitboard1), to_binary(bitboard2)
    mailbox = ''
    for bit in range(ROWS*(COLS+1)):
        mailbox += str(int(board2[bit] + board1[bit], 2))
    return mailbox

def possible_moves(bitboard1, bitboard2):
    overlap = bitboard1 | bitboard2 # find all piece positions
    shifted_up = (overlap << (COLS+1)) | (2**(COLS+1)-2) # bitshift up a row and fill in empty bottom row
    possible_moves = shifted_up & ~overlap & (2**(ROWS*(COLS+1))-1) # the last AND operation removes extra bits
    return possible_moves

def get_action():
    while True:
        move_col = input('Enter a move column between 1-%d: ' %COLS)
        if move_col.isnumeric():
            move_col = int(move_col)
            if move_col in range(1, COLS+1):
                a = ['0']*(COLS+1)
                a[move_col-1]='1'
                b = ''.join(a)
                move_col_bitboard = int(b*ROWS, 2)
                available_moves = possible_moves(bitboard1, bitboard2)
                action = available_moves & move_col_bitboard
                if not action:
                    print('Column not available!')
                break
            else:
                print('Column out of bounds!')
        else:
            print('Use the number keys!')
    return action

class Connect4:
    def __init__(self):
        pass
    
    def step(self,action):
        #self.state[action] = self.player # applies action
        #reward, done = self.is_win(self.state)
        info = None
        env.player *= -1 # change player
        return #self.state, reward, done, info
    
    def reset(self):
        #self.state = np.zeros(9, dtype=int) # [0 0 0 0 0 0 0 0 0]
        self.player = 1 # player 1 = 1, player 2 = -1
        return #self.state
    
    def render(self):
        return
    
    def is_win(self, bitboard):
        result = ''
        # Check -
        pairs = bitboard & (bitboard >> 1)
        if (pairs & (pairs >> 2 * 1)):
            result += '-'
        # Check |
        pairs = bitboard & (bitboard >> (COLS+1))
        if (pairs & (pairs >> 2 * (COLS+1))):
            result+= '|'
        # Check /
        pairs = bitboard & (bitboard >> COLS)
        if (pairs & (pairs >> 2 * COLS)):
            result+= '/'
        # Check \
        pairs = bitboard & (bitboard >> (COLS+2))
        if (pairs & (pairs >> 2 * (COLS+2))):
            result+= '\\'
        return result #reward, done

if __name__ == "__main__":
    ROWS, COLS = 6, 7
    bitboard1, bitboard2  =  0, 0
    env = Connect4()
    for episode in range(3):
            old_state = env.reset()
            #while True:
            action = get_action()

                # state, reward, done, info = env.step(action)
                #old_state = state
                #if done:
                    #break