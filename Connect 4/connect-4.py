
def to_binary(bitboard):
    return bin(bitboard)[2:].zfill(rows*(cols+1))

def make2D(string):
    output = []
    for i in range(rows):
        substring = string[(cols+1)*i:(cols+1)*i+cols]
        output.append(substring)
    return output

def to_mailbox(bitboard1, bitboard2):
    board1, board2  = to_binary(bitboard1), to_binary(bitboard2)
    mailbox = ''
    for bit in range(rows*(cols+1)):
        mailbox += str(int(board2[bit] + board1[bit], 2))
    return mailbox

def next_moves(bitboard1, bitboard2):
    overlap = bitboard1 | bitboard2 # find all piece positions
    shifted_up = (overlap << (cols+1)) | (2**(cols+1)-2) # bitshift up a row and fill in empty bottom row
    next_moves = shifted_up & ~overlap & (2**(rows*(cols+1))-1) # the last AND operation removes extra bits
    return next_moves

rows, cols = 6, 7
bitboard1, bitboard2  =  224482970800, 53052245848654 # example board state

m = to_mailbox(bitboard1, bitboard2)
d = make2D(m)
for i in d:
    print(i)

n = next_moves(bitboard1, bitboard2)
b = to_binary(n)
d2 = make2D(b)
for i in d2:
    print(i)