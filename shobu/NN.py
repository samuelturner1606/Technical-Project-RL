import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import sonnet as snt
from blocks import make_network

def int2tensor(bits: int) -> tf.Tensor:
    'Convert a bitboard to a tensor.'
    bytes = bits.to_bytes(length=3, byteorder='little')
    byte_array = np.frombuffer(bytes, dtype=np.uint8)
    bit_array = np.unpackbits(byte_array, bitorder='little').reshape((4, 6))
    tensor = tf.constant(bit_array[:, :4])
    return tensor

def tensor2int(tensor: tf.Tensor) -> int:
    'Convert a tensor to a bitboard.'
    pad = np.zeros((4, 6), dtype=np.uint8)
    pad[:, :4] = tensor
    flat = np.ravel(pad)
    pack = np.packbits(flat, bitorder='little')
    num = int.from_bytes(pack, byteorder='little')
    return num

def l(colour: bool):
    'Additional L constant-valued input planes.'
    if colour: 
        l = tf.ones((4,4), dtype=tf.uint8)
    else:
        l = tf.zeros((4,4), dtype=tf.uint8)
    print(l)
    return

# final convolution of 16*c (c = board combos) filters. 

class NN():
    'Gumbel AlphaZero model'
    def __init__(self) -> None:
        self.nn = make_network(6, 256, 128, 8)
    