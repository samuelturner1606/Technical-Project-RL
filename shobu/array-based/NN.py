import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import sonnet as snt
from blocks import make_network

def int_to_array(bits: int) -> np.ndarray:
    'Convert a bitboard to a ndarray.'
    bytes = bits.to_bytes(length=3, byteorder='little')
    byte_array = np.frombuffer(bytes, dtype=np.uint8)
    bit_array = np.unpackbits(byte_array, bitorder='little').reshape((4, 6))[:, :4]
    return bit_array

def array_to_int(bit_array: np.ndarray) -> int:
    'Convert a array to a bitboard, used for hashing and storage.'
    pad = np.zeros((4, 6), dtype=np.uint8)
    pad[:, :4] = bit_array
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

class Node:
  def __init__(self, prior: float):
    self.visit_count = 0
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.reward = 0

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

a = np.random.randint(2, size=(8, 2, 8, 8, 8, 8), dtype=np.uint8)
c = tf.convert_to_tensor(a, dtype=tf.float32)
pass