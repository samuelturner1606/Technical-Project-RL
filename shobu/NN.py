import numpy as np
import tensorflow as tf

def int_to_tensor(bits: int) -> tf.Tensor:
    'Convert a bitboard to a tensor.'
    bytes = bits.to_bytes(length=3, byteorder='little')
    byte_array = np.frombuffer(bytes, dtype=np.uint8)
    bit_array = np.unpackbits(byte_array, bitorder='little').reshape((4, 6))
    tensor = tf.constant(bit_array)
    return tensor

def tensor_to_int(tensor: tf.Tensor) -> int:
    'Convert a tensor to a bitboard.'
    flat = np.ravel(tensor)
    pack = np.packbits(flat, bitorder='little')
    num = int.from_bytes(pack, byteorder='little')   
    return num