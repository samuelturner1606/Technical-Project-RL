import numpy as np
import tensorflow as tf

def int_to_tensor(bits: int) -> tf.Tensor:
    'Convert a bitboard to a tensor.'
    bytes = bits.to_bytes(length=3, byteorder='little')
    byte_array = np.frombuffer(bytes, dtype=np.uint8)
    bit_array = np.unpackbits(byte_array, bitorder='little').reshape((4, 6))
    tensor = tf.constant(bit_array[:, :4])
    return tensor

def tensor_to_int(tensor: tf.Tensor) -> int:
    'Convert a tensor to a bitboard.'
    pad = np.zeros((4, 6), dtype=np.uint8)
    pad[:, :4] = tensor
    flat = np.ravel(pad)
    pack = np.packbits(flat, bitorder='little')
    num = int.from_bytes(pack, byteorder='little')
    return num