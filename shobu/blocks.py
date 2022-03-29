'''
Neural network blocks adapted from the code provided in the following paper:
"Danihelka, I., Guez, A., Schrittwieser, J. and Silver, D., 2021, September. Policy improvement by planning with Gumbel. In International Conference on Learning Representations."
'''
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import sonnet as snt
import functools


class BasicBlock(snt.Module):
  """Basic block composed of an inner op, a norm op and a non linearity."""
  def __init__(self, make_inner_op, non_linearity=tf.nn.relu, name='basic'):
    super().__init__(name=name)
    self._op = make_inner_op()
    self._norm = snt.BatchNorm(create_scale=False, create_offset=True, decay_rate=0.999, eps=1e-3)
    self._non_linearity = non_linearity
  def __call__(self, x: tf.Tensor, is_training: bool, test_local_stats: bool):
    x = self._op(x)
    x = self._norm(x, is_training=is_training, test_local_stats=test_local_stats)
    return  self._non_linearity(x)

class ResBlock(snt.Module):
  """Creates a residual block with an optional bottleneck."""
  def __init__(self, stack_size: int, make_first_op, make_inner_op, make_last_op, name):
    super().__init__(name=name)
    assert stack_size >= 2
    self._blocks = []
    for i, make_op in enumerate([make_first_op] + [make_inner_op] * (stack_size - 2) + [make_last_op]):
      self._blocks.append(
        BasicBlock(make_inner_op=make_op, non_linearity=lambda x: x if i == stack_size - 1 else tf.nn.relu, name=f'basic_{i}'))
  def __call__(self, x: tf.Tensor, is_training: bool, test_local_stats: bool):
    res = x
    for b in self._blocks:
      res = b(res, is_training, test_local_stats)
    return tf.nn.relu(x + res)
  
class BroadcastResBlock(ResBlock):
  """A residual block that broadcasts information across spatial dimensions.
  The block consists of a sequence of three layers:
   - a layer that mixes information across channels, e.g. a 1x1 convolution.
   - a layer that mixes information within each channel, a dense layer.
   - another layer to mix across channels.
  The same set of weights is used for mixing information within each channel.
  """
  def __init__(self, make_mix_channel_op, name):
    def broadcast(x: tnp.ndarray):
      n, h, w, c = x.shape
      # Process all planes at once, applynig the same linear layer to each.
      x = x.transpose((0, 3, 1, 2))  # NHWC -> NCHW
      x = x.reshape((n, c, h * w))
      x = snt.Linear(h * w, name='broadcast')(x)
      x = tf.nn.relu(x)
      x = x.reshape((n, c, h, w))
      x = x.transpose((0, 2, 3, 1))  # NCHW -> NHWC
      return x
    super().__init__( 
      stack_size=3,
      make_first_op=make_mix_channel_op,
      make_inner_op=lambda: broadcast,
      make_last_op=make_mix_channel_op,
      name=name)
  
def make_conv(output_channels: int, kernel_shape: int):
  return functools.partial(snt.Conv2D, output_channels, kernel_shape, with_bias=False, w_init=snt.initializers.TruncatedNormal(0.01))

def make_network(num_layers: int, output_channels: int, bottleneck_channels: int, broadcast_every_n: int):
  blocks = [BasicBlock(make_inner_op=make_conv(output_channels), non_linearity=tf.nn.relu, name='init_conv')]
  for i in range(num_layers):
    if broadcast_every_n > 0 and i % broadcast_every_n == broadcast_every_n - 1:
      blocks.append(BroadcastResBlock(make_mix_channel_op=make_conv(output_channels, kernel_shape=1), name=f'broadcast_{i}'))
    elif bottleneck_channels > 0:
      blocks.append(ResBlock(
        stack_size=4,
        make_first_op=make_conv(bottleneck_channels, kernel_shape=1),
        make_inner_op=make_conv(bottleneck_channels, kernel_shape=3),
        make_last_op=make_conv(output_channels, kernel_shape=1),
        name=f'bottleneck_res_{i}'))
    else:
      blocks.append(ResBlock(
        stack_size=2,
        make_first_op=make_conv(output_channels, kernel_shape=3),
        make_inner_op=make_conv(output_channels, kernel_shape=3),
        make_last_op=make_conv(output_channels, kernel_shape=3),
        name=f'res_{i}'))
    return blocks