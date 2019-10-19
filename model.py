import io
import PIL.Image
from pprint import pformat

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub



class BigBiGAN(object):

  def __init__(self, module):
    """Initialize a BigBiGAN from the given TF Hub module."""
    self._module = module

  def generate(self, z, upsample=False):
    """Run a batch of latents z through the generator to generate images.

    Args:
      z: A batch of 120D Gaussian latents, shape [N, 120].

    Returns: a batch of generated RGB images, shape [N, 128, 128, 3], range
      [-1, 1].
    """
    outputs = self._module(z, signature='generate', as_dict=True)
    return outputs['upsampled' if upsample else 'default']

  def make_generator_ph(self):
    """Creates a tf.placeholder with the dtype & shape of generator inputs."""
    info = self._module.get_input_info_dict('generate')['z']
    return tf.placeholder(dtype=info.dtype, shape=info.get_shape())

  def gen_pairs_for_disc(self, z):
    """Compute generator input pairs (G(z), z) for discriminator, given z.

    Args:
      z: A batch of latents (120D standard Gaussians), shape [N, 120].

    Returns: a tuple (G(z), z) of discriminator inputs.
    """
    # Downsample 256x256 image x for 128x128 discriminator input.
    x = self.generate(z)
    return x, z

  def encode(self, x, return_all_features=False):
    """Run a batch of images x through the encoder.

    Args:
      x: A batch of data (256x256 RGB images), shape [N, 256, 256, 3], range
        [-1, 1].
      return_all_features: If True, return all features computed by the encoder.
        Otherwise (default) just return a sample z_hat.

    Returns: the sample z_hat of shape [N, 120] (or a dict of all features if
      return_all_features).
    """
    outputs = self._module(x, signature='encode', as_dict=True)
    return outputs if return_all_features else outputs['z_sample']

  def make_encoder_ph(self):
    """Creates a tf.placeholder with the dtype & shape of encoder inputs."""
    info = self._module.get_input_info_dict('encode')['x']
    return tf.placeholder(dtype=info.dtype, shape=info.get_shape())

  def enc_pairs_for_disc(self, x):
    """Compute encoder input pairs (x, E(x)) for discriminator, given x.

    Args:
      x: A batch of data (256x256 RGB images), shape [N, 256, 256, 3], range
        [-1, 1].

    Returns: a tuple (downsample(x), E(x)) of discriminator inputs.
    """
    # Downsample 256x256 image x for 128x128 discriminator input.
    x_down = tf.nn.avg_pool(x, ksize=2, strides=2, padding='SAME')
    z = self.encode(x)
    return x_down, z

  def discriminate(self, x, z):
    """Compute the discriminator scores for pairs of data (x, z).

    (x, z) must be batches with the same leading batch dimension, and joint
      scores are computed on corresponding pairs x[i] and z[i].

    Args:
      x: A batch of data (128x128 RGB images), shape [N, 128, 128, 3], range
        [-1, 1].
      z: A batch of latents (120D standard Gaussians), shape [N, 120].

    Returns:
      A dict of scores:
        score_xz: the joint scores for the (x, z) pairs.
        score_x: the unary scores for x only.
        score_z: the unary scores for z only.
    """
    inputs = dict(x=x, z=z)
    return self._module(inputs, signature='discriminate', as_dict=True)

  def reconstruct_x(self, x, use_sample=True, upsample=False):
    """Compute BigBiGAN reconstructions of images x via G(E(x)).

    Args:
      x: A batch of data (256x256 RGB images), shape [N, 256, 256, 3], range
        [-1, 1].
      use_sample: takes a sample z_hat ~ E(x). Otherwise, deterministically
        use the mean. (Though a sample z_hat may be far from the mean z,
        typically the resulting recons G(z_hat) and G(z) are very
        similar.
      upsample: if set, upsample the reconstruction to the input resolution
        (256x256). Otherwise return the raw lower resolution generator output
        (128x128).

    Returns: a batch of recons G(E(x)), shape [N, 256, 256, 3] if
      `upsample`, otherwise [N, 128, 128, 3].
    """
    if use_sample:
      z = self.encode(x)
    else:
      z = self.encode(x, return_all_features=True)['z_mean']
    recons = self.generate(z, upsample=upsample)
    return recons

  def losses(self, x, z):
    """Compute per-module BigBiGAN losses given data & latent sample batches.

    Args:
      x: A batch of data (256x256 RGB images), shape [N, 256, 256, 3], range
        [-1, 1].
      z: A batch of latents (120D standard Gaussians), shape [M, 120].

    For the original BigBiGAN losses, pass batches of size N=M=2048, with z's
    sampled from a 120D standard Gaussian (e.g., np.random.randn(2048, 120)),
    and x's sampled from the ImageNet (ILSVRC2012) training set with the
    "ResNet-style" preprocessing from:

        https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_preprocessing.py

    Returns:
      A dict of per-module losses:
        disc: loss for the discriminator.
        enc: loss for the encoder.
        gen: loss for the generator.
    """
    # Compute discriminator scores on (x, E(x)) pairs.
    # Downsample 256x256 image x for 128x128 discriminator input.
    scores_enc_x_dict = self.discriminate(*self.enc_pairs_for_disc(x))
    scores_enc_x = tf.concat([scores_enc_x_dict['score_xz'],
                              scores_enc_x_dict['score_x'],
                              scores_enc_x_dict['score_z']], axis=0)

    # Compute discriminator scores on (G(z), z) pairs.
    scores_gen_z_dict = self.discriminate(*self.gen_pairs_for_disc(z))
    scores_gen_z = tf.concat([scores_gen_z_dict['score_xz'],
                              scores_gen_z_dict['score_x'],
                              scores_gen_z_dict['score_z']], axis=0)

    disc_loss_enc_x = tf.reduce_mean(tf.nn.relu(1. - scores_enc_x))
    disc_loss_gen_z = tf.reduce_mean(tf.nn.relu(1. + scores_gen_z))
    disc_loss = disc_loss_enc_x + disc_loss_gen_z

    enc_loss = tf.reduce_mean(scores_enc_x)
    gen_loss = tf.reduce_mean(-scores_gen_z)

    return dict(disc=disc_loss, enc=enc_loss, gen=gen_loss)
