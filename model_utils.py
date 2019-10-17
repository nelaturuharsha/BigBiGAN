import io
import IPython.display
import PIL.Image
from pprint import pformat

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def imgrid(imarray, cols=4, pad=1, padval=255, row_major=True):
  """Lays out a [N, H, W, C] image array as a single image grid."""
  pad = int(pad)
  if pad < 0:
    raise ValueError('pad must be non-negative')
  cols = int(cols)
  assert cols >= 1
  N, H, W, C = imarray.shape
  rows = N // cols + int(N % cols != 0)
  batch_pad = rows * cols - N
  assert batch_pad >= 0
  post_pad = [batch_pad, pad, pad, 0]
  pad_arg = [[0, p] for p in post_pad]
  imarray = np.pad(imarray, pad_arg, 'constant', constant_values=padval)
  H += pad
  W += pad
  grid = (imarray
          .reshape(rows, cols, H, W, C)
          .transpose(0, 2, 1, 3, 4)
          .reshape(rows*H, cols*W, C))
  if pad:
    grid = grid[:-pad, :-pad]
  return grid

def interleave(*args):
  """Interleaves input arrays of the same shape along the batch axis."""
  if not args:
    raise ValueError('At least one argument is required.')
  a0 = args[0]
  if any(a.shape != a0.shape for a in args):
    raise ValueError('All inputs must have the same shape.')
  if not a0.shape:
    raise ValueError('Inputs must have at least one axis.')
  out = np.transpose(args, [1, 0] + list(range(2, len(a0.shape) + 1)))
  out = out.reshape(-1, *a0.shape[1:])
  return out

def imshow(a, format='png', jpeg_fallback=True):
  """Displays an image in the given format."""
  a = a.astype(np.uint8)
  data = io.BytesIO()
  PIL.Image.fromarray(a).save(data, format)
  im_data = data.getvalue()
  try:
    disp = IPython.display.display(IPython.display.Image(im_data))
  except IOError:
    if jpeg_fallback and format != 'jpeg':
      print ('Warning: image was too large to display in format "{}"; '
             'trying jpeg instead.').format(format)
      return imshow(a, format='jpeg')
    else:
      raise
  return disp

def image_to_uint8(x):
  """Converts [-1, 1] float array to [0, 255] uint8."""
  x = np.asarray(x)
  x = (256. / 2.) * (x + 1.)
  x = np.clip(x, 0, 255)
  x = x.astype(np.uint8)
  return x


def get_flowers_data():
  """Returns a [32, 256, 256, 3] np.array of preprocessed TF-Flowers samples."""
  import tensorflow_datasets as tfds
  ds, info = tfds.load('tf_flowers', split='train', with_info=True)

  # Just get the images themselves as we don't need labels for this demo.
  ds = ds.map(lambda x: x['image'])

  # Filter out small images (with minor edge length <256).
  ds = ds.filter(lambda x: tf.reduce_min(tf.shape(x)[:2]) >= 256)

  # Take the center square crop of the image and resize to 256x256.
  def crop_and_resize(image):
    imsize = tf.shape(image)[:2]
    minor_edge = tf.reduce_min(imsize)
    start = (imsize - minor_edge) // 2
    stop = start + minor_edge
    cropped_image = image[start[0] : stop[0], start[1] : stop[1]]
    resized_image = tf.image.resize_bicubic([cropped_image], [256, 256])[0]
    return resized_image
  ds = ds.map(crop_and_resize)

  # Convert images from [0, 255] uint8 to [-1, 1] float32.
  ds = ds.map(lambda image: tf.cast(image, tf.float32) / (255. / 2.) - 1)

  # Take the first 32 samples.
  ds = ds.take(32)

  return np.array(list(tfds.as_numpy(ds)))