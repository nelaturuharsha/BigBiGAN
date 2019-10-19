import io
import PIL.Image
from PIL import Image
from pprint import pformat
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from model_utils import *
from model import *
import runway
from runway.data_types import *


def image_to_uint8(x):
  """Converts [-1, 1] float array to [0, 255] uint8."""
  x = np.asarray(x)
  x = (256. / 2.) * (x + 1.)
  x = np.clip(x, 0, 255)
  x = x.astype(np.uint8)
  return x

@runway.setup
def setup():
	global bigbigan

	module_path = 'https://tfhub.dev/deepmind/bigbigan-revnet50x4/1'

	module = hub.Module(module_path)
	bigbigan = BigBiGAN(module)

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	return sess

generator_input = {"input_image" : runway.image}
generator_output = {"output_image" : runway.image}

@runway.command("generate_image", inputs=generator_input, outputs=generator_output, description="Generates Image from encoding")
def generate_image(sess, inputs):
	
	enc_ph = bigbigan.make_encoder_ph()
	recon_x = bigbigan.reconstruct_x(enc_ph, upsample=True)

	image = np.array(inputs["input_image"])	
	
	#img = tf.image.resize_bicubic([image], [256, 256])[0]
	img = cv2.resize(image, (256, 256), cv2.INTER_CUBIC)
	img = (tf.cast(img, tf.float32) / 127.5 - 1).eval(session=sess)
	
	img = np.expand_dims(img, axis=0)
	
	_out_recons = sess.run(recon_x, feed_dict={enc_ph: img})

	generated_image = imgrid(image_to_uint8(_out_recons), cols=1)

	return {"output_image" : generated_image}


if __name__ == "__main__":
	runway.run()
