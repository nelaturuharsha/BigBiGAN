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

g = None

@runway.setup
def setup():
    global bigbigan
    global g

    module_path = 'https://tfhub.dev/deepmind/bigbigan-resnet50/1'

    module = hub.Module(module_path)
    bigbigan = BigBiGAN(module)

    g = tf.get_default_graph()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    enc_ph = bigbigan.make_encoder_ph()
    z = bigbigan.encode(enc_ph, return_all_features=True)['z_mean']
    recons = bigbigan.generate(z, upsample=True)

    return {
        'sess': sess,
        'enc_ph': enc_ph,
        'recons': recons
    }

generator_input = {"input_image" : runway.image}
generator_output = {"output_image" : runway.image}

@runway.command("generate_image", inputs=generator_input, outputs=generator_output, description="Generates Image from encoding")
def generate_image(model, inputs):
    image = np.array(inputs["input_image"].resize((256, 256)))
    image = image / 127.5 - 1
    image = np.expand_dims(image, axis=0)
    with g.as_default():
        _out_recons = model['sess'].run(model['recons'], feed_dict={model['enc_ph']: image})
    generated_image = imgrid(image_to_uint8(_out_recons), cols=1)
    return {"output_image" : generated_image}


if __name__ == "__main__":
    runway.run()
