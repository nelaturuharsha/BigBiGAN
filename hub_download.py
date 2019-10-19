import tensorflow_hub as hub
from pprint import pformat

module_path = 'https://tfhub.dev/deepmind/bigbigan-resnet50/1'

module = hub.Module(module_path)

for signature in module.get_signature_names():
  print('Signature:', signature)
  print('Inputs:', pformat(module.get_input_info_dict(signature)))
  print('Outputs:', pformat(module.get_output_info_dict(signature)))
  print()
