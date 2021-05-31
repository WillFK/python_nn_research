import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tempfile

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import tensorflow_model_analysis as tfma

from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tf_example_record

from tensorflow_metadata.proto.v0 import schema_pb2

import tensorflow_datasets as tfds

from google.protobuf import text_format


gcs_base_dir = "gs://celeb_a_dataset/"
celeb_a_builder = tfds.builder("celeb_a", data_dir=gcs_base_dir, version='2.0.0')

celeb_a_builder.download_and_prepare()

num_test_shards_dict = {'0.3.0': 4, '2.0.0': 2} # Used because we download the test dataset separately
version = str(celeb_a_builder.info.version)
print('Celeb_A dataset version: %s' % version)

# attributes
ATTR_KEY = "attributes"
IMAGE_KEY = "image"
#LABEL_KEY = "Smiling"
LABEL_KEY = "Eyeglasses"
#GROUP_KEY = "Young"
IMAGE_SIZE = 28

def preprocess_input_dict(feat_dict):
  # Separate out the image and target variable from the feature dictionary.
  image = feat_dict[IMAGE_KEY]
  label = feat_dict[ATTR_KEY][LABEL_KEY]
  #group = feat_dict[ATTR_KEY][GROUP_KEY]

  # Resize and normalize image.
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
  image /= 255.0

  # Cast label and group to float32.
  label = tf.cast(label, tf.float32)
  #group = tf.cast(group, tf.float32)

  feat_dict[IMAGE_KEY] = image
  feat_dict[ATTR_KEY][LABEL_KEY] = label
  #feat_dict[ATTR_KEY][GROUP_KEY] = group

  return feat_dict

def fk_test(feat_dic):
  print("yo")
  #print("label {}".format(feat_dic[LABEL_KEY]))
  return feat_dic

get_image_and_label = lambda feat_dict: (feat_dict[IMAGE_KEY], feat_dict[ATTR_KEY][LABEL_KEY])
#get_image_label_and_group = lambda feat_dict: (feat_dict[IMAGE_KEY], feat_dict[ATTR_KEY][LABEL_KEY], feat_dict[ATTR_KEY][GROUP_KEY])

def create_model():
  # For this notebook, accuracy will be used to evaluate performance.
  METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy')
  ]

  # The model consists of:
  # 1. An input layer that represents the 28x28x3 image flatten.
  # 2. A fully connected layer with 64 units activated by a ReLU function.
  # 3. A single-unit readout layer to output real-scores instead of probabilities.
  model = keras.Sequential([
      keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='image'),
      keras.layers.Dense(64, activation='relu')
      ,
      #keras.layers.Dense(1, activation=None)
  ])

  # TFCO by default uses hinge loss — and that will also be used in the model.
  model.compile(
      optimizer=tf.keras.optimizers.Adam(0.001),
      loss='hinge',
      metrics=METRICS)
  return model

def set_seeds():
  np.random.seed(121212)
  tf.compat.v1.set_random_seed(212121)

def save_model(model, subdir):
  base_dir = tempfile.mkdtemp(prefix='saved_models')
  model_location = os.path.join(base_dir, subdir)
  model.save(model_location, save_format='tf')
  return model_location

def tfds_filepattern_for_split(dataset_name, split):
  return f"{local_test_file_full_prefix()}*"


# Train data returning either 2 or 3 elements (the third element being the group)
def celeb_a_train_data_wo_group(batch_size):
  celeb_a_train_data = celeb_a_builder.as_dataset(split='train').shuffle(1024).repeat().batch(batch_size).map(preprocess_input_dict)
  return celeb_a_train_data.map(get_image_and_label)

# Test data for the overall evaluation
celeb_a_test_data = celeb_a_builder.as_dataset(split='test').batch(1).map(preprocess_input_dict).map(get_image_and_label)
# Copy test data locally to be able to read it into tfma
#copy_test_files_to_local()

BATCH_SIZE = 32

# Set seeds to get reproducible results
set_seeds()

model_unconstrained = create_model()
model_unconstrained.fit(celeb_a_train_data_wo_group(BATCH_SIZE), epochs=2, steps_per_epoch=100)

model_unconstrained.summary()

print('Overall Results, Unconstrained')
celeb_a_test_data = celeb_a_builder \
  .as_dataset(split='test') \
  .batch(1) \
  .map(preprocess_input_dict) \
  .map(fk_test) \
  .map(get_image_and_label)

results = model_unconstrained.evaluate(celeb_a_test_data)

model_location = save_model(model_unconstrained, 'model_export_unconstrained')
#eval_results_unconstrained = get_eval_results(model_location, 'eval_results_unconstrained')

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(model_location) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

print("this is weird")
#tfma.addons.fairness.view.widget_view.render_fairness_indicator(eval_results_unconstrained)