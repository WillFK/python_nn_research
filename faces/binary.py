import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt 
import tempfile

ATTR_KEY = "attributes"
IMAGE_KEY = "image"
LABEL_KEY = "Eyeglasses"
IMG_SIZE = 64#256
BATCH_SIZE = 50
EPOCHS = 100

# get data set
print('>>> Get Data Set')
gcs_base_dir = 'gs://celeb_a_dataset/'
celeb_a_builder = tfds.builder('celeb_a', data_dir=gcs_base_dir, version='2.0.0')
celeb_a_builder.download_and_prepare()

def preprocess_input_dict(feat_dict):
  # Separate out the image and target variable from the feature dictionary.
  image = feat_dict[IMAGE_KEY]
  label = feat_dict[ATTR_KEY][LABEL_KEY]

  # Resize and normalize image.
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
  image /= 255.0

  label = tf.cast(label, tf.float32)
  
  feat_dict[IMAGE_KEY] = image
  feat_dict[ATTR_KEY][LABEL_KEY] = label
  
  return feat_dict

get_image_and_label = lambda feat_dict: (feat_dict[IMAGE_KEY], feat_dict[ATTR_KEY][LABEL_KEY])

def create_model():
    print('>>> Creating model')
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE, 3))),
    model.add(keras.layers.Dense(60, input_dim=60, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy'])
    return model

train_ds = celeb_a_builder \
  .as_dataset(split='train') \
  .batch(BATCH_SIZE) \
  .map(preprocess_input_dict) \
  .map(get_image_and_label)

test_ds = celeb_a_builder \
  .as_dataset(split='test') \
  .batch(1) \
  .map(preprocess_input_dict) \
  .map(get_image_and_label)

model = create_model()

model.summary()

history = model.fit(train_ds, epochs=3, steps_per_epoch=EPOCHS)

model.evaluate(test_ds)

def save_model(model, subdir):
  base_dir = tempfile.mkdtemp(prefix='saved_models')
  model_location = os.path.join(base_dir, subdir)
  model.save(model_location, save_format='tf')
  return model_location

model_location = save_model(model, 'model_export_unconstrained')

converter = tf.lite.TFLiteConverter.from_saved_model(model_location) # path to the SavedModel directory
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
  f.write(tflite_model)