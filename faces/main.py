import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import tempfile

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


import tensorflow_datasets as tfds

# get dataset

(train_data, test_data), info = tfds.load(name = 'celeb_a', split = ['train', 'test'], with_info = True, batch_size = 32)

train_count = len(list(train_data))

print(train_count)

# attributes
ATTR_KEY = "attributes"
IMAGE_KEY = "image"
LABEL_KEY = "Smiling"
GROUP_KEY = "Young"
IMAGE_SIZE = 28

def preprocess_input_dict(feat_dict):
  # Separate out the image and target variable from the feature dictionary.
  image = feat_dict[IMAGE_KEY]
  label = feat_dict[ATTR_KEY][LABEL_KEY]
  group = feat_dict[ATTR_KEY][GROUP_KEY]

  # Resize and normalize image.
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
  image /= 255.0

  # Cast label and group to float32.
  label = tf.cast(label, tf.float32)
  group = tf.cast(group, tf.float32)

  feat_dict[IMAGE_KEY] = image
  feat_dict[ATTR_KEY][LABEL_KEY] = label
  feat_dict[ATTR_KEY][GROUP_KEY] = group

  return feat_dict

get_image_and_label = lambda feat_dict: (feat_dict[IMAGE_KEY], feat_dict[ATTR_KEY][LABEL_KEY])
get_image_label_and_group = lambda feat_dict: (feat_dict[IMAGE_KEY], feat_dict[ATTR_KEY][LABEL_KEY], feat_dict[ATTR_KEY][GROUP_KEY])
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
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dense(1, activation=None)
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

class PreprocessCelebA(object):
  """Class that deserializes, decodes and applies additional preprocessing for CelebA input."""
  def __init__(self, dataset_name):
    builder = tfds.builder(dataset_name)
    self.features = builder.info.features
    example_specs = self.features.get_serialized_info()
    self.parser = tfds.core.example_parser.ExampleParser(example_specs)

  def __call__(self, serialized_example):
    # Deserialize
    deserialized_example = self.parser.parse_example(serialized_example)
    # Decode
    decoded_example = self.features.decode_example(deserialized_example)
    # Additional preprocessing
    image = decoded_example[IMAGE_KEY]
    label = decoded_example[ATTR_KEY][LABEL_KEY]
    # Resize and scale image.
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0
    image = tf.reshape(image, [-1])
    # Cast label and group to float32.
    label = tf.cast(label, tf.float32)

    group = decoded_example[ATTR_KEY][GROUP_KEY]

    output = tf.train.Example()
    output.features.feature[IMAGE_KEY].float_list.value.extend(image.numpy().tolist())
    output.features.feature[LABEL_KEY].float_list.value.append(label.numpy())
    output.features.feature[GROUP_KEY].bytes_list.value.append(b"Young" if group.numpy() else b'Not Young')
    return output.SerializeToString()

def tfds_as_pcollection(beam_pipeline, dataset_name, split):
  return (
      beam_pipeline
   | 'Read records' >> beam.io.ReadFromTFRecord(tfds_filepattern_for_split(dataset_name, split))
   | 'Preprocess' >> beam.Map(PreprocessCelebA(dataset_name))
  )

  def get_eval_results(model_location, eval_subdir):
  base_dir = tempfile.mkdtemp(prefix='saved_eval_results')
  tfma_eval_result_path = os.path.join(base_dir, eval_subdir)

  eval_config_pbtxt = """
        model_specs {
          label_key: "%s"
        }
        metrics_specs {
          metrics {
            class_name: "FairnessIndicators"
            config: '{ "thresholds": [0.22, 0.5, 0.75] }'
          }
          metrics {
            class_name: "ExampleCount"
          }
        }
        slicing_specs {}
        slicing_specs { feature_keys: "%s" }
        options {
          compute_confidence_intervals { value: False }
          disabled_outputs{values: "analysis"}
        }
      """ % (LABEL_KEY, GROUP_KEY)

  eval_config = text_format.Parse(eval_config_pbtxt, tfma.EvalConfig())

  eval_shared_model = tfma.default_eval_shared_model(
        eval_saved_model_path=model_location, tags=[tf.saved_model.SERVING])

  schema_pbtxt = """
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "%s"
              value {
                dense_tensor {
                  column_name: "%s"
                  shape {
                    dim { size: 28 }
                    dim { size: 28 }
                    dim { size: 3 }
                  }
                }
              }
            }
          }
        }
        feature {
          name: "%s"
          type: FLOAT
        }
        feature {
          name: "%s"
          type: FLOAT
        }
        feature {
          name: "%s"
          type: BYTES
        }
        """ % (IMAGE_KEY, IMAGE_KEY, IMAGE_KEY, LABEL_KEY, GROUP_KEY)
  schema = text_format.Parse(schema_pbtxt, schema_pb2.Schema())
  coder = tf_example_record.TFExampleBeamRecord(
      physical_format='inmem', schema=schema,
      raw_record_column_name=tfma.ARROW_INPUT_COLUMN)
  tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
    arrow_schema=coder.ArrowSchema(),
    tensor_representations=coder.TensorRepresentations())
  # Run the fairness evaluation.
  with beam.Pipeline() as pipeline:
    _ = (
          tfds_as_pcollection(pipeline, 'celeb_a', 'test')
          | 'ExamplesToRecordBatch' >> coder.BeamSource()
          | 'ExtractEvaluateAndWriteResults' >>
          tfma.ExtractEvaluateAndWriteResults(
              eval_config=eval_config,
              eval_shared_model=eval_shared_model,
              output_path=tfma_eval_result_path,
              tensor_adapter_config=tensor_adapter_config)
    )
  return tfma.load_eval_result(output_path=tfma_eval_result_path)


BATCH_SIZE = 32

# Set seeds to get reproducible results
set_seeds()

model_unconstrained = create_model()
model_unconstrained.fit(celeb_a_train_data_wo_group(BATCH_SIZE), epochs=5, steps_per_epoch=1000)

print('Overall Results, Unconstrained')
celeb_a_test_data = celeb_a_builder.as_dataset(split='test').batch(1).map(preprocess_input_dict).map(get_image_label_and_group)
results = model_unconstrained.evaluate(celeb_a_test_data)