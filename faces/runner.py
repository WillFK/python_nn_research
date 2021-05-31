import numpy as np
import tensorflow as tf
import cv2
import pathlib

IMG_SIZE = 64

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter("model.tflite")

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

# input details
print(input_details)
# output details
print(output_details)

for file in pathlib.Path("inputs").iterdir():
    
    # read and resize the image
    img = cv2.imread(r"{}".format(file.resolve()))
    new_img = tf.cast(img, tf.float32)
    new_img = tf.image.resize(new_img, [IMG_SIZE, IMG_SIZE])
    # new_img = cv2.resize(new_img, (256, 256))
    new_img /= 255

    
    # input_details[0]['index'] = the index which accepts the input
    interpreter.set_tensor(input_details[0]['index'], [new_img])
    
    # run the inference
    interpreter.invoke()
    
    # output_details[0]['index'] = the index which provides the input
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print("For file {}, the output is {}".format(file.stem, output_data))