from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import models
from scipy import stats
import c_writer

# Settings
models_path = 'Models/Autoencoder/Diff_Conv'  # Where we can find the model files (relative path location)
keras_model_name = 'autoencoder_model'           # Will be given .h5 suffix
tflite_model_name = 'autoencoder_model'          # Will be given .tflite suffix
c_model_name = 'autoencoder_model'               # Will be given .h suffix1

# Load model
model = models.load_model(join(models_path, keras_model_name) + '.h5')


# Convert Keras model to a tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(join(models_path, tflite_model_name) + '.tflite', 'wb').write(tflite_model)

# Construct header file
hex_array = [format(val, '#04x') for val in tflite_model]
c_model = c_writer.create_array(np.array(hex_array), 'unsigned char', c_model_name)
header_str = c_writer.create_header(c_model, c_model_name)

# Save C header file
with open(join(models_path, c_model_name) + '.h', 'w') as file:
    file.write(header_str)

