from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
import c_writer
from model import Autoencoder
import numpy as np

WINDOW_SIZE = 1
LEARNING_RATE = 0.001  # Change this to your desired learning rate
OPTIMIZER = "adam"
LOSS = "mae"
MODELS_FOLDER_PATH = "Models/Autoencoder/autoencoder_model.h5"

# Recreate the model
model = Autoencoder(WINDOW_SIZE)

# Call the model to create its variables
# You can pass dummy input data with the expected shape
dummy_input = tf.zeros((1, WINDOW_SIZE * 3))
_ = model(dummy_input)

# Load the weights
model.load_weights(MODELS_FOLDER_PATH)

data_60hz30vol_normalized = [0.5075, 0.4825, 0.675 , 0.475 , 0.51 , 0.8 , 
                             0.5075, 0.4825, 0.6775, 0.475 , 0.51 , 0.8 , 
                             0.5075, 0.4825, 0.6775, 0.4775, 0.51 , 0.8 , 
                             0.5075, 0.4825, 0.68 , 0.4775, 0.51 , 0.7975, 
                             0.505 , 0.4825, 0.6825, 0.4775, 0.51 , 0.7975, 
                             0.505 , 0.4825, 0.6825, 0.4775, 0.51 , 0.7975, 
                             0.505 , 0.4825, 0.685 , 0.4775, 0.51 , 0.7975, 
                             0.505 , 0.4825, 0.685 , 0.48 , 0.51 , 0.7975, 
                             0.505 , 0.4825, 0.6875, 0.48 , 0.5075, 0.795 , 
                             0.505 , 0.4825, 0.6875, 0.48 , 0.5075, 0.795 , 
                             0.505 , 0.4825, 0.69 , 0.48 , 0.5075, 0.795 , 
                             0.505 , 0.485 , 0.6925, 0.48 , 0.5075, 0.795]
input_data_test = np.array(data_60hz30vol_normalized)





'''
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
'''
