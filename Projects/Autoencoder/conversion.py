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
H5_PATH = "Models/Autoencoder/autoencoder_model.h5"
MODELS_FOLDER_PATH = "Models/Autoencoder/H5_Conversion"
MODEL_NAME = "autoencoder_model"


###################################### LOADING THE H5 WEIGHTS #############################
# Recreate the model
model = Autoencoder(WINDOW_SIZE)

# Call the model to create its variables
# You can pass dummy input data with the expected shape
dummy_input = tf.zeros((1, WINDOW_SIZE * 3))
_ = model(dummy_input)

# Load the weights of h5 model (keras model)
model.load_weights(H5_PATH)
############################################################################################

##################################### CONVERTING FROM H5 MODEL #############################

#convert keras model to a tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(join(MODELS_FOLDER_PATH, MODEL_NAME) + '.tflite', 'wb').write(tflite_model)

#construct header file
hex_array = [format(val, '#04x') for val in tflite_model]
c_model = c_writer.create_array(np.array(hex_array), 'unsigned char', MODEL_NAME)
header_str = c_writer.create_header(c_model, MODEL_NAME)

# save c header file
with open(join(MODELS_FOLDER_PATH, MODEL_NAME) + '.h', 'w') as file:
    file.write(header_str)
#############################################################################################


###################################### TEST INFERENCE #######################################

'''
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
'''
data_60hz30vol_normalized = [0.5075, 0.4825, 0.675]
input_data_test = np.array(data_60hz30vol_normalized)

print(input_data_test.shape)

##### I will copy the code from google colab used to run inference as before #####

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path= MODELS_FOLDER_PATH + "/autoencoder_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
test_dataa = tf.expand_dims(tf.convert_to_tensor(input_data_test, dtype=tf.float32), 0)

input_data = test_dataa

interpreter.set_tensor(input_details[0]['index'], input_data)


interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print("MODEL PREDICTIONS: ", output_data)

#find MAE
mae = np.mean(np.abs(input_data_test - output_data))
print("MEAN ABSOLUTE ERROR: ", mae)