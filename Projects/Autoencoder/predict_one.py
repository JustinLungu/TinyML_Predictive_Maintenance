from save_model import Load_Model
import numpy as np
import sys
from os.path import join

WINDOW_SIZE = 24
DATA_SHAPE = 3 #x,y,z accelerometer data
MODELS_FOLDER_PATH = "Models/Autoencoder/"
PLOTS_FOLDER_PATH = "Projects/Autoencoder/Plots"

if __name__ == "__main__":

    #set the threshhold of prinitng data to console to maximum value
    #so avoid the loss of data on console while displaying
    np.set_printoptions(threshold=sys.maxsize)
    # setting up a random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)


    reading = [0.5075, 0.4825, 0.675 , 0.475 , 0.51 , 0.8 ,
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
    
    input_data = np.array(reading)
    input_data = np.expand_dims(input_data, axis = 0) # add a batch dimension


    #Predict on the reading
    loader = Load_Model()
    output = loader.predict_tflite(join(MODELS_FOLDER_PATH, "autoencoder_model.tflite"), input_data)

    print("Prediction of the one reading: ", output)

    #calculate mean absolute error
    mae = np.mean(np.abs(reading - output))
    print("MAE for the one reading: ", mae)
