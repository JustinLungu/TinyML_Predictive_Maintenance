import numpy as np
import sys
from data_preprocessing import Data, Preprocessing
from save_load_data import Saving_Loading
from model import AnomalyDetector


import os
import tensorflow as tf
import joblib



import subprocess



DATAPOINTS_PLOTTING = 2000
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
WINDOW_SIZE = 24
DATA_SHAPE = 3 #x,y,z accelerometer data
DATA_FOLDER_PATH = "Projects/Autoencoder/Preprocessed Data"
PLOTS_FOLDER_PATH = "Projects/Autoencoder/Plots"
MODELS_FOLDER_PATH = "Models/Autoencoder"
DO_PREPROCESSING = False
OPTIMIZER = "adam"
LOSS = "mse"
EPOCHS = 20
BATCH_SIZE = 512

def normalization(normal_data: Data, abnormal_data: Data):
    preprocess = Preprocessing()

    #normal data normalization (train,val,test)
    train = normal_data.train_data
    val = normal_data.val_data
    test = normal_data.test_data

    train = train.reshape(-1, WINDOW_SIZE * DATA_SHAPE)
    val = val.reshape(-1, WINDOW_SIZE * DATA_SHAPE)
    test = test.reshape(-1, WINDOW_SIZE * DATA_SHAPE)

    train = preprocess.min_max_scale_fit(train)
    val = preprocess.min_max_transform(val)
    test = preprocess.min_max_transform(test)

    normal_data.train_data = train.reshape(-1, WINDOW_SIZE, DATA_SHAPE)
    normal_data.val_data = val.reshape(-1, WINDOW_SIZE, DATA_SHAPE)
    normal_data.test_data = val.reshape(-1, WINDOW_SIZE, DATA_SHAPE)

    #abnormal data (the entire dataset)
    abnormal = abnormal_data.dataset
    abnormal = abnormal.reshape(-1, WINDOW_SIZE * DATA_SHAPE)
    abnormal = preprocess.min_max_scale_fit(abnormal)
    abnormal_data.dataset = abnormal.reshape(-1, WINDOW_SIZE, DATA_SHAPE)

def manipulate_data(normal_data, abnormal_data, save_load):
    # plot datasets
    normal_data.plotVibPattern(datapoints = DATAPOINTS_PLOTTING)
    abnormal_data.plotVibPattern(datapoints = DATAPOINTS_PLOTTING)

    normal_data.data_split_window(TRAIN_RATIO, VAL_RATIO, TEST_RATIO, WINDOW_SIZE)
    abnormal_data.dataset = abnormal_data.make_windows(abnormal_data.dataset, WINDOW_SIZE)
    
    normalization(normal_data, abnormal_data)

    # Save the preprocessed data in the specified folder
    save_load.save_data_json(normal_data, abnormal_data)


# Function to save model as .tflite
def save_as_tflite(model, filepath):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(filepath, 'wb') as f:
        f.write(tflite_model)

# Function to save model as .pkl
def save_as_pkl(model, filepath):
    joblib.dump(model, filepath)

# Function to create folder if it doesn't exist
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

if __name__ == "__main__":
    #set the threshhold of prinitng data to console to maximum value
    #so avoid the loss of data on console while displaying
    np.set_printoptions(threshold=sys.maxsize)
    # setting up a random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)

    normal_data = Data(capture = "1", hertz = "60", volume = "30")
    abnormal_data = Data(capture = "2", hertz = "40", volume = "60")
    save_load = Saving_Loading(DATA_FOLDER_PATH)

    if DO_PREPROCESSING == True:
        manipulate_data(normal_data, abnormal_data, save_load)
    else:
        save_load.load_data_json(normal_data, abnormal_data)

    #autoencoder training
    model = AnomalyDetector(OPTIMIZER, LOSS, normal_data.train_data, normal_data.val_data)
    model.train(EPOCHS, BATCH_SIZE)
    model.plot_loss(PLOTS_FOLDER_PATH)


    # Save models in different formats
    create_folder_if_not_exists(MODELS_FOLDER_PATH)

    # Save as .tflite
    tflite_filepath = os.path.join(MODELS_FOLDER_PATH, "model.tflite")
    save_as_tflite(model.autoencoder, tflite_filepath)

    # Save as .pkl
    pkl_filepath = os.path.join(MODELS_FOLDER_PATH, "model.pkl")
    save_as_pkl(model.autoencoder, pkl_filepath)

    # Convert to C array
    subprocess.run(['xxd', '-i', tflite_filepath, os.path.join(MODELS_FOLDER_PATH, 'autoencoder.cc')])

    # Modify file names
    model_cc_path = os.path.join(MODELS_FOLDER_PATH, 'autoencoder.cc')
    replace_text = "model_tflite".replace('/', '_').replace('.', '_')
    subprocess.run(['sed', '-i', f's/{replace_text}/g_model/g', model_cc_path])