import numpy as np
import sys
from data_preprocessing import Data, Preprocessing
import pandas as pd
import json
import os

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
WINDOW_SIZE = 24
#x,y,z accelerometer data
DATA_SHAPE = 3

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


def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data.tolist(), f)

def save_data(normal_data: Data, abnormal_data: Data, folder_path):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save normal data
    save_to_csv(normal_data.train_data.reshape(-1, 3), os.path.join(folder_path, 'normal_train.csv'))
    save_to_csv(normal_data.val_data.reshape(-1, 3), os.path.join(folder_path, 'normal_val.csv'))
    save_to_csv(normal_data.test_data.reshape(-1, 3), os.path.join(folder_path, 'normal_test.csv'))
    # Save normal data in JSON format
    save_to_json(normal_data.train_data, os.path.join(folder_path, 'normal_train.json'))
    save_to_json(normal_data.val_data, os.path.join(folder_path, 'normal_val.json'))
    save_to_json(normal_data.test_data, os.path.join(folder_path, 'normal_test.json'))

    # Save abnormal data
    save_to_csv(abnormal_data.dataset.reshape(-1, 3), os.path.join(folder_path, 'abnormal.csv'))
    # Save abnormal data in JSON format
    save_to_json(abnormal_data.dataset, os.path.join(folder_path, 'abnormal.json'))

if __name__ == "__main__":
    #set the threshhold of prinitng data to console to maximum value
    #so avoid the loss of data on console while displaying
    np.set_printoptions(threshold=sys.maxsize)
    # setting up a random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)

    normal_data = Data(capture = "1", hertz = "60", volume = "30")
    abnormal_data = Data(capture = "2", hertz = "40", volume = "60")

    #normal_data.plotVibPattern(datapoints = 2000)
    #abnormal_data.plotVibPattern(datapoints = 2000)

    normal_data.data_split_window(TRAIN_RATIO, VAL_RATIO, TEST_RATIO, WINDOW_SIZE)
    abnormal_data.dataset = abnormal_data.make_windows(abnormal_data.dataset, WINDOW_SIZE)
    
    normalization(normal_data, abnormal_data)

    # Save the preprocessed data in the specified folder
    save_data(normal_data, abnormal_data, "Projects/Autoencoder/Preprocessed Data")