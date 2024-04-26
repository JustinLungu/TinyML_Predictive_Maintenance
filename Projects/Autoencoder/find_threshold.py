import numpy as np
import sys
from data_preprocessing import Data, Preprocessing
from save_load_data import Saving_Loading
from model import AnomalyDetector
from save_model import Load_Model
from evaluation import Evaluation

import matplotlib.pyplot as plt
from keras.optimizers import Adam

DATA_FOLDER_PATH = "Projects/Autoencoder/Preprocessed Data"
PLOTS_FOLDER_PATH = "Projects/Autoencoder/Plots"
MODELS_FOLDER_PATH = "Models/Autoencoder"
WINDOW_SIZE = 24
DATA_SHAPE = 3

mae_array_norm = []
mae_array_abnorm = []

def plot_data(data):
    # Assuming abnormal_plot is a window_size x 3 matrix
    x_axis = data[:, 0]  # Extracting x-axis data
    y_axis = data[:, 1]  # Extracting y-axis data
    z_axis = data[:, 2]  # Extracting z-axis data

    # Plotting x-axis
    plt.plot(x_axis, label='X-axis')

    # Plotting y-axis
    plt.plot(y_axis, label='Y-axis')

    # Plotting z-axis
    plt.plot(z_axis, label='Z-axis')

    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.title(f'Preporcessed data')
    plt.legend()
    plt.show()

def calc_mae(real_sample, decoded_sample):

        mae = np.mean(np.abs(real_sample - decoded_sample))
        
        return mae


def mae_per_window(data, model: Load_Model, mae_array):

    for window in data:
        window = window.reshape(72)
        output = model.predict_tflite(MODELS_FOLDER_PATH, window)
        mae_window = calc_mae(window, output)
        mae_array.append(mae_window)


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
    save_load.load_data_json(normal_data, abnormal_data)

    #print(abnormal_data.dataset.shape)

    #plot_data(abnormal_data.dataset[1])

    load_model = Load_Model()


    mae_per_window(normal_data.train_data, load_model, mae_array_norm)
    mae_per_window(normal_data.val_data, load_model, mae_array_norm)
    mae_per_window(normal_data.test_data, load_model, mae_array_norm)

    mae_array_norm = np.asarray(mae_array_norm)
    
    print('NORMAL Recommended threshold (3x std dev + avg):', (3*np.std(mae_array_norm)) + np.average(mae_array_norm))


    #mae_per_window(abnormal_data.dataset, load_model, mae_array_abnorm)
    #mae_array_abnorm = np.asarray(mae_array_abnorm)
    #print('ABNORMAL Recommended threshold (3x std dev + avg):', (3*np.std(mae_array_abnorm)) + np.average(mae_array_abnorm))

    