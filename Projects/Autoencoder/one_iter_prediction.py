import numpy as np
import sys
from data_preprocessing import Data, Preprocessing
from save_model import Load_Model
from evaluation import Evaluation

import matplotlib.pyplot as plt

WINDOW_SIZE = 96
DATA_SHAPE = 3 #x,y,z accelerometer data
MODELS_FOLDER_PATH = "Models/Autoencoder/autoencoder_model.pkl"
PLOTS_FOLDER_PATH = "Projects/Autoencoder/Plots"


def plot_data(data, type):
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
    plt.title(f'Window {type} normalization')
    plt.legend()
    plt.show()

def normalization(window_data):
    preprocess = Preprocessing(WINDOW_SIZE)

    '''
    normalized_windows = []
    for window in window_data:
        # Flatten window to 2D array
        window_flat = window.reshape(-1, DATA_SHAPE)  
        normalized_window_flat = preprocess.min_max_scale_fit(window_flat)
            
        normalized_window_flat_rounded = np.round(normalized_window_flat, 2)  
        normalized_window = normalized_window_flat_rounded.reshape(-1, WINDOW_SIZE, DATA_SHAPE)
        normalized_windows.append(normalized_window)
    '''
    print(window_data.shape)

    window = window_data
    window_plot = window_data[0]

    plot_data(window_plot, "Before")

    window = window.reshape(-1, WINDOW_SIZE * DATA_SHAPE)
    window = preprocess.normalize_by_2(window)
    window_data = window.reshape(-1, WINDOW_SIZE, DATA_SHAPE)

    window_plot = window_data[0]
    plot_data(window_plot, "After")

    return np.array(window_data)



if __name__ == "__main__":

    #set the threshhold of prinitng data to console to maximum value
    #so avoid the loss of data on console while displaying
    np.set_printoptions(threshold=sys.maxsize)
    # setting up a random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)

    #shape(1, 24, 3): Raw data directly from the Arduino
    window_raw = np.array([[
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.03, -0.04, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        #####################
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.03, -0.04, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        #####################
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.03, -0.04, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        ######################
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.03, -0.04, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.04, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.03, -0.05, 1.00],
        [0.04, -0.05, 1.00]
        ]])
    #make it window size compatible with the model
    window_raw_norm = normalization(window_raw)
    print("Data taken directly from Arduino: \n", window_raw)
    print("Data taken directly from Arduino NORMALIZED: \n", window_raw_norm)
    

    normal_data = Data(capture = "1", hertz = "60", volume = "30")
    #make it window size compatible with the model
    window_normal_data = normal_data.make_windows(normal_data.dataset[:WINDOW_SIZE], WINDOW_SIZE)
    print("First 24 readings from the normal dataset in window size 24 format: \n", window_normal_data)
    window_normal_data_norm = normalization(window_normal_data)
    print("First 24 readings from the normal dataset in window size 24 format NORMALZIED: \n", window_normal_data_norm)



    #Load model
    loader = Load_Model()
    model = loader.load_pkl(MODELS_FOLDER_PATH)


    #Predictions
    raw_eval = Evaluation(window_raw_norm, "Raw Arduino", WINDOW_SIZE)
    normal_eval = Evaluation(window_normal_data_norm, "1 Window Normal Data", WINDOW_SIZE)

    raw_eval.predict(model)
    mse_raw_eval = raw_eval.calc_mae()

    normal_eval.predict(model)
    mse_eval_normal = normal_eval.calc_mae()
    print(f"Difference between the mae of Normal and Arduino predictions: {abs(mse_eval_normal - mse_raw_eval)}")


    raw_eval.visualize_window(num_samples = 1, folder_path = PLOTS_FOLDER_PATH)
    normal_eval.visualize_window(num_samples = 1, folder_path = PLOTS_FOLDER_PATH)



    print("Output of the Encoder \n", raw_eval.encoded_windows)

    print_decoded = raw_eval.decoded_windows
    print("Output of the Decoder: \n", print_decoded.reshape(WINDOW_SIZE, 3))