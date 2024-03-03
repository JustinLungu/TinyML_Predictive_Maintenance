import numpy as np
import sys
from data_preprocessing import Data, Preprocessing
from save_load_data import Saving_Loading
from model import AnomalyDetector
from save_model import Save_Model
from evaluation import Evaluation

import matplotlib.pyplot as plt

#if you modify any constant make sure to set this to true
#otherwise you can keep it at false
DO_PREPROCESSING = True

WINDOW_SIZE = 24
DATAPOINTS_PLOTTING = 2000
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
DATA_SHAPE = 3 #x,y,z accelerometer data
DATA_FOLDER_PATH = "Projects/Autoencoder/Preprocessed Data"
PLOTS_FOLDER_PATH = "Projects/Autoencoder/Plots"
MODELS_FOLDER_PATH = "Models/Autoencoder"
OPTIMIZER = "adam"
LOSS = "mse"
EPOCHS = 20
BATCH_SIZE = 512
NR_SAMPLES_VISUALIZE = 4


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

def normalization(normal_data: Data, abnormal_data: Data):
    preprocess = Preprocessing(WINDOW_SIZE)

    #normal data normalization (train,val,test)
    train = normal_data.train_data
    val = normal_data.val_data
    test = normal_data.test_data
    plot_data(test[1], "Test before")

    train = train.reshape(-1, WINDOW_SIZE * DATA_SHAPE)
    val = val.reshape(-1, WINDOW_SIZE * DATA_SHAPE)
    test = test.reshape(-1, WINDOW_SIZE * DATA_SHAPE)

    train = preprocess.normalize_by_2(train)
    val = preprocess.normalize_by_2(val)
    test = preprocess.normalize_by_2(test)

    normal_data.train_data = train.reshape(-1, WINDOW_SIZE, DATA_SHAPE)
    normal_data.val_data = val.reshape(-1, WINDOW_SIZE, DATA_SHAPE)
    normal_data.test_data = test.reshape(-1, WINDOW_SIZE, DATA_SHAPE)

    plot_data(normal_data.test_data[1], "Test after")

    #abnormal data (the entire dataset)
    abnormal = abnormal_data.dataset
    abnormal_plot = abnormal_data.dataset[1]

    plot_data(abnormal_plot, "Anomalous before")

    abnormal = abnormal.reshape(-1, WINDOW_SIZE * DATA_SHAPE)
    abnormal = preprocess.normalize_by_2(abnormal)
    abnormal_data.dataset = abnormal.reshape(-1, WINDOW_SIZE, DATA_SHAPE)

    abnormal_plot = abnormal_data.dataset[1]
    plot_data(abnormal_plot, "Anomalous after")


def manipulate_data(normal_data, abnormal_data, save_load):
    # plot datasets
    normal_data.plotVibPattern(datapoints = DATAPOINTS_PLOTTING)
    abnormal_data.plotVibPattern(datapoints = DATAPOINTS_PLOTTING)
    normal_data.data_split_window(TRAIN_RATIO, VAL_RATIO, TEST_RATIO, WINDOW_SIZE)
    abnormal_data.dataset = abnormal_data.make_windows(abnormal_data.dataset, WINDOW_SIZE)
    
    normalization(normal_data, abnormal_data)

    # Save the preprocessed data in the specified folder
    save_load.save_data_json(normal_data, abnormal_data)

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
    model = AnomalyDetector(OPTIMIZER, LOSS, normal_data.train_data, normal_data.val_data, WINDOW_SIZE)
    model.train(EPOCHS, BATCH_SIZE)
    model.plot_loss(PLOTS_FOLDER_PATH)

    #save model
    save_model = Save_Model()
    save_model.save(model.autoencoder, MODELS_FOLDER_PATH)

    #evaluation
    normal_eval = Evaluation(normal_data.test_data, "Normal", WINDOW_SIZE)
    abnormal_eval = Evaluation(abnormal_data.dataset, "Anomalous", WINDOW_SIZE)

    normal_eval.predict(model.autoencoder)
    abnormal_eval.predict(model.autoencoder)


    mse_eval_normal = normal_eval.calc_mse()
    mse_eval_abnormal = abnormal_eval.calc_mse()
    print(f"Difference between the mse of Normal and Anomalous predictions: {abs(mse_eval_normal - mse_eval_abnormal)}")

    normal_eval.visualize_window(NR_SAMPLES_VISUALIZE, PLOTS_FOLDER_PATH)
    abnormal_eval.visualize_window(NR_SAMPLES_VISUALIZE, PLOTS_FOLDER_PATH)
