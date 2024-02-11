import numpy as np
import sys
from data_preprocessing import Data, Preprocessing
from save_load_data import Saving_Loading
from model import AnomalyDetector
from save_model import Save_Model
from evaluation import Evaluation


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
NR_SAMPLES_VISUALIZE = 5

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

    #save model
    save_model = Save_Model()
    save_model.save(model.autoencoder, MODELS_FOLDER_PATH)

    #evaluation
    normal_eval = Evaluation(normal_data.test_data, "Normal")
    abnormal_eval = Evaluation(abnormal_data.dataset, "Anomalous")

    normal_eval.predict(model.autoencoder)
    abnormal_eval.predict(model.autoencoder)


    mse_eval_normal = normal_eval.calc_mse()
    mse_eval_abnormal = abnormal_eval.calc_mse()
    print(f"Difference between the mse of Normal and Anomalous predictions: {abs(mse_eval_normal - mse_eval_abnormal)}")

    normal_eval.visualize(NR_SAMPLES_VISUALIZE, PLOTS_FOLDER_PATH)
    abnormal_eval.visualize(NR_SAMPLES_VISUALIZE, PLOTS_FOLDER_PATH)

