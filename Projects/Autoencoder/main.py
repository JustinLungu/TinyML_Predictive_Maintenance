import numpy as np
import sys
from data_preprocessing import Data, Preprocessing

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

    normal_data.data_split_window(train_ratio = 0.7, val_ratio = 0.2, test_ratio = 0.1, window_size = 24)
    abnormal_data.dataset = abnormal_data.make_windows(abnormal_data.dataset, window_size = 24)
    print(normal_data.train_data.shape)