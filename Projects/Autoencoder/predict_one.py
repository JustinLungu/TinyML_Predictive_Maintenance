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


    reading = [0.46838379, 0.48919678, 0.74868774, 0.46890259, 0.48904419, 0.74890137, 0.46881104, 0.48913574, 0.74896240, 0.46868896, 0.48947144, 0.74896240, 0.46847534, 0.48898315, 0.74911499, 0.46856689, 0.48928833, 0.74868774, 0.46844482, 0.48916626, 0.74914551, 0.46856689, 0.48892212, 0.74865723, 0.46859741, 0.48889160, 0.74877930, 0.46844482, 0.48876953, 0.74868774, 0.46817017, 0.48883057, 0.74899292, 0.46862793, 0.48928833, 0.74832153, 0.46859741, 0.48919678, 0.74905396, 0.46829224, 0.48922729, 0.74893188, 0.46856689, 0.48913574, 0.74911499, 0.46871948, 0.48907471, 0.74914551, 0.46856689, 0.48934937, 0.74923706, 0.46841431, 0.48889160, 0.74838257, 0.46908569, 0.48895264, 0.74920654, 0.46878052, 0.48959351, 0.74926758, 0.46817017, 0.48889160, 0.74880981, 0.46929932, 0.48919678, 0.74896240, 0.46859741, 0.48925781, 0.74908447, 0.46911621, 0.48873901, 0.74853516]
    
    input_data = np.array(reading)
    #input_data = np.expand_dims(input_data, axis = 0) # add a batch dimension


    #Predict on the reading
    loader = Load_Model()
    output = loader.predict_tflite(MODELS_FOLDER_PATH, input_data)

    print("Prediction of the one reading: ", output)

    #calculate mean absolute error
    mae = np.mean(np.abs(reading - output))
    print("MAE for the one reading: ", mae)
