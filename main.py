# Main file from which we run the support functions
from data_prep import Data

if __name__ == '__main__':

    ################################# LOAD DATA #########################################
    normal_data = Data("1", "60", "30")
    anomalous_data = Data("2", "40", "60")
    normal_data.loadData()
    anomalous_data.loadData()

    #plot the datasets
    nr_readings = 2000
    normal_data.plotVibPattern(nr_readings)
    anomalous_data.plotVibPattern(nr_readings)

    ################################# SPLIT DATA ########################################

    ################################# NORMALIZE DATA ####################################

    ############################### PREPROCESS DATA #####################################
    """
    Do it separately for each dataset
    - look at frequency domain more
    - FFT function
    - check if the normal/abnormal are different enough in the frequency domain
    - energy features
    - wavelet transform
    - etc. (TBD)

    Finally, choose the most promising ones (for future: we can do MTL and maybe include more tasks)
    Useful articles to check for future: 
    - "Autoencoder-based multi-task learning for imputation and classification of incomplete data"
    """

    ############################## AUTOENCODER ARCHITECTURE ##############################

    ############################## TRAIN THE MODEL #######################################

    ############################## CHECK OVERFITTING/UNDERFITTING #########################
    """
    Do hyperparameter tuning for fix problems in case they appear
    """

    ############################## EVALUATE TRAINING ######################################
    """
    - See the error difference for both normal and abnormal
    - Plot loss histogram for both datasets
    - Maybe ROC & AUC
    """

    ############################## THRESHOLD PICK #########################################
    """
    Decide on a number which allows us to accept false positives
    """

    ############################### QUANTIZATION ##########################################
    """
    Make sure the model is under 256 KB
    """

    ############################### CONVERT TO TFLITE ######################################

