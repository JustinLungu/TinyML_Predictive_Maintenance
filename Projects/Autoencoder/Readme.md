This folder contains the code to train the autoencoder model.

- In the additional 2 folder we have the saved preprocessed data such that we don't have to preprocess it every time and some plots made to evaluate our model.
- In regards to each python file:
  -  main.py --> calls all the other files to perform the data_preprocessing, model_training, model_evaluation, model_saving, and model_conversion.
  -  data_preprocessing.py --> contains the code used to create the window sizes based on the values defined in main.py as well as the normalization step.
  -  save_load_data.py --> saves the preprocessed data in the "Preprocessed Data" folder and it also has code to load the data.
  -  model.py --> Contains the architecture of the autoencoder as well as the training step and plotting the loss function.
  -  save_model.py --> saves the trained model in the Models/Autoendoer/ folder of our repository. It saves in in the following formats: .h, .pkl, .tflite, .cc. This file also has the ability to load the models back in python.
  -  evaluation.py --> evaluates the model based on the testing data. It calculates the Mean Absolute Error (MAE) of the Correct data (original vs re-created by autoencoder) and the MAE of the Anomaly data (original vs re-created by autoencoder) and then it computes the difference between the two to see how significant it is. It also plots the predictions of the model.
 
The other two files are not related directly to the whole pipeline but provide us with some additional information:
  - predict_one.py --> it load the already trained model and it give it one window to predict and we get the MAE and predicted values of the matrix.
  - find_treshold.py --> helps up find a somewhat accurate threshold for our model to distinguish between anomaly and normal signal.
