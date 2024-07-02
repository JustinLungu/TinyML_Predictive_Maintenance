Here are the models that we trained. We have two folders for each of the projects (Convolutional neural network and Autoencoder):

- In the Autoencoder folder we have the following files (all of the being from a non-quantized autoencoder):
    - a .h file which is the hexadecimal representation of the model
    - a .pkl file which is useful if we want to load the already trained model in python
    - a .tflite file which is useful for visualizing the model in Netron and needed to convert it to the .h file
    - a .cc file which is also a hexadecimal representation of the model but with a different extension

- In the CNN folder we have the following files:
  - cnn.tflite is the TFLite version of the NON-quantized CNN (trained in the cnn.ipynb)
  - cnn.cc is the file containing the binary representation of the non-quantized CNN
  
