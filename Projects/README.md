In this folder you will find the Convolutional Neural Network (CNN), Autoencoder, and Arduino code, each in its own folder.
- In the CNN folder we have the code to create the CNN model written in a Jupyter notebook.
- In the Autoencoder folder, we have the code written in Python, as well as some additional plots and saved preprocessed data.
- In the Arduino folder, we have for each project the code (.ino) to run on the microcontroller, and the two useful models (.h and .tflite) from which the .h file is used by the .ino file.


*Additional notes*:
- autoencoder_v1.ipynb
The code used so far for the autoencoder and cnn in google colab

- cnn.ipynb
  The code used to train the CNN. Data files need to be uploaded to the session manually. Data for testing is set aside.
  24x3 segments with 50% overlap are created for training the model.
  The model is trained and saved.
  


- the Arduino folder contains code to deploy models


- 50 KB works with 2000 KTensorArea (doesn't give error anymore regarding size when uploading)
- 9 MIL KTensorArea does not work
- don't go over 250KB length size in your .h file for deploying the model
