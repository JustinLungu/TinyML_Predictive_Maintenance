- autoencoder_v1.ipynb
The code used so far for the autoencoder and cnn in google colab

- cnn.ipynb
  The code used to train the CNN. Data files need to be uploaded to the session manually. Data for testing is set aside.
  24x3 segments with 50% overlap are created for training the model.
  The model is trained and saved. Then quantized.

  TO REVIEW:
  
      Not sure that the quantization works, although trying to do a classification with the quantized model results in the same probabilities for each class as the non-quantized one. I think that's a bit iffy - shouldn't the  model lose some of the accuracy? The patterns in the data are also very easy to learn, so maybe it is fine after all?

- the Arduino folder contains code to deploy models
