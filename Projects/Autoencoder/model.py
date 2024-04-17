from keras.models import Model
from keras import layers, losses
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle

class Autoencoder(Model):
  # This is the constructor method for the Autoencoder class, where the architecture of the autoencoder is defined.
  def __init__(self, window_size):
    #calls the constructor of the parent class (tf.keras.Model) to properly initialize the model.
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu", input_shape = (window_size * 3, )),
      layers.Dense(16, activation="relu"),
      #layers.Dense(16, activation="relu"),
      #layers.Dense(8, activation="relu"),
      #layers.Dense(4, activation="relu"), # Smallest Layer Defined Here
      #layers.Dense(16, activation="tanh")
      #layers.GaussianNoise(0.2)
      ])
    
    self.decoder = tf.keras.Sequential([
      #layers.Dense(8, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense((window_size * 3), activation="sigmoid")])

  #This method defines the forward pass of the autoencoder model. It takes the input x, which represents the ECG signals.
  #The x is first passed through the encoder, and the resulting compressed representation is obtained.
  #The compressed representation is then passed through the decoder to reconstruct the original ECG signals.
  #The reconstructed signals are returned as the output of the autoencoder.
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  


def train(nr_epochs, nr_batches, autoencoder, train_data, val_data):
    #print(train_data.shape)
    history = autoencoder.fit(train_data, train_data,
                              epochs = nr_epochs,
                              batch_size = nr_batches,
                              validation_data=(val_data, val_data),
                              shuffle=False)
    
    return history

def plot_loss(folder_path, history):

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    
    # Create the Plots folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, "overfitting_plot.png"))
    plt.show()
        

