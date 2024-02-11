from keras.models import Model
from keras import layers, losses
import tensorflow as tf

class AnomalyDetector(Model):
  # This is the constructor method for the AnomalyDetector class, where the architecture of the autoencoder is defined.
  def __init__(self):
    #calls the constructor of the parent class (tf.keras.Model) to properly initialize the model.
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(128, activation="relu", input_shape = (24 * 3, )),
      layers.Dense(64, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")]) # Smallest Layer Defined Here

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense((24 * 3), activation="sigmoid")])

  #This method defines the forward pass of the autoencoder model. It takes the input x, which represents the ECG signals.
  #The x is first passed through the encoder, and the resulting compressed representation is obtained.
  #The compressed representation is then passed through the decoder to reconstruct the original ECG signals.
  #The reconstructed signals are returned as the output of the autoencoder.
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded