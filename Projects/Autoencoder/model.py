from keras.models import Model
from keras import layers, losses
import tensorflow as tf

class Autoencoder(Model):
  # This is the constructor method for the Autoencoder class, where the architecture of the autoencoder is defined.
  def __init__(self):
    #calls the constructor of the parent class (tf.keras.Model) to properly initialize the model.
    super(Autoencoder, self).__init__()
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
  
class AnomalyDetector:
    def __init__(self, optimizer, loss, train_data, val_data) -> None:
        self.model = Autoencoder()
        self.model.compile(optimizer = optimizer, loss = loss)

        # Flatten the input data before feeding it into the model
        self.train_data = train_data.reshape(-1, 24 * 3)
        self.val_data = val_data.reshape(-1, 24 * 3)

        self.history = None

    def train(self, nr_epochs, nr_batches):
        self.history = self.model.fit(self.train_data, self.train_data,
          epochs = nr_epochs,
          batch_size = nr_batches,
          validation_data=(self.val_data, self.val_data),
          shuffle=True)
        
        return self.history
        

