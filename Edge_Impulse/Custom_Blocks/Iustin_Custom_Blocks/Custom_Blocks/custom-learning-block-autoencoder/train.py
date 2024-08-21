import sklearn # do this first, otherwise get a libgomp error?!
import argparse, os, sys, random, logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, BatchNormalization, TimeDistributed
from tensorflow.keras.optimizers import Adam
from conversion import convert_to_tf_lite, save_saved_model

# Lower TensorFlow log levels
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set random seeds for repeatable results
RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Load files
parser = argparse.ArgumentParser(description='Running custom Keras models in Edge Impulse')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--learning-rate', type=float, required=True)
parser.add_argument('--out-directory', type=str, required=True)

args, unknown = parser.parse_known_args()

if not os.path.exists(args.out_directory):
    os.mkdir(args.out_directory)

# grab train/test set and convert into TF Dataset
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
#Y_train = np.load(os.path.join(args.data_directory, 'Y_split_train.npy'))
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')
#Y_test = np.load(os.path.join(args.data_directory, 'Y_split_test.npy'))

#classes = Y_train.shape[1]

MODEL_INPUT_SHAPE = X_train.shape[1:]
WINDOW_SIZE = 24
AXES = 3

train_dataset = X_train
validation_dataset = X_test

#train_dataset = tf.data.Dataset.from_tensor_slices((X_train))
#validation_dataset = tf.data.Dataset.from_tensor_slices((X_test))

# place to put callbacks (e.g. to MLFlow or Weights & Biases)
callbacks = []


class Autoencoder(Model):
  # This is the constructor method for the Autoencoder class, where the architecture of the autoencoder is defined.
  def __init__(self):
    #calls the constructor of the parent class (tf.keras.Model) to properly initialize the model.
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      Dense(64, activation="relu", input_shape = (WINDOW_SIZE * AXES, )),
      Dense(32, activation="relu"),
      Dense(16, activation="tanh"),
      #layers.GaussianNoise(0.2)
      ])
    
    self.decoder = tf.keras.Sequential([
      Dense(16, activation="relu"),
      Dense(32, activation="relu"),
      Dense((WINDOW_SIZE * AXES), activation="sigmoid")])

  #This method defines the forward pass of the autoencoder model. It takes the input x, which represents the ECG signals.
  #The x is first passed through the encoder, and the resulting compressed representation is obtained.
  #The compressed representation is then passed through the decoder to reconstruct the original ECG signals.
  #The reconstructed signals are returned as the output of the autoencoder.
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded



# model architecture
'''
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(WINDOW_SIZE * AXES,)))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="tanh"))
model.add(Dense(32, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(WINDOW_SIZE * AXES, activation="sigmoid"))
'''


# this controls the learning rate
opt = Adam(learning_rate=args.learning_rate, beta_1=0.9, beta_2=0.999) # lr =0.001
LOSS = "mae"
# this controls the batch size, or you can manipulate the tf.data.Dataset objects yourself
BATCH_SIZE = 1024
#train_dataset_batch = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
#validation_dataset_batch = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

autoencoder = Autoencoder()
autoencoder.compile(optimizer = opt, loss = LOSS)

train_dataset = train_dataset.reshape(-1, WINDOW_SIZE * AXES)
validation_dataset = validation_dataset.reshape(-1, WINDOW_SIZE * AXES)
#model.fit(train_dataset_batch, epochs=args.epochs, validation_data=validation_dataset_batch, verbose=2, callbacks=callbacks) # epochs = 10
autoencoder.fit(train_dataset, 
          epochs = args.epochs, 
          batch_size = BATCH_SIZE,
          validation_data = validation_dataset, 
          shuffle = False,
          verbose=2, 
          callbacks=callbacks) # epochs = 10



print('')
print('Training network OK')
print('')

# Use this flag to disable per-channel quantization for a model.
# This can reduce RAM usage for convolutional models, but may have
# an impact on accuracy.
disable_per_channel_quantization = False

# Save the model to disk
save_saved_model(autoencoder, args.out_directory)

# Create tflite files (f32 / i8)
convert_to_tf_lite(autoencoder, args.out_directory, validation_dataset, MODEL_INPUT_SHAPE,
    'model.tflite', 'model_quantized_int8_io.tflite', disable_per_channel_quantization)