import os
import tensorflow as tf
import joblib
import subprocess

class Save_Model(): 

    def save(self, model, folder_path):
        # Save models in different formats
        self._create_folder_if_not_exists(folder_path)

        # Save as .tflite
        tflite_filepath = os.path.join(folder_path, "model.tflite")
        self._save_as_tflite(model, tflite_filepath)

        # Save as .pkl
        pkl_filepath = os.path.join(folder_path, "model.pkl")
        self._save_as_pkl(model, pkl_filepath)

        # Convert to C array
        subprocess.run(['xxd', '-i', tflite_filepath, os.path.join(folder_path, 'autoencoder.cc')])

        # Modify file names
        model_cc_path = os.path.join(folder_path, 'autoencoder.cc')
        replace_text = "model_tflite".replace('/', '_').replace('.', '_')
        subprocess.run(['sed', '-i', f's/{replace_text}/g_model/g', model_cc_path])

    # Function to save model as .tflite
    def _save_as_tflite(self, model, filepath):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(filepath, 'wb') as f:
            f.write(tflite_model)

    # Function to save model as .pkl
    def _save_as_pkl(self, model, filepath):
        joblib.dump(model, filepath)

    # Function to create folder if it doesn't exist
    def _create_folder_if_not_exists(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)