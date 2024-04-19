import os
import tensorflow as tf
import joblib
import subprocess
import numpy as np

class Save_Model(): 

    def save(self, model, folder_path):
        # Save models in different formats
        self._create_folder_if_not_exists(folder_path)

        # Save as .tflite
        tflite_filepath = os.path.join(folder_path, "autoencoder_model.tflite")
        tflite_model = self._save_as_tflite(model, tflite_filepath)

        # Write TFLite model to a C source (or header) file
        c_model_name = os.path.join(folder_path, "autoencoder_model")
        with open(c_model_name + '.h', 'w') as file:
            file.write(self._hex_to_c_array(tflite_model, c_model_name))

        # Save as .pkl
        pkl_filepath = os.path.join(folder_path, "autoencoder_model.pkl")
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
        return tflite_model

    # Function to save model as .pkl
    def _save_as_pkl(self, model, filepath):
        joblib.dump(model, filepath)

    # Function to create folder if it doesn't exist
    def _create_folder_if_not_exists(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Function to convert hex data to C array format
    def _hex_to_c_array(self, hex_data, var_name):
        c_str = ''

        # Create header guard
        c_str += '#ifndef ' + var_name.upper() + '_H\n'
        c_str += '#define ' + var_name.upper() + '_H\n\n'

        # Add array length at top of file
        c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'

        # Declare C variable
        c_str += 'unsigned char ' + var_name + '[] = {'
        hex_array = []
        for i, val in enumerate(hex_data) :

            # Construct string from hex
            hex_str = format(val, '#04x')

            # Add formatting so each line stays within 80 characters
            if (i + 1) < len(hex_data):
                hex_str += ','
            if (i + 1) % 12 == 0:
                hex_str += '\n '
            hex_array.append(hex_str)

        # Add closing brace
        c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

        # Close out header guard
        c_str += '#endif //' + var_name.upper() + '_H'

        return c_str


class Load_Model():
    def predict_tflite(self, filepath, input_data):
        interpreter = tf.lite.Interpreter(model_path=filepath)
        interpreter.allocate_tensors()

        #Set input tensors
        input_data = input_data.astype(np.float32)
        input_index = interpreter.get_input_details()[0]['index']
        interpreter.set_tensor(input_index, input_data)

        #Invoke inference
        interpreter.invoke()

        #Retrieve output tensors
        output_index = interpreter.get_output_details()[0]['index']
        output_data = interpreter.get_tensor(output_index)
        return output_data

    def load_pkl(self, filepath):
        return joblib.load(filepath)

    def load_c_array(self, array, array_len):
        # Convert C array to numpy array
        byte_array = np.frombuffer(array, dtype=np.uint8)
        return byte_array[:array_len]

    def load_from_folder(self, folder_path):
        # Load TFLite model
        tflite_filepath = os.path.join(folder_path, "autoencoder_model.tflite")
        tflite_model = self.load_tflite(tflite_filepath)

        # Load Pickle model
        pkl_filepath = os.path.join(folder_path, "autoencoder_model.pkl")
        pkl_model = self.load_pkl(pkl_filepath)

        return tflite_model, pkl_model