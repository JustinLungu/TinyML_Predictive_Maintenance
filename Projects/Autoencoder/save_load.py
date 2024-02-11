import os
import pandas as pd
import numpy as np
import json

class Saving_Loading:
    def __init__(self, folder_path) -> None:
        self.folder_path = folder_path

    def _save_to_csv(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    def _save_to_json(self, data, filename):
        with open(filename, 'w') as f:
            json.dump(data.tolist(), f)

    def _load_from_csv(self, filename):
        df = pd.read_csv(filename)
        return df.values

    def _load_from_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return np.array(data)

    def save_data_csv(self, normal_data, abnormal_data, window_size):
        # Create the folder if it doesn't exist
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        # Save normal data
        self._save_to_csv(normal_data.train_data.reshape(-1, window_size * 3), os.path.join(self.folder_path, 'normal_train.csv'))
        self._save_to_csv(normal_data.val_data.reshape(-1, window_size * 3), os.path.join(self.folder_path, 'normal_val.csv'))
        self._save_to_csv(normal_data.test_data.reshape(-1, window_size * 3), os.path.join(self.folder_path, 'normal_test.csv'))

        # Save abnormal data
        self._save_to_csv(abnormal_data.dataset.reshape(-1, window_size * 3), os.path.join(self.folder_path, 'abnormal.csv'))
        

    def save_data_json(self, normal_data, abnormal_data):
        # Save normal data in JSON format
        self._save_to_json(normal_data.train_data, os.path.join(self.folder_path, 'normal_train.json'))
        self._save_to_json(normal_data.val_data, os.path.join(self.folder_path, 'normal_val.json'))
        self._save_to_json(normal_data.test_data, os.path.join(self.folder_path, 'normal_test.json'))

        # Save abnormal data in JSON format
        self._save_to_json(abnormal_data.dataset, os.path.join(self.folder_path, 'abnormal.json'))


    def load_data_json(self, normal_data, abnormal_data):
        # Load normal data
        normal_data.train_data = self._load_from_json(os.path.join(self.folder_path, 'normal_train.json'))
        normal_data.val_data = self._load_from_json(os.path.join(self.folder_path, 'normal_val.json'))
        normal_data.test_data = self._load_from_json(os.path.join(self.folder_path, 'normal_test.json'))

        # Load abnormal data
        abnormal_data.dataset = self._load_from_json(os.path.join(self.folder_path, 'abnormal.json'))

    def load_data_csv(self, normal_data, abnormal_data, window_size):
        # Load normal data
        normal_data.train_data = self._load_from_csv(os.path.join(self.folder_path, 'normal_train.csv')).reshape(-1, window_size, 3)
        normal_data.val_data = self._load_from_csv(os.path.join(self.folder_path, 'normal_val.csv')).reshape(-1, window_size, 3)
        normal_data.test_data = self._load_from_csv(os.path.join(self.folder_path, 'normal_test.csv')).reshape(-1, window_size, 3)

        # Load abnormal data
        abnormal_data.dataset = self._load_from_csv(os.path.join(self.folder_path, 'abnormal.csv')).reshape(-1, window_size, 3)
