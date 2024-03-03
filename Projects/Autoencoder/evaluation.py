import numpy as np
import matplotlib.pyplot as plt
import os
import random

class DecodedWindowsError(Exception):
    pass

class Evaluation():

    def __init__(self, data, data_type, window_size) -> None:
        self.data = data
        self.type = data_type
        self.decoded_windows = None
        self.encoded_windows = None
        self.window_size = window_size

    def predict(self, model):
        test_reshaped = self.data.reshape(self.data.shape[0], -1)
        self.encoded_windows = model.encoder(test_reshaped).numpy()
        self.decoded_windows = model.decoder(self.encoded_windows).numpy()
        self.decoded_windows = self.decoded_windows.reshape(-1, self.window_size, 3)

    def calc_mse(self, sample_size = 0, type = None):

        if self.decoded_windows is None:  # Check if decoded_windows is None
            raise DecodedWindowsError("Decoded windows not available. Call predict() first.")
        
        
        
        if type == "Anomaly":
            # Flatten the data for MSE calculation
            #random_number = random.randint(0, self.decoded_windows.shape[0] - (sample_size-1))
            sample_flattened = self.data[:sample_size].reshape(-1, self.window_size * 3)
            decoded_flattened = self.decoded_windows[:sample_size].reshape(-1, self.window_size * 3)
        else:
            sample_flattened = self.data.reshape(-1, self.window_size * 3)
            decoded_flattened = self.decoded_windows.reshape(-1, self.window_size * 3)
        print(f"Shape for mse for decoded windows: {len(decoded_flattened)}")

        mse = np.mean((sample_flattened - decoded_flattened)**2)

        print(f'Mean Squared Error for {self.type} data: {mse}')

        return mse

    def visualize_window(self, num_samples, folder_path):
        if self.decoded_windows is None:
            raise DecodedWindowsError("Decoded windows not available. Call predict() first.")

        num_figures = -(-num_samples // 4)  # Ceiling division to determine the number of figures needed

        for fig_num in range(num_figures):
            plt.figure(figsize=(12, 8))

            for i in range(4):
                sample_index = fig_num * 4 + i
                if sample_index >= num_samples:
                    break

                plt.subplot(2, 2, i + 1)

                # Raw data x,y,z 24x3 matrix
                #print(self.data.shape)
                window_raw = self.data[i]
                #print(window_raw.shape)
                raw_x_axis = window_raw[:, 0]  # Extracting x-axis data
                raw_y_axis = window_raw[:, 1]  # Extracting y-axis data
                raw_z_axis = window_raw[:, 2]  # Extracting z-axis data

                # Plotting x-axis
                plt.plot(raw_x_axis, label='Original X-axis')

                # Plotting y-axis
                plt.plot(raw_y_axis, label='Original Y-axis')

                # Plotting z-axis
                plt.plot(raw_z_axis, label='Original Z-axis')

                # Decoded data x,y,z 24x3 matrix
                window_dec = self.decoded_windows[i]
                decoded_x_axis = window_dec[:, 0]  # Extracting x-axis data
                decoded_y_axis = window_dec[:, 1]  # Extracting y-axis data
                decoded_z_axis = window_dec[:, 2]  # Extracting z-axis data

                # Plotting x-axis
                plt.plot(decoded_x_axis, label='Decoded X-axis', linestyle='--')

                # Plotting y-axis
                plt.plot(decoded_y_axis, label='Decoded Y-axis', linestyle='--')

                # Plotting z-axis
                plt.plot(decoded_z_axis, label='Decoded Z-axis', linestyle='--')
                #plt.plot(self.data[sample_index].flatten(), label='Original')
                #plt.plot(self.decoded_windows[sample_index].flatten(), label='Decoded', linestyle='--')
                plt.title(f'{self.type} - Sample {sample_index + 1}')
                plt.legend()

            plt.tight_layout()

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            plt.savefig(os.path.join(folder_path, f"{self.type}_prediction_{fig_num}.png"))
            plt.show()