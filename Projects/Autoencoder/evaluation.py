import numpy as np
import matplotlib.pyplot as plt
import os

class DecodedWindowsError(Exception):
    pass

class Evaluation():

    def __init__(self, data, data_type) -> None:
        self.data = data
        self.type = data_type
        self.decoded_windows = None
        self.encoded_windows = None

    def predict(self, model):
        test_reshaped = self.data.reshape(self.data.shape[0], -1)
        self.encoded_windows = model.encoder(test_reshaped).numpy()
        self.decoded_windows = model.decoder(self.encoded_windows).numpy()

    def calc_mse(self):

        if self.decoded_windows is None:  # Check if decoded_windows is None
            raise DecodedWindowsError("Decoded windows not available. Call predict() first.")

        # Flatten the data for MSE calculation
        sample_flattened = self.data.reshape(-1, 24 * 3)
        decoded_flattened = self.decoded_windows.reshape(-1, 24 * 3)

        # Calculate mean squared error
        mse = np.mean((sample_flattened - decoded_flattened)**2)

        print(f'Mean Squared Error for {self.type} data: {mse}')
        return mse

    def visualize(self, num_samples, folder_path):
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
                plt.plot(self.data[sample_index].flatten(), label='Original')
                plt.plot(self.decoded_windows[sample_index].flatten(), label='Decoded', linestyle='--')
                plt.title(f'{self.type} - Sample {sample_index + 1}')
                plt.legend()

            plt.tight_layout()

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            plt.savefig(os.path.join(folder_path, f"{self.type}_prediction_{fig_num}.png"))
            plt.show()