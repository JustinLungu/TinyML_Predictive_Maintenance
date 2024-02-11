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

        if self.decoded_windows is None:  # Check if decoded_windows is None
            raise DecodedWindowsError("Decoded windows not available. Call predict() first.")


        # Select a few samples for visualization
        samples_to_visualize = self.data[:num_samples]
        decoded_to_visualize = self.decoded_windows[:num_samples]

        # Plot the original and decoded samples
        for i in range(num_samples):
            plt.figure(figsize=(8, 4))
            plt.subplot(2, 1, 1)
            plt.plot(samples_to_visualize[i].flatten(), label='Original')
            plt.title(f'{self.type} - Sample {i + 1} - Original')

            plt.subplot(2, 1, 2)
            plt.plot(decoded_to_visualize[i].flatten(), label='Decoded', linestyle='--')
            plt.title(f'{self.type} - Sample {i + 1} - Decoded')

            plt.tight_layout()

            # Create the folder if it doesn't exist
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            plt.savefig(os.path.join(folder_path, f"{self.type}_prediction.png"))
            plt.show()