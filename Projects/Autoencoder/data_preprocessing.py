import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, capture, hertz, volume) -> None:
        self.hertz = hertz
        self.volume = volume
        # traverse into Collected Data
        data_folder = os.path.join("Collected Data")
        self.filename = os.path.join(data_folder, "capture" + capture + "_" + hertz + "hz_" + volume + "vol.txt")
        self.dataset = self._readData(self.filename)
        # Add a new column for sequential timestamps
        self.dataset['timestamp'] = np.arange(1, len(self.dataset) + 1)
        
        self.train_data = None
        self.val_data = None
        self.test_data = None


    #--------------To read the data from a file
    # defining function for loading the dataset
    def _readData(self, filePath):
        # attributes of the dataset
        columnNames = ['x-axis','y-axis','z-axis']
        #read the specified file using pandas function and return the data
        data = pd.read_csv(filePath, header=None, names=columnNames, sep='\t', na_values=';')
        return data
    
    # defining the function to plot a single axis data
    # setup color, title, limit and add grid.
    def _plotAxis(self, axis, x, y, title):
        axis.plot(x, y, color='green', linewidth=1)
        axis.set_title(title)
        axis.xaxis.set_visible(False)
        axis.set_ylim([min(y)-np.std(y), max(y)+np.std(y)])
        axis.set_xlim([min(x), max(x)])
        axis.grid(True)

    # defining a function to plot the data for a given vibration pattern
    def plotVibPattern(self, datapoints):
        #select a subset of samples
        subset = self.dataset[:datapoints]
        # make subplots of x, y, z over timestamp
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(20,10), sharex=True)
        self._plotAxis(ax0, subset['timestamp'], subset['x-axis'], 'X-AXIS')
        self._plotAxis(ax1, subset['timestamp'], subset['y-axis'], 'Y-AXIS')
        self._plotAxis(ax2, subset['timestamp'], subset['z-axis'], 'Z-AXIS')
        # set the size and title
        plt.subplots_adjust(hspace=0.2)
        title = "Accelerometer data for " + self.hertz + "HZ and " + self.volume + " volume"
        fig.suptitle(title)
        plt.subplots_adjust(top=0.9)
        plt.show()


    def data_split_window(self, train_ratio, val_ratio, test_ratio, window_size):
        # Calculate the total ratio sum
        total_ratio = train_ratio + val_ratio + test_ratio
        
        # check ratios sum up to 1.0
        if abs(total_ratio - 1.0) > 1e-6:  # small epsilon for numerical stability
            raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must be equal to 1.0")

        # Split the data
        self.train_data, temp_data = train_test_split(self.dataset, test_size=(1 - train_ratio), random_state=42, shuffle=False)
        remaining_size = val_ratio + test_ratio
        self.val_data, self.test_data = train_test_split(temp_data, test_size=(test_ratio / remaining_size), random_state=42, shuffle=False)

        self.train_data = self.make_windows(self.train_data, window_size)
        self.val_data = self.make_windows(self.val_data, window_size)
        self.test_data = self.make_windows(self.test_data, window_size)

    def _windows(self, data, size):
        start = 0
        while start< data.count():
            yield int(start), int(start + size)
            start += (size/2)

    def make_windows(self, data, window_size):
        segments = np.empty((0,window_size,3))
        for (start, end) in self._windows(data['timestamp'], window_size):
            x = data['x-axis'][start:end]
            y = data['y-axis'][start:end]
            z = data['z-axis'][start:end]

            if(len(data['timestamp'][start:end]) == window_size):
                segments = np.vstack([segments,np.dstack([x, y, z])])
        return segments


class Preprocessing:
    def __init__(self) -> None:
        pass

    def min_max_scale_fit(self, data):
        self.min_vals = np.min(data, axis=0)
        self.max_vals = np.max(data, axis=0)
        scaled_data = (data - self.min_vals) / (self.max_vals - self.min_vals)
        return scaled_data

    def min_max_transform(self, data):
        scaled_data = (data - self.min_vals) / (self.max_vals - self.min_vals)
        return scaled_data