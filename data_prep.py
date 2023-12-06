#--------------read the data from a file-------------------------
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class Data:

    _dataset = []

    def __init__(self, capture, hertz, volume) -> None:
        self._dataset = []
        self.capture = capture
        self.hertz = hertz
        self.volume = volume

    ########################## LOAD DATA ######################################################

    # defining function for loading the dataset
    def _readData(self, filePath:str) -> None:
        # attributes of the dataset
        columnNames = ['x-axis','y-axis','z-axis']
        #read the specified file using pandas function and return the data
        data = pd.read_csv(filePath, header=None, names=columnNames, sep='\t', na_values=';')
        return data

    def loadData(self) -> None:
        folder_path = "Collected_data"  # Specify the folder path here
        #Load data
        filename = f"capture{self.capture}_{self.hertz}hz_{self.volume}vol.txt"
        file_path = os.path.join(folder_path, filename)
        self._dataset = self._readData(file_path)
        # Add a new column for sequential timestamps
        self._dataset['timestamp'] = np.arange(1, len(self._dataset) + 1)

    def getData(self):
        return self._dataset

    ######################## DATA VISUALIZATION ###############################################
    # defining the function to plot a single axis data
    # setup color, title, limit and add grid.
    def _plotAxis(self, axis, x, y, title:str) -> None:
        axis.plot(x, y, color='green', linewidth=1)
        axis.set_title(title)
        axis.xaxis.set_visible(False)
        axis.set_ylim([min(y)-np.std(y), max(y)+np.std(y)])
        axis.set_xlim([min(x), max(x)])
        axis.grid(True)

    # defining a function to plot the data for a given vibration pattern
    def plotVibPattern(self, readings:int) -> None:
        data = self._dataset[:readings]
        # make subplots of x, y, z over timestamp
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(20,10), sharex=True)
        self._plotAxis(ax0, data['timestamp'], data['x-axis'], 'X-AXIS')
        self._plotAxis(ax1, data['timestamp'], data['y-axis'], 'Y-AXIS')
        self._plotAxis(ax2, data['timestamp'], data['z-axis'], 'Z-AXIS')
        # set the size and title
        plt.subplots_adjust(hspace=0.2)
        title = "Accelerometer data for " + self.hertz + "HZ and " + self.volume + " volume"
        fig.suptitle(title)
        plt.subplots_adjust(top=0.9)
        plt.show()


