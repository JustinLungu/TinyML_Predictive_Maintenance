# TinyML_Predictive_Maintenance
This repository will contain all the files and descriptions used in our project to deploy a tiny machine learning on Arduino Nano 33 BLE Sense Lite regarding predictive maintenance.



The project we are conducting, namely processing the sensor data on-board the robots using embedded AI/ML models, has advantages over traditional signal processing that can be done on-board the robots and also over typical AI/ML processing of the data in the cloud. The most important advantage of employing embedded AI/ML over traditional embedded signal processing is shifting towards an inspection paradigm that is data-driven rather than model-based, and is thus applicable to a wide variety of infrastructures where prior knowledge of the system may not be readily available.  The most important advantage of employing embedded AI/ML over traditional processing of data using AI/ML models in the cloud is reducing the amount of data that has to be transmitted over a network to reach the cloud server, thus preserving the power of the embedded devices and improving the security of the oftentimes critical system data.

We will approach this task by building and training two models. One will be a convolutional neural network, which specializes in classifying different types of vibrations, while the other is an autoencoder, which focuses on reconstructing learned signals (both of which can be found in the Project folder). We will use both to attempt to detect anomalies in accelerometer data. 


Abaqus Ubuntu installation guide: https://github.com/franaudo/abaqus-ubuntu#run-setup
