/*
  Arduino LSM9DS1 - Simple Accelerometer

  This example reads the acceleration values from the LSM9DS1
  sensor and continuously prints them to the Serial Monitor
  or Serial Plotter.

  The circuit:
  - Arduino Nano 33 BLE Sense

  created 10 Jul 2019
  by Riccardo Rizzo

  This example code is in the public domain.
*/

#include <Arduino_LSM9DS1.h>

void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Started");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.println();
  Serial.println("Acceleration in g's");
  Serial.println("X\tY\tZ");
}

void loop() {
  
  float buffer[24][3] = {0};
  float mse = 0.0;
  float diff = 0.0;
  float sumSquaredDiff = 0.0;

  int readingsTaken = 0; // Variable to keep track of the number of readings taken
  
  // Read 24 readings from accelerometer
  for(int i = 0; i < 24; ++i){
      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(buffer[i][0], buffer[i][1], buffer[i][2]);
        Serial.print("Reading ");
        Serial.print(i);
        Serial.print("   ");
        Serial.print(buffer[i][0]);
        Serial.print("   ");
        Serial.print(buffer[i][1]);
        Serial.print("   ");
        Serial.println(buffer[i][2]);
        readingsTaken++; // Increment the number of readings taken
      } else {
        i--;
        //Serial.println("Acceleration not available!");
      }
  }
  
  // Print the number of readings taken
  Serial.print("Readings taken: ");
  Serial.println(readingsTaken);
  
  Serial.println("Reading Window Done");

}
