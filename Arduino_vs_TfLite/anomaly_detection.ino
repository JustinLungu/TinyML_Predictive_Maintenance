#include <TensorFlowLite.h>
#include <Arduino_LSM9DS1.h>

//#include <cstdlib>

#include "tensorflow/lite/micro/all_ops_resolver.h" // register TF Lite operations
#include "tensorflow/lite/micro/micro_interpreter.h" // runs the inference engine
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// import model
#include "autoencoder_model.h"

#define DEGUG 0

namespace{
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  int INPUT_LENGTH = 72;


  // Set aside some memory (arena) - used to perform calculations and store the input and output buffers
  // if there are issues allocatingmemory in the code - might be from here, make the number higher
  constexpr int kTensorArenaSize = 60000;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize]; //uint8_t tensor_arena[kTensorArenaSize];;// idk what the differrence here is and what the stuff does so just try both
}


void setup() {
  Serial.begin(9600);
  tflite::InitializeTarget();

  while (!Serial);
  Serial.println("Started");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  model = tflite::GetModel(autoencoder_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  //static tflite::AllOpsResolver resolver;
  static tflite::MicroMutableOpResolver<3> resolver;  // NOLINT
  resolver.AddRelu();
  resolver.AddFullyConnected();
  resolver.AddLogistic();

  
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
}


void loop() {
  // data from 60 HZ 30 vol
  float inputData[INPUT_LENGTH] = {0.5075, 0.4825, 0.675 , 0.475 , 0.51 , 0.8 , 0.5075, 0.4825, 0.6775, 0.475 , 0.51 , 0.8 , 0.5075, 0.4825, 0.6775, 0.4775, 0.51 , 0.8 , 0.5075, 0.4825, 0.68 , 0.4775, 0.51 , 0.7975, 0.505 , 0.4825, 0.6825, 0.4775, 0.51 , 0.7975, 0.505 , 0.4825, 0.6825, 0.4775, 0.51 , 0.7975, 0.505 , 0.4825, 0.685 , 0.4775, 0.51 , 0.7975, 0.505 , 0.4825, 0.685 , 0.48 , 0.51 , 0.7975, 0.505 , 0.4825, 0.6875, 0.48 , 0.5075, 0.795 , 0.505 , 0.4825, 0.6875, 0.48 , 0.5075, 0.795 , 0.505 , 0.4825, 0.69 , 0.48 , 0.5075, 0.795 , 0.505 , 0.485 , 0.6925, 0.48 , 0.5075, 0.795
  };

  float buffer[INPUT_LENGTH] = {0.0};
  float mae = 0.0;
  float diff = 0.0;
  float sumAbsDiff = 0.0;
  int readingsTaken = 0;
/*
  Serial.println("Reading: ");
  // Read readings from accelerometer
  for(int i = 0; i < INPUT_LENGTH; i += 3){
      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(buffer[i], buffer[i+1], buffer[i+2]);
        buffer[i] = (buffer[i] + 2) / 4;
        buffer[i+1] = (buffer[i+1] + 2) / 4;
        buffer[i+2] = (buffer[i+2] + 2) / 4;

      } else {
        i -= 3;
      }
  }

*/
  /*
  int j = 0;
  // Copy data to the input tensor
  for (int i = 0; i < INPUT_LENGTH; ++i) {
      input->data.f[i] = buffer[j]; // Copy the first element of each row
      input->data.f[i + 1] = buffer[j+1]; // Copy the second element of each row
      input->data.f[i + 2] = buffer[j+2]; // Copy the third element of each row
      j += 3;
  }*/

  for (int i = 0; i < INPUT_LENGTH; ++i) {
    input->data.f[i] = float((inputData[i] + 2)/4); 
  }


  // Run inference, and report any error. Call the invoke function and check for errors
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }
  Serial.println("Invoke Successful");

  float output_buffer[INPUT_LENGTH] = {0};

  // Copy data to the input tensor
  for (int i = 0; i < INPUT_LENGTH; ++i) {
      output_buffer[i] = output->data.f[i]; 
  }

  #if DEBUG
    Serial.println("Output for model done");

    // Read 24 readings from model output
    for(int i = 0; i < 96; ++i){
          Serial.print(output_buffer[i]);
          Serial.print("   ");
          Serial.print(output_buffer[i+1]);
          Serial.print("   ");
          Serial.println(output_buffer[i+2]);
    }
  #endif

  //for(int i = 0; i < INPUT_LENGTH; ++i){
  //        Serial.print(output_buffer[i], 4);
  //        Serial.print("   ");
  //}

  // calculate the mean squared error
  sumAbsDiff = 0.0;
  for (int i = 0; i < INPUT_LENGTH; ++i) {
      diff = abs(inputData[i] - output_buffer[i]);
      sumAbsDiff += diff;
  }

  mae = sumAbsDiff / INPUT_LENGTH;

  Serial.print("Mean Absolute Error: ");
  Serial.println(mae, 5);
  Serial.print("Miliseconds: ");
  Serial.println(millis());

  // light up LED depending on anomaly or not
  if(mae > TRESHOLD) {
    // do something
  } else {
    // do something
  }
  

  delay(10000);

}


