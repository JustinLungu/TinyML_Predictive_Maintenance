#include <TensorFlowLite.h>
#include <Arduino_LSM9DS1.h>


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

  // Set aside some memory (arena) - used to perform calculations and store the input and output buffers
  // if there are issues allocatingmemory in the code - might be from here, make the number higher
  constexpr int kTensorArenaSize = 20000;
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

  static tflite::AllOpsResolver resolver;

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

  float inputData[288] = {0.5075, 0.4825, 0.675 , 0.475 , 0.51  , 0.8   , 0.5075, 0.4825,
       0.6775, 0.475 , 0.51  , 0.8   , 0.5075, 0.4825, 0.6775, 0.4775,
       0.51  , 0.8   , 0.5075, 0.4825, 0.68  , 0.4775, 0.51  , 0.7975,
       0.505 , 0.4825, 0.6825, 0.4775, 0.51  , 0.7975, 0.505 , 0.4825,
       0.6825, 0.4775, 0.51  , 0.7975, 0.505 , 0.4825, 0.685 , 0.4775,
       0.51  , 0.7975, 0.505 , 0.4825, 0.685 , 0.48  , 0.51  , 0.7975,
       0.505 , 0.4825, 0.6875, 0.48  , 0.5075, 0.795 , 0.505 , 0.4825,
       0.6875, 0.48  , 0.5075, 0.795 , 0.505 , 0.4825, 0.69  , 0.48  ,
       0.5075, 0.795 , 0.505 , 0.485 , 0.6925, 0.48  , 0.5075, 0.795 ,
       0.505 , 0.485 , 0.6925, 0.4825, 0.5075, 0.7925, 0.505 , 0.485 ,
       0.695 , 0.4825, 0.5075, 0.7925, 0.505 , 0.485 , 0.695 , 0.4825,
       0.5075, 0.7925, 0.505 , 0.485 , 0.6975, 0.4825, 0.5075, 0.7925,
       0.505 , 0.485 , 0.7   , 0.485 , 0.5075, 0.7925, 0.505 , 0.485 ,
       0.7   , 0.485 , 0.5075, 0.7925, 0.505 , 0.485 , 0.7025, 0.485 ,
       0.5075, 0.79  , 0.505 , 0.485 , 0.705 , 0.485 , 0.5075, 0.79  ,
       0.505 , 0.485 , 0.705 , 0.4875, 0.5075, 0.79  , 0.505 , 0.485 ,
       0.7075, 0.4875, 0.505 , 0.79  , 0.505 , 0.485 , 0.7075, 0.4875,
       0.505 , 0.7875, 0.505 , 0.4875, 0.71  , 0.4875, 0.505 , 0.7875,
       0.505 , 0.4875, 0.7125, 0.4875, 0.505 , 0.7875, 0.5025, 0.4875,
       0.7125, 0.49  , 0.505 , 0.7875, 0.505 , 0.4875, 0.715 , 0.49  ,
       0.505 , 0.7875, 0.5025, 0.4875, 0.7175, 0.49  , 0.505 , 0.785 ,
       0.5025, 0.4875, 0.7175, 0.49  , 0.505 , 0.785 , 0.5025, 0.4875,
       0.72  , 0.4925, 0.505 , 0.785 , 0.5025, 0.4875, 0.7225, 0.4925,
       0.505 , 0.785 , 0.5025, 0.4875, 0.7225, 0.4925, 0.505 , 0.7825,
       0.5025, 0.4875, 0.725 , 0.4925, 0.505 , 0.7825, 0.5025, 0.4875,
       0.725 , 0.495 , 0.505 , 0.7825, 0.5025, 0.4875, 0.7275, 0.495 ,
       0.505 , 0.7825, 0.5025, 0.49  , 0.73  , 0.495 , 0.505 , 0.78  ,
       0.5025, 0.49  , 0.73  , 0.495 , 0.505 , 0.78  , 0.5025, 0.49  ,
       0.7325, 0.495 , 0.5025, 0.78  , 0.5025, 0.49  , 0.735 , 0.4975,
       0.5025, 0.78  , 0.5025, 0.49  , 0.735 , 0.4975, 0.5025, 0.78  ,
       0.5025, 0.49  , 0.7375, 0.4975, 0.5025, 0.7775, 0.5025, 0.49  ,
       0.7375, 0.4975, 0.5025, 0.7775, 0.5025, 0.49  , 0.74  , 0.5   ,
       0.5025, 0.7775, 0.5025, 0.4925, 0.7425, 0.5   , 0.5025, 0.7775,
       0.5025, 0.49  , 0.745 , 0.5   , 0.5025, 0.775 , 0.5025, 0.4925,
       0.745 , 0.5025, 0.5025, 0.775 , 0.5025, 0.4925, 0.7475, 0.5025,
       0.5025, 0.775 , 0.5025, 0.4925, 0.7475, 0.5025, 0.5025, 0.7725};


  float buffer[288] = {0.0};
  float mae = 0.0;
  float diff = 0.0;
  float sumAbsDiff = 0.0;
  int readingsTaken = 0;

  Serial.println("Reading: ");
  // Read 24 readings from accelerometer
  for(int i = 0; i < 288; i += 3){
      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(buffer[i], buffer[i+1], buffer[i+2]);
        buffer[i] = (buffer[i] + 2) / 4;
        buffer[i+1] = (buffer[i+1] + 2) / 4;
        buffer[i+2] = (buffer[i+2] + 2) / 4;
        #if DEBUG
          Serial.print(buffer[i]);
          Serial.print("   ");
          Serial.print(buffer[i+1]);
          Serial.print("   ");
          Serial.println(buffer[i+2]);
          readingsTaken++; // Increment the number of readings taken
        #endif
      } else {
        i -= 3;
      }
  }

  #if DEBUG
    // Print the number of readings taken
    Serial.print("Readings taken: ");
    Serial.println(readingsTaken);
    Serial.println("Reading Window Done");
  #endif

  /*
  int j = 0;
  // Copy data to the input tensor
  for (int i = 0; i < 288; ++i) {
      input->data.f[i] = buffer[j]; // Copy the first element of each row
      input->data.f[i + 1] = buffer[j+1]; // Copy the second element of each row
      input->data.f[i + 2] = buffer[j+2]; // Copy the third element of each row
      j += 3;
  }*/

  for (int i = 0; i < 288; ++i) {
    input->data.f[i] = float((inputData[i] + 2)/4); 
  }

  #if DEBUG
    Serial.println("Input for model done");
  #endif


  // Run inference, and report any error. Call the invoke function and check for errors
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }
  Serial.println("Invoke Successful");

  float output_buffer[288] = {0};

  // Copy data to the input tensor
  for (int i = 0; i < 288; ++i) {
      output_buffer[i] = output->data.f[i]; // Copy the first element of each row
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

  // calculate the mean squared error
  sumAbsDiff = 0.0;
  for (int i = 0; i < 288; ++i) {
      diff = abs(buffer[i] - output_buffer[i]);
      sumAbsDiff += diff;
  }
  mae = sumAbsDiff / 288;

  Serial.print("Mean Absolute Error: ");
  Serial.println(mae);
  Serial.print("Miliseconds: ");
  Serial.println(millis());
  

  delay(1000);

}


