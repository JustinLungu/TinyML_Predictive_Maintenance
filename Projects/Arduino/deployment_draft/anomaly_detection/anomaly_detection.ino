#include <TensorFlowLite.h>
#include <Arduino_LSM9DS1.h>


#include "tensorflow/lite/micro/all_ops_resolver.h" // register TF Lite operations
#include "tensorflow/lite/micro/micro_interpreter.h" // runs the inference engine
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// import model
#include "autoencoder_model.h"


namespace{
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Set aside some memory (arena) - used to perform calculations and store the input and output buffers
  // if there are issues allocatingmemory in the code - might be from here, make the number higher
  constexpr int kTensorArenaSize = 20000;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize]; //uint8_t tensor_arena[kTensorArenaSize];;// idk what the differrence here is and what the stuff does so just try both

  // Min-Max scaling parameters
    float min_vals[3]; // Assuming 3 axes for accelerometer data
    float max_vals[3];
}


void min_max_scale_fit(const float* data, int size) {

    for(int i = 0; i < 3; ++i){
      min_vals[i] = {99};
      max_vals[i] = {0};
    }

    for (int i = 0; i < size; i += 3) {
      min_vals[0] = min(min_vals[0], data[i]);
      max_vals[0] = max(max_vals[0], data[i]);

      min_vals[1] = min(min_vals[1], data[i+1]);
      max_vals[1] = max(max_vals[1], data[i+1]);

      min_vals[2] = min(min_vals[2], data[i+2]);
      max_vals[2] = max(max_vals[2], data[i+2]);
    }
}

void min_max_transform(float* data, int size) {
    for (int i = 0; i < size; i += 3) {
      data[i] = (data[i] - min_vals[0]) / (max_vals[0] - min_vals[0]);

      data[i+1] = (data[i+1] - min_vals[1]) / (max_vals[1] - min_vals[1]);

      data[i+2] = (data[i+2] - min_vals[2]) / (max_vals[2] - min_vals[2]);
    }
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

  float buffer[72] = {0};
  float mse = 0.0;
  float diff = 0.0;
  float sumSquaredDiff = 0.0;
  int readingsTaken = 0;

  Serial.println("Reading: ");
  // Read 24 readings from accelerometer
  for(int i = 0; i < 72; i += 3){
      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(buffer[i], buffer[i+1], buffer[i+2]);
        Serial.print(buffer[i]);
        Serial.print("   ");
        Serial.print(buffer[i+1]);
        Serial.print("   ");
        Serial.println(buffer[i+2]);
        readingsTaken++; // Increment the number of readings taken
      } else {
        i -= 3;
      }
  }

  // Print the number of readings taken
  Serial.print("Readings taken: ");
  Serial.println(readingsTaken);

  Serial.println("Reading Window Done");


  // Normalize data using Min-Max scaling
  min_max_scale_fit(buffer, 72);
  min_max_transform(buffer, 72);

  Serial.println("Printing normalized data:");
  //print normalized data
  for(int i = 0; i < 72; i += 3){
      Serial.print(buffer[i]);
      Serial.print("   ");
      Serial.print(buffer[i+1]);
      Serial.print("   ");
      Serial.println(buffer[i+2]);
  }
  

  
  int j = 0;
  // Copy data to the input tensor
  for (int i = 0; i < 24; ++i) {
      input->data.f[i] = buffer[j]; // Copy the first element of each row
      input->data.f[i + 24] = buffer[j+1]; // Copy the second element of each row
      input->data.f[i + 48] = buffer[j+2]; // Copy the third element of each row
      j += 3;
  }

  Serial.println("Input for model done");
  
  // Run inference, and report any error. Call the invoke function and check for errors
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }
  Serial.println("Invoke Successful");

  float output_buffer[72] = {0};

  j = 0;
  // Copy data to the input tensor
  for (int i = 0; i < 24; ++i) {
      output_buffer[j] = output->data.f[i]; // Copy the first element of each row
      output_buffer[j+1] = output->data.f[i + 24]; // Copy the second element of each row
      output_buffer[j+2] = output->data.f[i + 48]; // Copy the third element of each row
      j += 3;
  }

  Serial.println("Output for model done");

  // Read 24 readings from model output
  for(int i = 0; i < 24; ++i){
        Serial.print(output_buffer[i]);
        Serial.print("   ");
        Serial.print(output_buffer[i+1]);
        Serial.print("   ");
        Serial.println(output_buffer[i+2]);
  }

  // calculate the mean squared error
  sumSquaredDiff = 0.0;
  for (int i = 0; i < 72; ++i) {
      diff = buffer[i] - output_buffer[i];
      sumSquaredDiff += diff * diff;
  }
  mse = sumSquaredDiff / 72;

  Serial.print("Mean Squared Error: ");
  Serial.println(mse);
  Serial.print("Miliseconds: ");
  Serial.println(millis());
  

  delay(1000);

}

