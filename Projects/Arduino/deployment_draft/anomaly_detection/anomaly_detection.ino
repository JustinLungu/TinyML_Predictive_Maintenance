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
  

  
  // Read 24 readings from accelerometer
  for(int i = 0; i < 72; i += 3){
      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(buffer[i], buffer[i+1], buffer[i+2]);
      } else {
        i--;
      }
  }

  Serial.println("Reading Window Done");
  
  int j = 0;
  // Copy data to the input tensor
  for (int i = 0; i < 24; ++i) {
      input->data.f[i] = buffer[j]; // Copy the first element of each row
      input->data.f[i + 24] = buffer[j+1]; // Copy the second element of each row
      input->data.f[i + 48] = buffer[j+2]; // Copy the third element of each row
      j += 3;
  }
  
  // Run inference, and report any error. Call the invoke function and check for errors
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  float output_buffer[72] = {0};

  // Copy data to the input tensor
  for (int i = 0; i < 24; ++i) {
      output_buffer[j] = output->data.f[i]; // Copy the first element of each row
      output_buffer[j+1] = output->data.f[i + 24]; // Copy the second element of each row
      output_buffer[j+2] = output->data.f[i + 48]; // Copy the third element of each row
      j += 3;
  }


  // calculate the mean squared error
  sumSquaredDiff = 0.0;
  for (int i = 0; i < 72; ++i) {
      diff = buffer[i] - output_buffer[i];
      sumSquaredDiff += diff * diff;
  }
  mse = sumSquaredDiff / 72;

  Serial.println(mse);
  Serial.println("Miliseconds:");
  Serial.println(millis());
  

  delay(1000);

}


