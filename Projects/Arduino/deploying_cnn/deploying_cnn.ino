
#include <TensorFlowLite.h>
#include <Arduino_LSM9DS1.h>

//#include <cstdlib>

#include "tensorflow/lite/micro/all_ops_resolver.h"// register TF Lite operations
#include "tensorflow/lite/micro/micro_interpreter.h"// runs the inference engine
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


// import model
#include "cnn_model.h"

#define DEGUG 0

namespace{
  //NOT EXISTENT ANYMORE: tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  int INPUT_LENGTH = 72;
  float TRESHOLD_MAX = 0.018;  // Treshold calculated by looking at MAE values from the train and test set and their mean and std
  float TRESHOLD_MIN = 0.012; 

  // Set aside some memory (arena) - used to perform calculations and store the input and output buffers
  // if there are issues allocatingmemory in the code - might be from here, make the number higher
  constexpr int kTensorArenaSize = 10 * 1024;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize]; 
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
  static tflite::MicroMutableOpResolver<6> resolver;  // NOLINT
  resolver.AddRelu();
  resolver.AddFullyConnected();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddSoftmax();
  resolver.AddReshape();

  
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  pinMode(LEDR, OUTPUT); //LEDR
  pinMode(LEDR, OUTPUT); // LEDR

  input = interpreter->input(0);
  output = interpreter->output(0);
}


void loop() {
  //3x1 input

  // used to read data from the accelerometer as input for the model  
  float inputBuffer[1][24][3][1] = {0};
  float buffer[24][3] = {0};
  int readingsTaken = 0;

  // Serial.println(input->dims->data[0]);
  // Serial.println(input->dims->data[1]);
  // Serial.println(input->dims->data[2]);
  // Serial.println(input->dims->data[3]);


  // Read 24 readings from accelerometer
  for(int i = 0; i < 24; i += 1){
      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(buffer[i][0], buffer[i][1], buffer[i][2]);

        readingsTaken++; // Increment the number of readings taken
      } else {
        i -= 1;
      }
  }


  // Serial.println("THE INPUT WAS");

  // for(int i = 0; i < 24; ++i){
  //         Serial.print(buffer[i][0]);
  //         Serial.print("   ");
  //         Serial.print(buffer[i][1]);
  //         Serial.print("   ");
  //         Serial.print(buffer[i][2]);

  // }
  // Serial.println(" ");

  // float buffer[INPUT_LENGTH] = {0.0};
  float mae = 0.0;
  float diff = 0.0;
  float sumAbsDiff = 0.0;


  
  for (int i = 0; i < 24; ++i) {
    input->data.f[i] = (buffer[i][0] + 2)/4; //inputData[i];
    input->data.f[i+24] = (buffer[i][2] + 2)/4; //inputData[i];
    input->data.f[i+48] = (buffer[i][2] + 2)/4; //inputData[i]; 
  }

  // // Serial.println("THE INPUT Normalized WAS");

  // // for(int i = 0; i < INPUT_LENGTH; ++i){
  // //         Serial.print(input->data.f[i], 8);
  // //         Serial.print("   ");
  // // }
  // // Serial.println(" ");


  // Run inference, and report any error. Call the invoke function and check for errors
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  float output_buffer[3] = {0};
  float max_confidence = -1;
  int label = -1;

  // Copy data to the input tensor
  for (int i = 0; i < 3; ++i) {
      output_buffer[i] = output->data.f[i]; 
      if(output_buffer[i] > max_confidence){
        label = i;
        max_confidence = output_buffer[i];
      }
  }
  Serial.println(label+1);


  delay(2000);

}
