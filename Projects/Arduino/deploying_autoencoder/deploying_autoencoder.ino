
#include <TensorFlowLite.h>
#include <Arduino_LSM9DS1.h>

//#include <cstdlib>

#include "tensorflow/lite/micro/all_ops_resolver.h"// register TF Lite operations
#include "tensorflow/lite/micro/micro_interpreter.h"// runs the inference engine
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

//ALREADY HAVE IT: #include "TensorFlowLite.h"
//#include "tensorflow/lite/micro/kernels/micro_ops.h"
//NOT EXISTENT ANYMORE: #include "tensorflow/lite/micro/micro_error_reporter.h"
//ALREADY HAVE IT: #include "tensorflow/lite/micro/micro_interpreter.h"
//#include "tensorflow/lite/micro/micro_mutable_op_resolver.h" //theoretically same thing as the all_ops_resolver.h
//NOT EXISTENT ANYMORE: #include "tensorflow/lite/version.h"

// import model
#include "autoencoder_model.h"

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

  //NOT EXISTENT ANYMORE: static tflite::MicroErrorReporter micro_error_reporter;
  //NOT EXISTENT ANYMORE: error_reporter = &micro_error_reporter;

  model = tflite::GetModel(autoencoder_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  /////////////////////////////////// TO CHECK OUT AND COMPARE ////////////////////////////////////////
  // Pull in only needed operations (should match NN layers)
  // Available ops:
  //  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/micro_ops.h
  /*
  static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddBuiltin(
    tflite::BuiltinOperator_FULLY_CONNECTED,
    tflite::ops::micro::Register_FULLY_CONNECTED(),
    1, 3);
  */
  /////////////////////////////////////////////////////////////////////////////////////////////////////

  //static tflite::AllOpsResolver resolver;
  static tflite::MicroMutableOpResolver<4> resolver;  // NOLINT
  resolver.AddRelu();
  resolver.AddFullyConnected();
  resolver.AddLogistic();
  resolver.AddTanh();
  
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
  //float inputData[INPUT_LENGTH] = {0.5075, 0.4825, 0.675};

  // used to read data from the accelerometer as input for the model  
  float inputBuffer[INPUT_LENGTH] = {0};
  float buffer[INPUT_LENGTH] = {0};
  int readingsTaken = 0;

  // // Read 24 readings from accelerometer
  // for(int i = 0; i < INPUT_LENGTH; i += 3){
  //     if (IMU.accelerationAvailable()) {
  //       IMU.readAcceleration(inputBuffer[i], inputBuffer[i + 1], inputBuffer[i + 2]);
  //       Serial.println("set");
  //     } else {
  //       i -= 3;
  //     }
  // }
  // Serial.println("Reading Window Done");




  // Read 24 readings from accelerometer
  for(int i = 0; i < 72; i += 3){
      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(buffer[i], buffer[i+1], buffer[i+2]);
        // Serial.print(buffer[i]);
        // Serial.print("   ");
        // Serial.print(buffer[i+1]);
        // Serial.print("   ");
        // Serial.println(buffer[i+2]);
        readingsTaken++; // Increment the number of readings taken
      } else {
        i -= 3;
      }
  }


  // Serial.println("THE INPUT WAS");

  // for(int i = 0; i < INPUT_LENGTH; ++i){
  //         Serial.print(buffer[i], 8);
  //         Serial.print("   ");
  // }
  // Serial.println(" ");

  // float buffer[INPUT_LENGTH] = {0.0};
  float mae = 0.0;
  float diff = 0.0;
  float sumAbsDiff = 0.0;


  
  for (int i = 0; i < INPUT_LENGTH; ++i) {
    input->data.f[i] = (buffer[i] + 2)/4; //inputData[i]; 
  }

  // Serial.println("THE INPUT Normalized WAS");

  // for(int i = 0; i < INPUT_LENGTH; ++i){
  //         Serial.print(input->data.f[i], 8);
  //         Serial.print("   ");
  // }
  // Serial.println(" ");


  // Run inference, and report any error. Call the invoke function and check for errors
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }
  //Serial.println("Invoke Successful");

  float output_buffer[INPUT_LENGTH] = {0};

  // Copy data to the input tensor
  for (int i = 0; i < INPUT_LENGTH; ++i) {
      output_buffer[i] = output->data.f[i]; 
  }

  // Serial.println("THE INPUT WAS");

  // for(int i = 0; i < INPUT_LENGTH; ++i){
  //         Serial.print(inputBuffer[i], 8);
  //         Serial.print("   ");
  // }
  // Serial.println(" ");

  // Serial.println("THE INPUT NORMALIZED WAS");

  // for(int i = 0; i < INPUT_LENGTH; ++i){
  //         Serial.print((inputBuffer[i] + 2)/4, 8);
  //         Serial.print("   ");
  // }
  // Serial.println(" ");


  // Serial.println("THE OUTPUT WAS");
  // for(int i = 0; i < INPUT_LENGTH; ++i){
  //         Serial.print(output_buffer[i], 8);
  //         Serial.print("   ");
  // }

  // calculate the mean absolute error
  sumAbsDiff = 0.0;
  for (int i = 0; i < INPUT_LENGTH; ++i) {
      diff = abs(((buffer[i]+2)/4) - output_buffer[i]);
      sumAbsDiff += diff;
  }

  mae = sumAbsDiff / INPUT_LENGTH;
  Serial.println(" ");
  Serial.print("Mean Absolute Error: ");
  Serial.println(mae, 5);

  if(mae > TRESHOLD_MIN && mae < TRESHOLD_MAX) {
    //normal data, turn blue on
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDB, LOW);

  } else {
    // annomaly, turn red on
    digitalWrite(LEDR, LOW);
    digitalWrite(LEDB, HIGH);
  }

  Serial.print("Miliseconds: ");
  Serial.println(millis());
  

  delay(2000);

}
