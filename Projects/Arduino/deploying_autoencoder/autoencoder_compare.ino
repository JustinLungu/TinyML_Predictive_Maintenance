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
  float TRESHOLD = 0.004;  // Treshold calculated by looking at MAE values from the train and test set and their mean and std


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
  
  // data from 60 HZ 30 vol, already normalized
  float inputData[INPUT_LENGTH] = {0.5075, 0.4825, 0.675 , 0.475 , 0.51 , 0.8 , 0.5075, 0.4825, 0.6775, 0.475 , 0.51 , 0.8 , 0.5075, 0.4825, 0.6775, 0.4775, 0.51 , 0.8 , 0.5075, 0.4825, 0.68 , 0.4775, 0.51 , 0.7975, 0.505 , 0.4825, 0.6825, 0.4775, 0.51 , 0.7975, 0.505 , 0.4825, 0.6825, 0.4775, 0.51 , 0.7975, 0.505 , 0.4825, 0.685 , 0.4775, 0.51 , 0.7975, 0.505 , 0.4825, 0.685 , 0.48 , 0.51 , 0.7975, 0.505 , 0.4825, 0.6875, 0.48 , 0.5075, 0.795 , 0.505 , 0.4825, 0.6875, 0.48 , 0.5075, 0.795 , 0.505 , 0.4825, 0.69 , 0.48 , 0.5075, 0.795 , 0.505 , 0.485 , 0.6925, 0.48 , 0.5075, 0.795
  };

  //3x1 input
  //float inputData[INPUT_LENGTH] = {0.5075, 0.4825, 0.675};

  // used to read data from the accelerometer as input for the model  
  float inputBuffer[INPUT_LENGTH] = {0};

  // Read 24 readings from accelerometer
  for(int i = 0; i < INPUT_LENGTH; i += 3){
      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(inputBuffer[i], inputBuffer[i + 1], inputBuffer[i + 2]);
      }
  }
  Serial.println("Reading Window Done");


  float buffer[INPUT_LENGTH] = {0.0};
  float mae = 0.0;
  float diff = 0.0;
  float sumAbsDiff = 0.0;
  int readingsTaken = 0;

  
  for (int i = 0; i < INPUT_LENGTH; ++i) {
    input->data.f[i] = (inputBuffer[i] + 2)/4; //inputData[i]; 
  }

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

  Serial.println("THE INPUT WAS");

  for(int i = 0; i < INPUT_LENGTH; ++i){
          Serial.print((inputBuffer[i] + 2)/4, 8);
          Serial.print("   ");
  }
  Serial.println(" ");

  for(int i = 0; i < INPUT_LENGTH; ++i){
          Serial.print(output_buffer[i], 8);
          Serial.print("   ");
  }

  // calculate the mean absolute error
  sumAbsDiff = 0.0;
  for (int i = 0; i < INPUT_LENGTH; ++i) {
      diff = abs(inputData[i] - output_buffer[i]);
      sumAbsDiff += diff;
  }

  mae = sumAbsDiff / INPUT_LENGTH;
  Serial.println(" ");
  Serial.print("Mean Absolute Error: ");
  Serial.println(mae, 5);

  if(mae > TRESHOLD) {
    // annomaly, turn red on
    digitalWrite(LEDR, LOW);
    digitalWrite(LEDB, HIGH);

  } else {
    //normal data, turn blue on
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDB, LOW);
  }

  Serial.print("Miliseconds: ");
  Serial.println(millis());
  

  delay(10000);

}
