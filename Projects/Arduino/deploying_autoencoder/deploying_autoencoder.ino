#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h" // register TF Lite operations
#include "tensorflow/lite/micro/micro_interpreter.h" // runs the inference engine
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// import model
#include "autoencoder_model.h"


// debug flag to print metrics from the inference engine - something abotu errors and debugging
#define DEBUG 0


// keep pointers and variables unique to the file, we define pointers to use later
namespace{
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Set aside some memory (arena) - used to perform calculations and store the input and output buffers
  // if there are issues allocatingmemory in the code - might be from here, make the number higher
  constexpr int kTensorArenaSize = 20000; // <--------TODO: change this and see what happens
  alignas(16) uint8_t tensor_arena[kTensorArenaSize]; //uint8_t tensor_arena[kTensorArenaSize];;// idk what the differrence here is and what the stuff does so just try both
}


//1. Load the model
//2. Allocate necessary buffers - formatted as tensors
//3. Run ingference on some dummy data


void setup() {
  Serial.begin(9600);
  tflite::InitializeTarget();

  // Map the model into a usable data structure. TFLite reads in the model from the .h file. "model" will be the handle for the model.
  model = tflite::GetModel(autoencoder_model);
  // check that the flat buffer version in the model matches what TFLite can handle. This can be done with the micro_error_reporter thingy that does not work :))
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Load operations. See available operations in micro_ops.h. I need: FULLY_CONNECTED, CONV2D,  - idk look at the model's layers
  // Load all ops and yolo
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with. This will run inference for us.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  // Set asidememory for the arena. Give an error if this fails, because the arena is not big enough.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Create handles for the input and output buffers. See supported data types in common.h
  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

//------------------------------------------------------------------- Print the dimensions of the input and output to see that they match what you are expecting to have - 24x3 for CNN and 3 for autoencoder
/*
#if DEBUG
//  Serial.print("am I printing? nr of dims        ");
//  Serial.println(output->dims->size);
//  Serial.print("Output Dim 1 size: ");
//  Serial.println(output->dims->data[0]);
//  Serial.print("Dim 2 size: ");
//  Serial.println(output->dims->data[1]);
//  Serial.print("Type: ");
//  Serial.println(output->type);
//  Serial.print("Input Dim 2 size: ");
//  Serial.println(input->dims->size);
#endif
*/
}







void loop() {
// ------------------------------> TODO: measure how long inference takes
#if DEBUG
  unsigned long start_timestamp = micros();
#endif

  // Dummy data to run inference on. Adapt based on the model you are testing
  float inputData[3] = {0.800000, 0.2, 0.166667};


  // TODO: QUANTIZE INPUT AND DEQUANTIZE OUTPUT? Do we need that?
  // something like that ???   int8_t x_quantized = x / input->params.scale + input->params.zero_point;   input->data.int8[0] = x_quantized;


  // we need to copy the inputData values to the input tensor, input->data[x][y]
  for (int i = 0; i < 3; i++) {
    input->data.f[i] = inputData[i];
  }

/*
  // Copy data to the input tensor
  for (int i = 0; i < 24; ++i) {
      input->data.int8[i] = inputData[i][0];               /// input->params.scale + input->params.zero_point; 
      input->data.int8[i + 24] = inputData[i][1];          /// input->params.scale + input->params.zero_point; 
      input->data.int8[i + 48] = inputData[i][2];          /// input->params.scale + input->params.zero_point;
  }
*/

/*
  // If you print the input tensor values, you should see the values from your inputData array.
  #if DEBUG
  Serial.println("Input Data:");
  for (int i = 0; i < 24; ++i) {
      Serial.print(input->data.f[i]);
      Serial.print(", ");
      Serial.print(input->data.f[i + 24]);
      Serial.print(", ");
      Serial.println(input->data.f[i + 48]);
  }
  #endif
*/


  // Run inference, and report any error. Call the invoke function and check for errors
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  // The invoke function is a blocking function, so the output value will be in the tensor buffer associated with the output handle ("output")
  // Obtain the quantized output from model's output tensor /  read it from there
  Serial.println(output->dims->data[1]);
  for (int i = 0; i < output->dims->data[1]; ++i) {
    Serial.println(static_cast<float>(output->data.f[i]));
  }

/*
  for (int i = 0; i < output->dims->data[1]; ++i) {
    Serial.println(static_cast<float>(output->data.f[i]));
  }
  // Do we need to dequantize the output? I just saw it being done in some other examples.
  // --------------------------------------------> TODO: for the final label chose the index with the highest probability
*/

/*
#if DEBUG
  Serial.print("Time for inference: ");
  Serial.println(micros() - start_timestamp);  
#endif
*/
  delay(1000);

}
