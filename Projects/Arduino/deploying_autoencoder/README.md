With this sketch you can deploy the (non-qunatized) autoencoder on the Arduino.

The model (binary representation) is in the .h file.

One dummy reading from the accelerometer is hardcoded and used to run one inference. The output should be the reconstructed 3 numbers, the same (or a very close) value to the original input.

TO DO: 

       - see if using one reading as input is good enough, or if we should use a window of data (like in the CNN approach) which the autoencoder re-constructs.

       - see if we can make the quantized version of the model work 
       
       - measure inference speed
       

TO REVIEW: (same as CNN) How can we see how much RAM we need? What do we need to consider going forward?
            We need to (?) compare the original and reconstructed values in order to see if the vibration is normal or anomalous (too much difference means anomaly). What classifies as too much difference? How do you measure that? Over how many readings should one "classification" be done? 
           
