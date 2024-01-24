With this sketch you can deploy the (non-qunatized) CNN on the Arduino.

The model (binary representation) is in the .h file. 

A slice of (already normalized) data is hardcoded and usedto run one inference. The output is the label with the highest probability.

From the testing we have done so far, it seems to work correctly.

TO DO:  - add data normalization steps
        - see if we can make the quantized version of the model work
        - measure inference speed

TO REVIEW:  How can we see how much RAM we need? What do we need to consider going forward?
