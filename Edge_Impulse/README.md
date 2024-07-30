# How to use Edge Impulse with Windows and/or Linux


## Edge Impulse CLI installation:

official documentation: https://docs.edgeimpulse.com/docs/tools/edge-impulse-cli/cli-installation

Overview of the steps:
1. Create an account on edge impulse
2. have python3 installed on your machine
3. have node.js version 20 or above installed on your machine
4. install the CLI tools

### Steps for Windows:



### Steps for Linux:







Useful links for Edge Impulse - board setup:
- https://docs.edgeimpulse.com/docs/edge-ai-hardware/mcu/arduino-nano-33-ble-sense
- https://docs.edgeimpulse.com/docs/edge-impulse-studio/deployment
- might help with deployment issue: https://forum.edgeimpulse.com/t/arduino-deployment-not-visible/4614/2

Re-connecting to an already existing project:
  In order to re-connect with the board via Edge Impulse website, run this command in terminal: **edge-impulse-daemon** and you're all set.

Creating a new project with an already connected board:
  The only difference is that now, once you've created the project no the website, you then run the following command in the terminal: **edge-impulse-daemon --clean**. Then, you enter your username/e-mail and password and again you're all set. You should see that the board is connected in the website interface.
