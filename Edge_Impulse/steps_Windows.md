Following:
https://harvard-edge.github.io/cs249r_book/contents/labs/arduino/nicla_vision/motion_classification/motion_classification.html
(using the CLI Data Forwarder tool)

# Installation - Windows
1. Create an Edge Impulse account.
2. Install Python 3 on your host computer.
3. Install Node.js v20 or above on your host computer.
For Windows users, install the Additional Node.js tools (called Tools for Native Modules on newer versions) when prompted.
4. Install the CLI tools via: (Note: in the Node.js command prompt!!)
      npm install -g edge-impulse-cli --force
Tools will be available on the path you did all this on

# Connect board to a project:
edge-impulse-data-forwarder --clean
- user-name and password
- select port to which board is connected
- selecct project to which you connect
* use without --clean to just connect the same as the last time you used it
