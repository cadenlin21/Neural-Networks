# ABCD Neural Network From Scratch

This project encodes a fully functional, feed-forward, backpropagation neural network coded from scratch in Java. The network has one input layer, two hidden layers, and one output layer. Weights are trained through gradient descent. The program has been and can be trained to identify simple patterns, such as counting the number of fingers on an image of a human hand.

To run the network yourself: 

  1. Download and unzip the zip file 
  2. Open and compile the project in an IDE or through the command line
  3. Run Network.java, changing the program's functionalities by modifying the config file. Instructions for the format of the config file are provided in the documentation of the Network file. The project currently includes sample training, testing, and config files that you can use for yourself. 
  4. Input data should be numerical values between 0 and 1. Therefore, to train or test the network on a pattern like detecting human fingers, you should find a way to convert the images into patterns of numbers. One way is to convert traditional image formats into bitmaps and read and scale the color values at each pixel. 
