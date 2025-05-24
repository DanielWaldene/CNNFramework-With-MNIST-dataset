# CNN
1. use 'make' to compile and link
2. use 'make run' to run the program
3. use 'make clean' to clean build directory
3. to change structure of the CNN make network structure in main and then add the layers in order that you want
4. Makefile should be able to compile and link any additional src files
5. To view the CNN in action uncomment the 'printTensor3()' and 'printVector()' functions inside of the 'forward()' and 'backward()' functions inside network.cpp.
6. To change the number of images used during training and testing change the number of images loaded into the class imageset which is determined by a parameter in the imageset constructor.