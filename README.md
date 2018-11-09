# mt: Generative model and inference to learn handwritten characters
This part of the code of my master's thesis. Its purpose is to recognize characters. Given an input image containing a character, 
it should learn how to draw the character. This is done with stochastic variational inference, where a neural network is used to 
predict the values of the generative model. 

## Installation
The project is implemented in python using pyro, so to be able to run the project, the following packages and frameworks have to be installed: 
  - python (3.x)
  - pytorch (https://pytorch.org)
  - pyro (http://pyro.ai)
  - opencv (https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html)
  - matplotlib
 
 In the current version, CUDA is required. 
 
  
## How to run the code
To run the project, the *main.py* file has to be executed. It creates and saves the input images, calls the optimization 
process and visualizes the loss and the parameters. 

*prior.py* contains the generative model to generate a character, both for generating a first input character and for generating 
the inferred drawn characters. Characters are always drawn in an image, so here the terms "character" and "image" can be used simultaneously. 

*modelAndInference.py* contains the variational inference part. It calls the generative model and the neural network.

*visualize.py* visualizes lines defined by control points.

  
## Author
* **Nicolina Wenzler**
  
  
