# Art-Generator-Using-NST
###
This project is an upgradation of Neural Style Transfer model implemented by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge [1].\
\
In this project, I have taught a neural network to draw a painting in a white screen using styles of two different paintings.\
\
**(Below is the white screen on which the network will paint. Use dark mode theme to see it.)**

<p align="left">
  <img src="/Images/Content.jpg" width="350" title="hover text">
</p>

**( This is the first Style image. )**

<p align="left">
  <img src="/Images/Style1.jpg" width="350" title="hover text">
</p> 


**( This is the second Style image. )**

<p align="left">
  <img src="/Images/Style2.jpg" width="350" title="hover text">
</p> 

### 
An art generator model using VGG19 . The model extracts style of art from both paintings and merges it with our white screen image to generate a new painting using Neural Style Transfer. Gradient Descent and Transfer Learning is used to reduce Cost Function and optimize generated image.
### 
Clone this repo and then pip install requirements.txt file.
### 
Then run Art_Generator.py to see generated images.
