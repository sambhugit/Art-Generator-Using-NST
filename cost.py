import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
#from keras.models import load_model
#from keras.applications.vgg19 import VGG19

tf.compat.v1.disable_eager_execution()

current_dir = os.path.dirname(os.path.realpath('cost.py'))

content_image = scipy.misc.imread(os.path.join(current_dir,'Images/Content.jpg'))

style_image1 = scipy.misc.imread(os.path.join(current_dir,'Images/Style1.jpg'))
style_image2 = scipy.misc.imread(os.path.join(current_dir,'Images/Style2.jpg'))

def compute_content_cost(a_C, a_G):
    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_C_unrolled = tf.transpose(tf.reshape(a_C, [-1]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [-1]))
    
    J_content = tf.reduce_sum((a_C_unrolled - a_G_unrolled)**2) / (4 * n_H * n_W * n_C)
    
    return J_content

def gram_matrix(A):
   
    GA = tf.matmul(A, tf.transpose(A)) 

    return GA

def compute_layer_style_cost(a_S, a_G):
    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_S = tf.reshape(a_S, [n_H*n_W, n_C])
    a_G = tf.reshape(a_G, [n_H*n_W, n_C])

    
    GS = gram_matrix(tf.transpose(a_S)) 
    GG = gram_matrix(tf.transpose(a_G))

    J_style_layer = tf.reduce_sum((GS - GG)**2) / (4 * n_C**2 * (n_W * n_H)**2)
    
    
    return J_style_layer

def compute_style_cost(model, STYLE_LAYERS, sess):
    
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style_1, J_style_2, alpha = 10, beta = 40, gamma = 25):
    
    #gamma = 30
    
    J = alpha * J_content + beta * J_style_1 + gamma*J_style_2
    
    return J

def Image_size_optimizer(image_path,type_of_image):

    image = Image.open('Images/%s.jpg'%type_of_image)
    image = image.resize((400, 300))
    image.save('%s400x300.jpg'%type_of_image)

    image = scipy.misc.imread("%s400x300.jpg"%type_of_image)

    image = reshape_and_normalize_image(image)

    return image


