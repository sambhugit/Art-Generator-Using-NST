import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow.compat.v1 as tf
from cost import *
import Google_drive_downloader_vgg as vgg_down


current_dir = os.path.dirname(os.path.realpath('Art_Generator.py'))

content_image = scipy.misc.imread(os.path.join(current_dir,'Images/Content.jpg'))

style_image1 = scipy.misc.imread(os.path.join(current_dir,'Images/Style1.jpg'))
style_image2 = scipy.misc.imread(os.path.join(current_dir,'Images/Style2.jpg'))

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]



def contentf():
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as test:

        tf.compat.v1.set_random_seed(1)
        a_C = tf.compat.v1.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.compat.v1.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_content = compute_content_cost(a_C, a_G)
    return J_content
    
def gramf():
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as test:

        tf.compat.v1.set_random_seed(1)
        A = tf.compat.v1.random_normal([3, 2*1], mean=1, stddev=4)
        GA = gram_matrix(A)
    return GA
    
def stylef():
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as test:
        tf.compat.v1.set_random_seed(1)
        a_S1 = tf.compat.v1.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_S2 = tf.compat.v1.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        a_G = tf.compat.v1.random_normal([1, 4, 4, 3], mean=1, stddev=4)
        J_style_layer1 = compute_layer_style_cost(a_S1, a_G)
        J_style_layer2 = compute_layer_style_cost(a_S2, a_G)
    return J_style_layer1 , J_style_layer2

def tcostf():
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as test:
        np.random.seed(3)
        J_content = np.random.randn()    
        J_style1 = np.random.randn()
        J_style2 = np.random.randn()
        J = total_cost(J_content, J_style1, J_style2)
    return J

def model_nn(sess, input_image, num_iterations = 30000):
    
    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model['input'].assign(input_image))

    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        ### START CODE HERE ### (1 line)
        _ = sess.run(train_step)
        ### END CODE HERE ###
        
        # Compute the generated image by running the session on the current model['input']
        ### START CODE HERE ### (1 line)
        generated_image = sess.run(model['input'])
        ### END CODE HERE ###

        # Print every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js1, Js2 = sess.run([J, J_content, J_style_1, J_style_2])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style1 cost = " + str(Js1))
            print("style2 cost = " + str(Js2))
            
            # save current generated image in the "/output" directory
            save_image("Output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image('Output/generated_image.jpg', generated_image)
    
    return generated_image
        

if __name__ == "__main__":

    content_image_resized = Image_size_optimizer(content_image,"Content")
    style_image_resized_1 = Image_size_optimizer(style_image1,"Style1")
    style_image_resized_2 = Image_size_optimizer(style_image2,"Style2")

    vgg_down.weight_file_id()

    J_content = contentf()
    GA = gramf()
    J_style_1, J_style_2 = stylef()
    J = tcostf()

    tf.compat.v1.reset_default_graph()

    sess = tf.compat.v1.InteractiveSession()
    generated_image = generate_noise_image(content_image_resized)

    model = load_vgg_model("Imagenet/imagenet-vgg-verydeep-19.mat")

    sess.run(model['input'].assign(content_image_resized))

    out = model['conv4_2']
    a_C = sess.run(out)
    a_G = out

    J_content = compute_content_cost(a_C, a_G)
    sess.run(model['input'].assign(style_image_resized_1))
    J_style_1 = compute_style_cost(model, STYLE_LAYERS, sess)
    sess.run(model['input'].assign(style_image_resized_2))
    J_style_2 = compute_style_cost(model, STYLE_LAYERS, sess)
    J = total_cost(J_content, J_style_1, J_style_2,  alpha = 10, beta = 40, gamma = 25)

    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)

    model_nn(sess, generated_image)
