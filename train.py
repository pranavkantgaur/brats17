# -*- coding: utf-8 -*-
# Implementation of Wang et al 2017: Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks. https://arxiv.org/abs/1709.00382

# Author: Guotai Wang
# Copyright (c) 2017-2018 University College London, United Kingdom. All rights reserved.
# http://cmictig.cs.ucl.ac.uk
#
# Distributed under the BSD-3 licence. Please see the file licence.txt
# This software is not certified for clinical use.
#
from __future__ import absolute_import, print_function

import numpy as np
import nibabel as ni
import random
from scipy import ndimage
import time
import os
import sys
import tensorflow as tf
#from tensorflow.contrib.data import Iterator
from tensorflow.contrib.layers.python.layers import regularizers
from niftynet.layer.loss_segmentation import LossFunction
from util.data_loader import *
from util.train_test_func import *
from util.parse_config import parse_config
from util.MSNet import MSNet

class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'MSNet':
            return MSNet
        # add your own networks here
        print('unsupported network:', name)
        exit()

def train(config_file):
    # 1, load configuration parameters
    config = parse_config(config_file)
    config_data  = config['data']
    config_net   = config['network']
    config_train = config['training']
     
    random.seed(config_train.get('random_seed', 1))
    assert(config_data['with_ground_truth'])

    net_type    = config_net['net_type']
    net_name    = config_net['net_name']
    class_num   = config_net['class_num']
    batch_size  = config_data.get('batch_size', 5)
   
    # 2, construct graph
    full_data_shape  = [batch_size] + config_data['data_shape']
    full_label_shape = [batch_size] + config_data['label_shape']
    x = tf.placeholder(tf.float32, shape = full_data_shape)
    w = tf.placeholder(tf.float32, shape = full_label_shape)   
    y = tf.placeholder(tf.int64,   shape = full_label_shape)
   
    w_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    b_regularizer = regularizers.l2_regularizer(config_train.get('decay', 1e-7))
    net_class = NetFactory.create(net_type)
    net = net_class(num_classes = class_num,
                    w_regularizer = w_regularizer,
                    b_regularizer = b_regularizer,
                    name = net_name)
    net.set_params(config_net)
    predicty = net(x, is_training = True) # prediction corresponding to input x
    proby    = tf.nn.softmax(predicty) # represents the probbaility of ith voxel belonging to fiducial or non-fiducial class
    # TODO convert class probability to labels
    # for each voxel(x,y) identify the channel id(or z) with highest probability
    #  label(x, y, z) = max_i_(prob_class_k(x,y,z))
    class_labels = tf.argmax(proby, 4)	
    print("structure of class labels: ", class_labels.shape)	

    # compute convolution of predicty with shape priors
    shape_prior_number = config_data['shape_prior_number'] # read shape prior template data
    shape_prior_data_root = config_data['shape_prior_data_root']
    shape_prior_names = config_data['shape_prior_names']
    
    filter_data = []	 
    filter_width = []
    filter_height = []
    filter_depth = []	
    #filter_data_nii = ni.load(shape_prior_data_root + shape_prior_names), load all filters in a loop, keep them ready for convolution
    for filter_id in range(shape_prior_number):	
    	#shape_prior_names.get() #TODO load fiducial template filename from the file at shape_prior_names file location
	filter_data_file_path = shape_prior_data_root#os.path.join(shape_prior_data_root, shape_prior_names)	
        filter_data_nii = ni.load(filter_data_file_path)
	filter_data.append(filter_data_nii.get_data()) # 3D array
    	filter_width.append(int(filter_data_nii.shape[0]))
    	filter_height.append(int(filter_data_nii.shape[1]))
    	filter_depth.append(int(filter_data_nii.shape[2]))
    
    in_3d = class_labels #in_3d = #initialize from predicty TODO check

    print("Maxium element in class labels data: ", tf.reduce_max(class_labels))
    print("Minimum element in class labels data: ", tf.reduce_min(class_labels))

    in_width = int(in_3d.shape[2]) 
    in_height = int(in_3d.shape[3]) 
    in_depth = int(in_3d.shape[1])
    in_num_batches = int(in_3d.shape[0])

    filter_id = 0
    in_channels = 1 	# TODO why 2 channels?
    out_channels = 1
    print("input shape is: ", in_3d.shape)
    in_3d = tf.cast(in_3d, tf.float16)
    print("input data type is: ", in_3d.dtype)
    print("filter shape is: ", filter_data[0].shape)
    filter_data[0] = tf.cast(filter_data[0], tf.float16)
    print("filter data type is: ", filter_data[0].dtype)
    prior_conv = []
    for filter_id in range(shape_prior_number):
      kernel_3d = tf.reshape(filter_data[filter_id], [filter_depth[filter_id], filter_height[filter_id], filter_width[filter_id], in_channels, out_channels])
      print("After reshaping filter: ", kernel_3d.shape)
      #kernel_3d = tf.to_float(kernel_3d)	
      in_3d = tf.reshape(in_3d, [in_num_batches, in_depth, in_height, in_width, out_channels])
      print("input shape after reshaping: ", in_3d.shape)	
      strides = [1,1,1,1,1]
      padding = "VALID"
      #dilations = 
      # call conv9d to compute convolution of filters over predicty
      prior_conv.append(tf.nn.conv3d(in_3d, kernel_3d, strides, padding))

    prior_regularization_parameter = 0.0000005# TODO based on regularization parameter, set value using cross validation

    #Compute loss term in customized/regularized loss function 
    loss_func = LossFunction(n_class = class_num, loss_type = 'Shape_Prior')  
#    loss_func = LossFunction(n_class=class_num)
    loss = loss_func(predicty, y, prior_conv, prior_regularization_parameter, weight_map = w)
    print('size of predicty:',predicty)
    
  
    # 3, initialize session and saver
    lr = config_train.get('learning_rate', 1e-3)
    opt_step = tf.train.AdamOptimizer(lr).minimize(loss)




    sess = tf.InteractiveSession()   
    sess.run(tf.global_variables_initializer())  
    saver = tf.train.Saver()
    
    dataloader = DataLoader(config_data)
    dataloader.load_data()
    # TODO does it load the fiducial template also, how to access it?
    
    # 4, start to train
    loss_file = config_train['model_save_prefix'] + "_loss.txt"
    start_it  = config_train.get('start_iteration', 0)
    if( start_it > 0):
        saver.restore(sess, config_train['model_pre_trained'])
    loss_list, temp_loss_list = [], []
    for n in range(start_it, config_train['maximal_iteration']):
        train_pair = dataloader.get_subimage_batch()
        tempx = train_pair['images']
        tempw = train_pair['weights']
        tempy = train_pair['labels']
        opt_step.run(session = sess, feed_dict={x:tempx, w: tempw, y:tempy})

        if(n%config_train['test_iteration'] == 0): # testing at every 100th iteration 
            batch_dice_list = []
            for step in range(config_train['test_step']):
                train_pair = dataloader.get_subimage_batch()
                tempx = train_pair['images']
                tempw = train_pair['weights']
                tempy = train_pair['labels']
                dice = loss.eval(feed_dict ={x:tempx, w:tempw, y:tempy}) #evaluation of loss function
                batch_dice_list.append(dice) #append dice losses
            batch_dice = np.asarray(batch_dice_list, np.float32).mean() # batch dice
            t = time.strftime('%X %x %Z')
            print(t, 'n', n,'loss', batch_dice)
            loss_list.append(batch_dice)
            np.savetxt(loss_file, np.asarray(loss_list))

        if((n+1)%config_train['snapshot_iteration']  == 0): #snapshot of the network
            saver.save(sess, config_train['model_save_prefix']+"_{0:}.ckpt".format(n+1))
    sess.close()
    
if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print('Number of arguments should be 2. e.g.')
        print('    python train.py config17/train_wt_ax.txt')
        exit()
    config_file = str(sys.argv[1])
    assert(os.path.isfile(config_file))
    train(config_file)
