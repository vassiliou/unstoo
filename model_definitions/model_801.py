# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 15:33:05 2016

@author: chrisv
"""

model_name = '801'

checkpath = None #'model.ckpt-11000'

log_device_placement = False

learning_rate = 10e-6
batch_size = 15
max_steps = 40000

FC6_SHAPE=[512,256]
FC7_SHAPE=[256,128]
FC8_SHAPE=[128,64]

layer_shapes=(FC6_SHAPE, FC7_SHAPE, FC8_SHAPE)

DOWNSAMPLE_FACTOR = 2

NUM_CLASSES = 2
weights = [10,1]
