# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 15:33:05 2016

@author: chrisv
"""

model_name = '701'

# which checkpoint file to use when evaluating
checkpath = None#'model.ckpt-25000'

log_device_placement = False

learning_rate = 4*10e-6
batch_size = 15
max_steps = 40000

FC6_SHAPE=[512,256]
FC7_SHAPE=[256,256]
FC8_SHAPE=[256,125]

layer_shapes=(FC6_SHAPE, FC7_SHAPE, FC8_SHAPE)

DOWNSAMPLE_FACTOR = 2

NUM_CLASSES = 2
weights = [10,1]
