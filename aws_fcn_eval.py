from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from settings import *

import aws_fcn_model as model
import aws_fcn_input as inpt

import skimage.io

import uns
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model_id")
parser.add_argument("evaluate")
args = parser.parse_args()

### load up the model definition:

import aws_fcn_model as model

from settings import *

sys.path.append(MODELCHECKPOINTS_PATH)

modelname = 'model_'+args.model_id
mod = __import__(modelname)

## put model params into namespace

checkpath=mod.checkpath
log_device_placement = mod.log_device_placement
learning_rate=mod.learning_rate
batch_size=mod.batch_size
max_steps=mod.max_steps
FC6_SHAPE=mod.FC6_SHAPE
FC7_SHAPE=mod.FC7_SHAPE
FC8_SHAPE=mod.FC8_SHAPE

layer_shapes=mod.layer_shapes

NUM_CLASSES = mod.NUM_CLASSES
weights = mod.weights

### Enter the directory containing the raw input images, and a target directory where to write the predictions:
#RECORD_DIRECTORY = os.path.join(base_path,'validation')

prediction_directory = os.path.join(PROBABILITIES_PATH, modelname)
os.makedirs(prediction_directory, exist_ok=True)

### Setup paths to model checkpoint directory:

checkpath = os.path.join(MODELCHECKPOINTS_PATH, modelname+'_log/', checkpath)
print('Retrieving model weights from ' + checkpath)


#### We need to get a hold of the filenames of the records to run inference on  Eventually, we'll do this via uns. For debugging now, do it manually..

#pattern = os.path.join(RECORD_DIRECTORY, '*.rec')
#print(pattern)
#filepaths = sorted(glob.glob(pattern))
filepaths = uns.uns_files(args.evaluate, 'records', args.model_id)


####


def evalmodel(file_paths,bsize=1, source_dir=RECORD_DIRECTORY,target_directory=prediction_directory):
  """Evaluate the model against a test dataset

    Saves pixel probability predictions and masks as an array of shape [# examples] , 2, 210, 290
    Slice [:,0,:,:] corresponds to predictions and  slice [:,1,:,0] corresponds to ground truth (or zeros, if unknown)"""


  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get the bottled data.

    image_batch, mask_batch = model.inputs(file_paths,bsize,train=False)

    # The model's predictions

    logits, prediction = model.inference(image_batch,layer_shapes,train=False,random_fc8=True)

    pixel_probabilities = tf.reshape( tf.nn.softmax(tf.reshape(logits, (-1, NUM_CLASSES))) , (-1,210,290,NUM_CLASSES))

    # Calculate loss.
    #loss = model.loss(logits, mask_batch,NUM_CLASSES)

    #accuracy = model.accuracy(logits,labels)

    #train_op = model.train(loss, global_step)

    # Build the summary operation based on the TF collection of Summaries.

    summary_op = tf.merge_all_summaries()

    saver = tf.train.Saver(tf.all_variables())

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.log_device_placement=FLAGS.log_device_placement
    sess = tf.Session(config=config)


    # Start running operations on the Graph.
    print('Initializing all variables...', end='')
    sess.run(init)
    print('done')

    # Restore the model from the checkpoint

    saver.restore(sess,checkpath)

    # Start the queue runners.
    print('Starting queue runners...', end='')
    tf.train.start_queue_runners(sess=sess)
    print('done')

    ### The main loop: for each filename,

    for counter, file_path in enumerate(file_paths):
      probabilities, ground_truth = sess.run([pixel_probabilities,mask_batch])
      probabilities, ground_truth = probabilities[0,:,:,0], ground_truth[0,:,:,0]

      to_write = np.array([probabilities, ground_truth])
      ## Write the prediction to disk
      f_name = os.path.split(file_path)[1]
      f_name = os.path.splitext(f_name)[0]
      write_path = os.path.join(target_directory,f_name)
      np.save(write_path,to_write)
      if counter % 10 ==0:
        print('Saving prediction {d} to: '.format(d=counter) + str(write_path) )



def main(argv=None):  # pylint: disable=unused-argument
    print('Running predicions on ' + filepaths[0] + '...')
    evalmodel(filepaths)

if __name__ == '__main__':
  tf.app.run()
