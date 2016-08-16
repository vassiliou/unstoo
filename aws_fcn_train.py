from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import glob
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model_id")
args = parser.parse_args()

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


### Loading the model parameters

import aws_fcn_model as model
import uns
from settings import *
sys.path.append(MODEL_DEF_PATH)

modelname = 'model_'+args.model_id
mod = __import__(modelname)

train_dir = os.path.join(MODELCHECKPOINTS_PATH,'model_'+args.model_id+'_log')
os.makedirs(train_dir, exist_ok=True)

## put model params into namespace

checkpath=mod.checkpath
log_device_placement = mod.log_device_placement
learning_rate=mod.learning_rate
batch_size=mod.batch_size
max_steps=mod.max_steps
FC6_SHAPE=mod.FC6_SHAPE
FC7_SHAPE=mod.FC7_SHAPE
FC8_SHAPE=mod.FC8_SHAPE
try:
    randomize = mod.randomize
except AttributeError:
    randomize = False


layer_shapes=mod.layer_shapes

DOWNSAMPLE_FACTOR = mod.DOWNSAMPLE_FACTOR

NUM_CLASSES = mod.NUM_CLASSES
weights = mod.weights

#### Setting up the checkpoint (if exists) and making sure fc8 is initialized correctly

random_fc8 = True

if checkpath is not None:
    random_fc8 = False
    checkpath = os.path.join(base_path,checkpath)
    print('Retrieving model weights from ' +checkpath)

filepaths = uns.uns_files('train', 'records', args.model_id)

####

def compute_dice(preds, truth):
    numer = 2*np.sum(np.logical_and(preds, truth), axis=(1,2))
    denom = np.sum(preds, axis=(1,2)) + np.sum(truth, axis=(1,2))
    scores = np.empty(preds.shape[0])
    scores[denom<1] = 1  # denom==0 implies numer==0
    idx = denom>0
    scores[idx] = numer[idx]/denom[idx]
    return scores.mean()

###

def train():
  """Train model for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get the input  data.
    image_batch, mask_batch = model.inputs(filepaths,batch_size,train=train)

    mask_labels = tf.split(3, 2,mask_batch)[1]

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, pred_pix = model.inference(image_batch,layer_shapes,random_fc8)

    # Calculate loss.
    loss = model.loss(logits, mask_batch,NUM_CLASSES,weights=np.array(weights))

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = model.train(loss, global_step, batch_size, learning_rate)

    #accuracy = model.accuracy(logits,labels)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=log_device_placement))
    print('Initializing all variables...')
    sess.run(init)

    if random_fc8 == False:
      saver.restore(sess,checkpath)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)
    print('Starting queue runners...')

    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

    moving_dice=[]

    for step in xrange(max_steps):
      start_time = time.time()
      #test = sess.run(fuse_pool)
      #print(test.shape)
      #print(fc6_batch.get_shape())
      #print(pool_batch.get_shape())
      #print(mask_batch.get_shape())
      _, loss_value, pred_pixels,ground_truth = sess.run([train_op, loss, pred_pix,mask_labels])
      #print(sess.run(images)[0])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 5 == 0:
        num_examples_per_step = batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, batch loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 20 == 0:
          ground_truth=np.logical_not(ground_truth[:,:,:,0])
          dice = compute_dice(ground_truth,np.logical_not(pred_pixels))
          print("Dice score this batch: " + str(dice))
          moving_dice.append(dice)

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == max_steps:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        moving_avg_dice=np.array(moving_dice).mean()
        print('Averge dice score over last 1000 steps: ' + str(moving_avg_dice))
        moving_dice=[]


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
  tf.app.run()
