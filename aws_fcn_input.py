from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from settings import *


def read_record(filename_queue):
    class FCNRecord(object):
        pass
    result = FCNRecord()
    result.mask_height = int(420/DOWNSAMPLE_FACTOR)
    result.mask_width = int(580/DOWNSAMPLE_FACTOR)
    result.mask_depth = 1
    result.img_depth = 1
    img_len = result.mask_height*result.mask_width*result.img_depth
    mask_len = result.mask_height*result.mask_width*result.mask_depth
    record_len = img_len + mask_len

    reader = tf.FixedLengthRecordReader(record_bytes=record_len)
    result.key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    #print(record_bytes.get_shape())
    int_image = tf.reshape(tf.slice(record_bytes, [0], [img_len]),[result.mask_height, result.mask_width])
    rgb_image = tf.pack([int_image,int_image,int_image])
    rgb_img = tf.transpose(rgb_image,(1,2,0))
    result.image = tf.cast(rgb_img,tf.float32)
    bool_mask = tf.cast( tf.reshape(tf.slice(record_bytes, [img_len], [mask_len]),[result.mask_height, result.mask_width]), tf.bool)
    hot_mask= tf.pack( [bool_mask, tf.logical_not(bool_mask)])
    h_mask = tf.transpose(hot_mask,(1,2,0))
    result.mask = tf.cast(h_mask, tf.float32)
    return result


def _generate_fcn_batch(img, mask, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 1
  if shuffle==True:
    num_preprocess_threads=16
    img_batch, mask_batch = tf.train.shuffle_batch(
        [img, mask],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    img_batch, mask_batch = tf.train.batch(
        [img, mask],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)
       
  # Display the images in the visualizer.
  tf.image_summary('images', img_batch)
  # Display the masks in the visualizer.
  tf.image_summary('masks', 255*mask_batch[:,:,:,:1])
  # print(images.get_shape())
  # print(label_batch.get_shape())
  return img_batch, mask_batch


def inputs(filenames, batch_size,train=True):
  """Construct input for evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the MNIST data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """


  #for f in filenames:
   # if not tf.gfile.Exists(f):
    #  raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  if train == True:
    filename_queue = tf.train.string_input_producer(filenames)
  else:
    filename_queue = tf.train.string_input_producer(filenames,shuffle=False)

  # Read examples from files in the filename queue.
  read_input = read_record(filename_queue)

  # Subtract off the mean and divide by the variance of the pixels??
 
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.1
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)

  #print(min_queue_examples)
  print ('Filling queue with %d bottlenecked inputs before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_fcn_batch(read_input.image, read_input.mask,
                                         min_queue_examples, batch_size,
                                         shuffle=train)

