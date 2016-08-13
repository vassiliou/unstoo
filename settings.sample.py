import os
import sys

base_path = '/home/ubuntu/'

TRAIN_PATH = os.path.join(base_path, 'train/')
TEST_PATH = os.path.join(base_path, 'test/')
MODELCHECKPOINTS_PATH = os.path.join(base_path, 'model-checkpoints/')
PROBABILITIES_PATH = os.path.join(base_path, 'probability-outputs/')
MASKOUTPUT_PATH = os.path.join(base_path, 'mask-output')
MODEL_DEF_PATH = os.path.join(base_path, 'unstoo/model_definitions/')
TRAINING_BIN_PATH = os.path.join(base_path, 'unstoo/training.bin')
RECORD_DIRECTORY = TRAIN_PATH

sys.path.append(MODEL_DEF_PATH)

vgg_path = os.path.join(base_path, 'vgg16.npy')

DOWNSAMPLE_FACTOR = 2

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 500
