import os
from datetime import datetime
import glob
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

IMAGE_DIR = "../DATA/Mixed_images/*.jpg"

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3

LR = 3e-4

BATCH_SIZE = 4
BUFFER_SIZE = len(os.listdir('../DATA/Mixed_images'))
EPOCHS = 50

N_DIMS = 150
SEED = tf.random.normal([4, N_DIMS])

MODEL_DIR = "../models"

STEPS_PER_EPOCH = len(glob.glob(IMAGE_DIR)) // BATCH_SIZE

NOW = (str(datetime.now()).replace(' ', '_').replace(':', '_')).split(".")[0]
TB_LOGS = f'../Logs/{NOW}'