import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from kaggle_datasets import KaggleDatasets
import matplotlib.pyplot as plt
import numpy as np

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMAGE_SIZE = [256, 256]




def filenames():
    PATH = "input/" 
    MONET_FILENAMES = tf.io.gfile.glob(str(PATH + '/monet_tfrec/*.tfrec'))
    print('Monet TFRecord Files:', len(MONET_FILENAMES))

    PHOTO_FILENAMES = tf.io.gfile.glob(str(PATH + '/photo_tfrec/*.tfrec'))
    print('Photo TFRecord Files:', len(PHOTO_FILENAMES))
    return MONET_FILENAMES, PHOTO_FILENAMES 


'''
images for the competition are already sized to 256x256. As these images are RGB images, 
set the channel to 3. Additionally, we need to scale the images to a [-1, 1] scale. 
Because we are building a generative model, 
we don't need the labels or the image id so we'll only return the image from the TFRecord.
'''
def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def read_tfrecord(example):
    tfrecord_format = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    return image



# function to extract the image from the files.
def load_dataset(filenames, labeled=True, ordered=False):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTOTUNE)
    return dataset




def main():
    MONET_FILENAMES, PHOTO_FILENAMES  = filenames()
    monet_ds = load_dataset(MONET_FILENAMES, labeled=True).batch(1)
    photo_ds = load_dataset(PHOTO_FILENAMES, labeled=True).batch(1)

    example_monet = next(iter(monet_ds))
    example_photo = next(iter(photo_ds))

    # visualize a photo example 
    plt.subplot(121)
    plt.title('Photo')
    plt.imshow(example_photo[0] * 0.5 + 0.5)
    plt.show()

    # visualize a Monet example.
    plt.subplot(122)
    plt.title('Monet')
    plt.imshow(example_monet[0] * 0.5 + 0.5)
    plt.show() 

    
