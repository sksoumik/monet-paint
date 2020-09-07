import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import numpy as np

from cycle_gan import Generator, Discriminator
from cycle_gan import CycleGan

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


def generator_discrimanator():
    with strategy.scope():
        monet_generator = Generator(
        )  # transforms photos to Monet-esque paintings
        photo_generator = Generator(
        )  # transforms Monet paintings to be more like photos

        monet_discriminator = Discriminator(
        )  # differentiates real Monet paintings and generated Monet paintings
        photo_discriminator = Discriminator(
        )  # differentiates real photos and generated photos
    return monet_generator, photo_generator, monet_discriminator, photo_discriminator


'''
Define the loss function
'''

with strategy.scope():

    def discriminator_loss(real, generated):
        real_loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)

        generated_loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated),
                                                      generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5


'''
The generator wants to fool the discriminator into thinking the generated image is real. 
The perfect generator will have the discriminator output only 1s. 
Thus, it compares the generated image to a matrix of 1s to find the loss.
'''

with strategy.scope():

    def generator_loss(generated):
        return tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated),
                                                      generated)


'''
We want our original photo and the twice transformed photo to be similar to one another. 
Thus, we can calculate the cycle consistency loss be finding the average of their difference.
'''

with strategy.scope():

    def calc_cycle_loss(real_image, cycled_image, LAMBDA):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

        return LAMBDA * loss1


'''
The identity loss compares the image with its generator (i.e. photo with photo generator). 
If given a photo as input, we want it to generate the same image as the image was originally a photo. 
The identity loss compares the input with the output of the generator.
'''

with strategy.scope():

    def identity_loss(real_image, same_image, LAMBDA):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return LAMBDA * 0.5 * loss


'''
Train the cycleGAN
Let's compile our model. Since we used tf.keras.Model to build our CycleGAN, we can just ude the fit function to train our model.
'''

with strategy.scope():
    monet_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    monet_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

with strategy.scope():

    def gan_model():
        monet_generator, photo_generator, monet_discriminator, photo_discriminator = generator_discrimanator(
        )
        cycle_gan_model = CycleGan(monet_generator, photo_generator,
                                   monet_discriminator, photo_discriminator)
        return cycle_gan_model


def main():
    MONET_FILENAMES, PHOTO_FILENAMES = filenames()
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

    cycle_gan_model = gan_model()

    cycle_gan_model.compile(m_gen_optimizer=monet_generator_optimizer,
                            p_gen_optimizer=photo_generator_optimizer,
                            m_disc_optimizer=monet_discriminator_optimizer,
                            p_disc_optimizer=photo_discriminator_optimizer,
                            gen_loss_fn=generator_loss,
                            disc_loss_fn=discriminator_loss,
                            cycle_loss_fn=calc_cycle_loss,
                            identity_loss_fn=identity_loss)
    cycle_gan_model.fit(tf.data.Dataset.zip((monet_ds, photo_ds)), epochs=50)

    # Visualize monet-esque photos that are generated from our model

    monet_generator, _, _, _ = generator_discrimanator()

    _, ax = plt.subplots(5, 2, figsize=(12, 12))
    for i, img in enumerate(photo_ds.take(5)):
        prediction = monet_generator(img, training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

        ax[i, 0].imshow(img)
        ax[i, 1].imshow(prediction)
        ax[i, 0].set_title("Input Photo")
        ax[i, 1].set_title("Monet-esque")
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")
    plt.show()


if __name__ == "__main__":
    main()