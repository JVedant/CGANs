import tensorflow as tf
from tqdm import tqdm
from model import Generator, Discriminator
import config
import os
from utils import generate_and_save_images
from loss import generator_loss, discriminator_loss
from glob import glob
from dataset import parse_image, load_images


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


generator = Generator()
discriminator = Discriminator()
generator_optimizer = discriminator_optimizer = tf.keras.optimizers.Adam(config.LR)


@tf.function
def train_step(images):
    noise = tf.random.normal([config.BATCH_SIZE, config.N_DIMS])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        dis_loss = discriminator_loss(real_output, fake_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = dis_tape.gradient(dis_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_loss, dis_loss


def train():
    images = glob(config.IMAGE_DIR)

    dataset = tf.data.Dataset.list_files(images)
    dataset = dataset.map(parse_image, num_parallel_calls=20)
    dataset = dataset.map(load_images, num_parallel_calls=20)
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=config.BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    sample_seed = config.SEED

    checkpoint_dir = '../Logs/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')

    for epoch in tqdm(range(1, config.EPOCHS), desc="Epoch"):
        for image_batch in tqdm(dataset, desc="Image Batch", total=(config.STEPS_PER_EPOCH)):
            gen_loss, dis_loss = train_step(image_batch)

        print(f"gen_loss:{gen_loss} | disc_loss:{dis_loss}")
        generate_and_save_images(generator, sample_seed, epoch)

        ckpt_manager.save()


if __name__ == "__main__":
    train()