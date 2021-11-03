import tensorflow as tf
import config
import random

def parse_image(image_path:str) -> dict:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    return {"image": image}

@tf.function
def normalize(input_image: tf.Tensor):
    input_image = tf.cast(input_image, dtype=tf.float32)
    input_image = input_image / 255.0
    return input_image

@tf.function
def load_images(datapoint: dict):
    input_image = datapoint['image']
    input_image = tf.image.resize(input_image, size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))

    if tf.random.uniform(()) > 0.4:
        input_image = tf.image.random_flip_left_right(image=input_image)

    if tf.random.uniform(()) > 0.7:
        input_image = tf.image.random_flip_up_down(image=input_image)

    if tf.random.uniform(()) > 0.4:
        rot_range = random.randint(1, 4)
        input_image = tf.image.rot90(input_image, k=rot_range)

    input_image = normalize(input_image)

    return input_image