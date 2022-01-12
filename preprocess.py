import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from tensorflow.python.ops.gen_array_ops import deep_copy

def add_blocks(image, spacing = .2, size = .1):
    offset = [int(random.random() * spacing * image.shape[i]) for i in range(2)]
    block_size = int(size * image.shape[0]), int(size * image.shape[1]), image.shape[2]
    block = np.zeros(block_size)
    mask = np.ones(image.shape)
    for i in range(int(1/spacing)):
        x_min = max(0, int(offset[0] + image.shape[0] * spacing * i))
        x_max = min(image.shape[0], int(offset[0] + image.shape[0] * spacing * i + block_size[0]))
        for j in range(int(1/spacing)):
            y_min = max(0, int(offset[1] + image.shape[1] * spacing * j))
            y_max = min(image.shape[1], int(offset[1] + image.shape[1] * spacing * j + block_size[1]))
            mask[x_min:x_max, y_min:y_max, :] = block[:x_max-x_min, :y_max-y_min]
    noise = np.random.uniform(size = image.shape)
    return image * mask + noise * (1 - mask)

def preprocessing_fn(image_in):
    image = add_blocks(image_in)
    return image

if __name__ == '__main__':
    image_path = './cub200data/CUB_200_2011/test/197.Marsh_Wren/Marsh_Wren_0119_188404.jpg'
    _, ax = plt.subplots(1, 2, figsize = (8, 8))
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (256, 256))
    image = image.numpy()
    image_before = image
    image_after = add_blocks(image)
    ax[0].imshow(image_before)
    ax[1].imshow(image_after)
    plt.show()