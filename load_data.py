# coding=utf-8

# Load dataset, train and test.

# https://www.tensorflow.org/tutorials/load_data/images

import tensorflow as tf
import numpy as np
import pathlib
import os

# The following functions convert file paths to an (name, image_data, label) tuple

# Obtain label from the file_path
def get_label(file_path, class_names):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == class_names

# Get name of the image as 'class_name#img_name'
# https://stackoverflow.com/questions/54752287/get-input-filenames-from-tensorflow-dataset-iterators
def get_filename(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] + '#' + parts[-1]

def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels = 3)
    # Resize the image to the desired size
    return tf.image.resize(img, [224, 224])

# shuffle_buffer_size should be greater than or equal to the full size of the
# dataset.
def prepare_for_training(ds, batch_size, cache = True, shuffle_buffer_size = 50000,
                            data_aug = False):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size = shuffle_buffer_size)

    ds = ds.batch(batch_size)

    # Data augmentation
    if data_aug:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor = (-0.1, 0),
                                                            width_factor = (-0.1, 0)),
        ])
        ds = ds.map(lambda z, x, y: (z, data_augmentation(x, training = True), y),
                num_parallel_calls = tf.data.experimental.AUTOTUNE)

    # 'prefetch' lets the dataset fetch batches in the background while the
    # model is training.
    ds = ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

    return ds

def load_data(data_dir, class_names):

    def process_path(file_path):
        label = get_label(file_path, class_names)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        # Get image name
        filename = get_filename(file_path)
        return filename, img, label

    # Create a dataset of the file paths
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

    # Use Dataset.map to create a dataset of image, label pairs.
    # Set 'num_parallel_calls' so multiple images are loaded/processed in parallel.
    labeled_ds = list_ds.map(process_path, num_parallel_calls =
                                tf.data.experimental.AUTOTUNE)

    return labeled_ds

'''
Funtion to read any data set. The data set has to be in a folder with one subfolder
for class containing the images in that class. It returns a tf.dataset.
Args:
    train_path: Path to the training folder. String.
    test_path: Path to the test folder. String.
    batch_size: Batch size. Integer.
    train_size: Number of images in the training set. Integer.
    cache: Whether to cache the data set. Boolean.
'''
def load_train_test(train_path, test_path, batch_size, train_size,
                            cache = True):
    train_dir = pathlib.Path(train_path)
    # Read class names in alphanumeric order
    class_names = [item.name for item in train_dir.glob('*')]
    class_names = np.array(sorted(class_names))
    test_dir = pathlib.Path(test_path)
    labeled_train_ds = load_data(train_dir, class_names)
    labeled_train_ds =  prepare_for_training(labeled_train_ds, batch_size,
                                                cache = cache,
                                                shuffle_buffer_size = train_size)
    labeled_test_ds = load_data(test_dir, class_names)

    return labeled_train_ds, labeled_test_ds

'''
Funtion to read the cifar data sets, previously saved. It returns a tf.dataset.
Args:
    path_name: 'cifar10' or 'cifar100'. String
    batch_size: Batch size. Integer.
    noise: Type of noise. None for none, 'AN' for asymmetric noise and 'RA' for symmetric noise.
    rate: Noise rate. Integer between 0 and 100.
    data_aug: Whether to use data augmentation. Boolean.
    cache: Whether to cache the data set. Boolean.
'''
def load_train_test_cifar(path_name, batch_size, noise = None, rate = 0,
                            data_aug = False, cache = True):
    if noise is not None:
        train_data = path_name + '_train_data_' + noise + str(rate) + '.npy'
        train_labels = path_name + '_train_labels_' + noise + str(rate) + '.npy'
    else:
        train_data = path_name + '_train_data.npy'
        train_labels = path_name + '_train_labels.npy'

    test_data = path_name + '_test_data.npy'
    test_labels = path_name + '_test_labels.npy'

    train_data = np.load(train_data).astype(np.float32)
    train_labels = np.load(train_labels)
    test_data = np.load(test_data).astype(np.float32)
    test_labels = np.load(test_labels)

    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)

    fn_train = np.arange(len(train_data))
    fn_test = np.arange(len(test_data))

    train_ds = tf.data.Dataset.from_tensor_slices((fn_train, train_data, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((fn_test, test_data, test_labels))

    train_ds =  prepare_for_training(train_ds, batch_size, cache = cache,
                                                shuffle_buffer_size = 48000, data_aug = data_aug)
    return train_ds, test_ds
