# coding=utf-8

# Functions to handle dictionaries for relabelling or removing instances

import tensorflow as tf
import numpy as np
from load_data import *
from collections import deque

def add_change_in_dict(dict, key, new_class):
    # The next line create a new item in the dictionary if key does not
    # exist in dict and change its value if it exists.
    # At this point, key and new_class are not tensors.
    dict[key] = new_class

def add_removal_in_dict(dict, key):
    dict[key] = 'remove'

def add_record(dict, tf_keys, logits, record_length):
    keys = tf_keys.numpy()
    for idx in range(len(logits)):
        key = keys[idx]
        pred = np.argmax(logits[idx])

        if key not in dict:
            dict[key] = deque([0] * record_length, record_length)

        dict[key].appendleft(pred)

def add_not_change(dict, key):
    dict[key] = -1

def clear_record(dict, key, record_length):
    dict[key] = deque([0] * record_length, record_length)

def remove_from_dictionary(dict, key):
    if key in dict:
        dict.pop(key)

def perform_changes_in_batch(fn_batch, y_batch, changes_dict, not_change_dict):
    # In order to be able to change the batch of labels, we need to create
    # a new one, copying what we do not want to change. To do that, we use
    # .numpy(), then we change the labels and then we use .convert_to_tensor().
    new_labels = y_batch.numpy()
    fn_batch_np = fn_batch.numpy()

    for fn in fn_batch_np:
        if fn in changes_dict:
            if fn not in not_change_dict: # or not_change_dict[fn] == 0:
                idx, = np.where(fn_batch_np == fn)
                new_labels[idx] = changes_dict[fn]

    new_labels = tf.convert_to_tensor(new_labels)
    return new_labels

def perform_removals_in_batch(fn_batch, x_batch, y_batch, removals_dict):
    # In this case, we need to remove the corresponding images and their
    # corresponding labels and filenames.
    fn_batch_np = fn_batch.numpy()

    to_remove = []

    # Get indexes of the images to remove
    for fn in fn_batch_np:
        if fn in removals_dict:
            idx, = np.where(fn_batch_np == fn)
            to_remove.append(idx)

    new_labels = np.delete(y_batch.numpy(), to_remove, axis = 0)
    new_images = np.delete(x_batch.numpy(), to_remove, axis = 0)
    new_filenames = np.delete(fn_batch_np, to_remove)

    new_labels = tf.convert_to_tensor(new_labels)
    new_images = tf.convert_to_tensor(new_images)
    new_filenames = tf.convert_to_tensor(new_filenames)

    return new_filenames, new_images, new_labels

def check_record(fn_batch, record_dict, removals_dict, record_length, changes_dict):
    fn_batch_np = fn_batch.numpy()
    for i in range(len(fn_batch_np)):
        record_i = record_dict[fn_batch_np[i]]
        changes = 0
        for j in range(len(record_i) - 1):
            if record_i[j] != record_i[j+1]:
                changes += 1
        if changes >= record_length - 1:
            add_removal_in_dict(removals_dict, fn_batch_np[i])
            # and remove it from the changes_dict if it exists
            remove_from_dictionary(changes_dict, fn_batch_np[i])

def check_high_prob_wrong_label(fn_batch, y_batch, logits, changes_dict,
                                prob_thres, record_dict, removals_dict,
                                not_change_dict, record_length):
    y_batch_np = y_batch.numpy()
    fn_batch_np = fn_batch.numpy()
    for idx in range(len(logits)):
        if np.argmax(logits[idx]) != np.argmax(y_batch_np[idx]) and max(logits[idx]) > prob_thres:
            if fn_batch_np[idx] not in not_change_dict:
                # tf.print(max(logits[idx]))
                new_class = y_batch_np[idx]
                new_class[np.argmax(y_batch_np[idx])] = False
                new_class[np.argmax(logits[idx])] = True
                add_change_in_dict(changes_dict, fn_batch_np[idx], new_class)
                add_not_change(not_change_dict, fn_batch_np[idx])
                clear_record(record_dict, fn_batch_np[idx], record_length)
                remove_from_dictionary(removals_dict, fn_batch_np[idx])

def update_not_change_dict(dict, not_change_epochs):
    to_remove = []
    for key, value in dict.items():
        if value == not_change_epochs:
            to_remove.append(key)
        else:
            dict[key] = value + 1

    for key in to_remove:
        remove_from_dictionary(dict, key)
