# coding=utf-8

# Fit method to train the model and record the loss of each sample

import tensorflow as tf
from dictionaries import *
from load_data import *
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statistics import NormalDist
import os
import tempfile
import scipy.stats as stats

# https://tensorflow.google.cn/guide/keras/train_and_evaluate?hl=zh-cn#part_ii_writing_your_own_training_evaluation_loops_from_scratch
'''
Function to fit the model.
Args:
    model: model to fit. Instace of Keras Model.
    train_dataset: training data set. TF Dataset as read in load_data.
    optimizer: optimizer to use. Instance of Keras Optimizer.
    epochs: number of epochs to train. Integer.
    global_batch_size: global batch size used during training. Integer.
    quantile_prob: quantile to use for the probability threshold in the relabelling mechanism. Float.
    changes_dict: dictionary containing the changes in the training set. Python dictionary.
    removals_dict: dictionary containing the removals in the training set. Python dictionary.
    record_dict: dictionary containing the last predictions of the instances. Python dictionary.
    not_change_dict: dictionary containing the instances that cannot be changed. Python dictionary.
    record_length: length of the record dictionary. Integer.
    not_change_epochs: number of epochs after a change during which there is not possible
        to change the label of that instance again nor remove it. Integer.
    quantile_loss: quantile to use for the loss threshold in the filtering mechanism. Float.
'''
def fit(model, train_dataset, optimizer, epochs, global_batch_size,
            quantile_prob, changes_dict, removals_dict,
            record_dict, not_change_dict, record_length,
            not_change_epochs, quantile_loss):

    epoch = 0

    threshold_mean_loss = 0
    previous_threshold_mean_loss = 0
    apply_thresholds = True
    overlap = 1
    previous_overlap = 1
    start_rafni = False
    previous_prob_threshold = 1
    prob_threshold = 1

    areas = []

    while epoch < epochs:
        print('Start of epoch %d' % (epoch,))
        losses_epoch = []
        prob_bad_epoch = []

        # Iterate over the batches of the dataset
        for step, (fn_batch_train, x_batch_train, y_batch_train) in enumerate(train_dataset):
            # Perform changes in batch
            ny_batch_train = perform_changes_in_batch(fn_batch_train,
                                                        y_batch_train,
                                                        changes_dict,
                                                        not_change_dict)

            # Perform removals in batch
            nfn_batch_train, nx_batch_train, ny_batch_train = perform_removals_in_batch(
                                                                fn_batch_train,
                                                                x_batch_train,
                                                                ny_batch_train,
                                                                removals_dict)

            new_batch_size = len(nfn_batch_train)
            # Define loss function without reduction
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits = True,
                                    reduction = tf.keras.losses.Reduction.NONE)

            # Open a GradientTape to record the operations run during the
            # forward pass, which enables autodifferentiation,
            with tf.GradientTape() as tape:
                # for selfie
                # l2_loss = tf.math.add_n([tf.nn.l2_loss(var) for var in model.trainable_variables])
                # Run the forwards pass of the layer.
                # The operations that the layer applies to its inputs are
                # going to be recorded on the GradientTape.
                logits = model(nx_batch_train, training = True)

                # Record predictions
                add_record(record_dict, nfn_batch_train, logits, record_length)

                # Compute the loss value for each sample
                losses = loss_fn(ny_batch_train, logits)

                # Get the losses of the batch inside the losses_epoch list
                losses_epoch.extend(losses)

                # Get the probabilities of the misclassified samples
                for idx in range(len(logits)):
                    ny = ny_batch_train.numpy()
                    if np.argmax(logits[idx]) != np.argmax(ny[idx]):
                        prob_bad_epoch.append(np.max(logits[idx]))

                ls_array = np.array(losses)

                # Use the loss of each sample:
                # 1. to restore the original class
                if start_rafni:
                    # Check if it is necessary to change the label of any instance
                    check_high_prob_wrong_label(nfn_batch_train, ny_batch_train,
                                                logits, changes_dict, prob_threshold,
                                                record_dict, removals_dict,
                                                not_change_dict, record_length)

                    # Check if the prediction of an instance has changed
                    # (record_length - 1) times or more. In that case, it adds
                    # them to the removal dictionary
                    check_record(nfn_batch_train, record_dict, removals_dict,
                                    record_length, changes_dict)
                # 2. to filter instances
                    for idx in range(len(losses)):
                        if (losses[idx] > threshold_mean_loss
                            and nfn_batch_train[idx].numpy() not in not_change_dict):
                            add_removal_in_dict(removals_dict, nfn_batch_train[idx].numpy())
                            # and remove it from the changes dictionary if it exists there
                            remove_from_dictionary(changes_dict,  nfn_batch_train[idx].numpy())

                # Reduce the loss over the minibatch
                if new_batch_size != 0:
                    loss_value = tf.reduce_sum(losses * (1. / new_batch_size))
                else:
                    loss_value = tf.reduce_sum(losses * (1. / global_batch_size))

            # Use the gradient tape to automatically retrieve the gradients
            # of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating the value
            # of the varaibles to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches
            if step % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' %
                        (step, float(loss_value)))

        # Update not_change_dict in order to not change an instance that
        # has been changed in the last 5 epochs
        update_not_change_dict(not_change_dict, not_change_epochs)

        losses_epoch = np.array(losses_epoch)

        # Modify loss and probability thresholds
        if apply_thresholds:

            # Probability threshold (if there are misclassified samples)
            if len(prob_bad_epoch) != 0:
                prob_bad_epoch = np.array(prob_bad_epoch)
                previous_prob_threshold = prob_threshold
                prob_threshold = np.quantile(prob_bad_epoch, quantile_prob)
                print('Prob threshold')
                print(prob_threshold)

            # Loss threshold
            previous_threshold_mean_loss = threshold_mean_loss
            gm = GaussianMixture(n_components=2, warm_start = True, tol = 0.1, reg_covar = 0.15).fit(losses_epoch.reshape(-1,1))
            noisy_distribution_idx = np.argmax(gm.means_)
            normal_distribution_idx = np.argmin(gm.means_)
            noisy_losses_mean = gm.means_[noisy_distribution_idx][0]
            noisy_losses_std = gm.covariances_[noisy_distribution_idx][0][0]
            threshold_mean_loss = np.quantile(losses_epoch, quantile_loss)

            normal_losses_mean = gm.means_[normal_distribution_idx][0]
            normal_losses_std = gm.covariances_[normal_distribution_idx][0][0]

            previous_overlap = overlap
            overlap = NormalDist(mu = normal_losses_mean, sigma = normal_losses_std).overlap(NormalDist(mu = noisy_losses_mean, sigma = noisy_losses_std))
            areas.append(overlap)
            if start_rafni == False and (overlap < 0.15 or (epoch !=0 and previous_overlap < overlap)):
                start_rafni = True
                print("Epoch threshold: " + str(epoch+1))

            if (start_rafni and ((noisy_losses_mean - normal_losses_mean <= 0.3))):
                apply_thresholds = False
                threshold_mean_loss = previous_threshold_mean_loss
                prob_threshold = previous_prob_threshold

        epoch = epoch + 1

    print(areas)
    return model, changes_dict, removals_dict
