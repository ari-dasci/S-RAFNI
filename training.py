import tensorflow as tf
import tensorflow.keras.backend as K
from fit import *
from load_data import *
from dictionaries import *
import os
import argparse

'''
Function to train the network.
Args:
    image_dir = Path to the directory where the train and test set are located. String.
    batch_size = Batch size. Integer.
    epochs = Total number of epochs to train the network. Integer.
    epoch_threshold: Epoch threshold so that there is no change in the training set before
        that threshold and there is no removal before 1.5*epoch_threshold. Integer.
    prob_threshold: Probability threshold to use in the relabelling mechanism. Float.
    record_length: Length of the record dictionary. Integer.
    not_change_epochs: Number of epochs after a change during which there is not possible
        to change the label of that instance again nor remove it. Integer.
    fine_tune: Whether to fine-tune the backbone network. Boolean.
    save_names: Output file. String.
    quantile_loss: Quantile to use for the loss threshold in the filtering mechanism. Float.
    fold: Number of fold in the cross-validation (if necessary). Integer.
    backbone_network: Which backbone network to use. ResNet or EfficientNet. String.
    data_set: Which data set to use. 'cifar10', 'cifar100' or 'other'. String.
    noise: Noise type to use when using cifar datasets. 'RA' for symmetric and 'AN' for asymmetric. String.
    rate: Noise rate to use when using cifar dataests. Integer.
'''
def train(image_dir, batch_size, epochs, epoch_threshold,
            prob_threshold, record_length, not_change_epochs,
            fine_tune, save_names, quantile_loss, fold = 0,
            backbone_network = 'ResNet', data_set = 'other',
            noise = 'RA', rate = 20):

    if data_set == 'other':
        labeled_train_ds, labeled_test_ds = load_train_test(image_dir + '/train',
                                                            image_dir + '/test',
                                                            batch_size, 2000)
        num_classes = len(os.listdir(image_dir + '/train'))
    elif data_set == 'cifar10':
        labeled_train_ds, labeled_test_ds = load_train_test_cifar('cifar10', batch_size, noise = noise, rate = rate)
        num_classes = 10
    else:
        labeled_train_ds, labeled_test_ds = load_train_test_cifar('cifar100', batch_size, noise = noise, rate = rate)
        num_classes = 100

    changes_dict = dict()
    removals_dict = dict()
    record_dict = dict()
    not_change_dict = dict()

    if backbone_network == 'ResNet':
        resnet50 = tf.keras.applications.ResNet50(include_top = False,
                                                    weights = 'imagenet',
                                                    pooling = 'avg')
    else:
        efficientnetB0 = tf.keras.applications.EfficientNetB0(include_top = False,
                                                    weights = 'imagenet',
                                                    pooling = 'avg')

    if not fine_tune:
        # Freeze the base model
        if backbone_network == 'ResNet':
            resnet50.trainable = False
        else:
            efficientnetB0.trainable = False

    inputs = tf.keras.Input(shape = (None, None, 3))
    x = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, (224, 224)))(inputs)
    if backbone_network == 'ResNet':
        x = tf.keras.applications.resnet.preprocess_input(x)
    else:
        x = tf.keras.applications.efficientnet.preprocess_input(x)

    if not fine_tune:
        if backbone_network == 'ResNet':
            x = resnet50(x, training = False)
        else:
            x = efficientnetB0(x, training = False)
    else:
        if backbone_network == 'ResNet':
            x = resnet50(x)
        else:
            x = efficientnetB0(x)

    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    predictions = tf.keras.layers.Dense(num_classes, activation = 'softmax')(x)

    model = tf.keras.Model(inputs = inputs, outputs = predictions)

    opt = tf.keras.optimizers.SGD(lr = 1e-3, decay = 1e-6, momentum = 0.9,
                                    nesterov = True)

    model, changes_dict, removals_dict = fit(model, labeled_train_ds,
                                                opt, epochs, batch_size,
                                                prob_threshold,
                                                changes_dict, removals_dict,
                                                epoch_threshold, record_dict,
                                                not_change_dict, record_length,
                                                not_change_epochs,
                                                quantile_loss)

    # Save removals and changes in save_names file
    with open(save_names, 'a') as f:
        f.write('Removals dictionary: ')
        f.write(str(len(removals_dict)) + '\n')
        for key, value in removals_dict.items():
            f.write(str(key) + ': ')
            f.write('\n')
        f.write('Changes dictionary: ')
        f.write(str(len(changes_dict)) + '\n')
        for key, value in changes_dict.items():
            f.write(str(key) + ': ')
            f.write(str(np.argmax(value)))
            f.write('\n')

    labeled_test_ds = labeled_test_ds.batch(1)

    test_accuracy = tf.keras.metrics.CategoricalAccuracy()

    for (fn, x, y) in labeled_test_ds:
        logits = model(x, training = False)
        test_accuracy.update_state(y, logits)

    acc = test_accuracy.result().numpy()
    with open(save_names, 'a') as f:
        f.write('Test accuracy')
        f.write(str(acc))
    print('Test set accuracy' + str(acc))
    print('Number of removed images: ' + str(len(removals_dict)))
    print('Number of changed labels: ' + str(len(changes_dict)))

    changes_dict.clear()
    removals_dict.clear()
    record_dict.clear()
    not_change_dict.clear()

    del model
    if backbone_network == 'ResNet':
        del resnet50
    else:
        del efficientnetB0
    K.clear_session()

    return acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type = str,
        default = '',
        help = "Path to folders of labeled images."
    )
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 16,
        help = "Batch size"
    )
    parser.add_argument(
        '--epochs',
        type = int,
        default = 50,
        help = "Number of epochs to train"
    )
    parser.add_argument(
        '--epoch_threshold',
        type = int,
        default = 5,
        help = 'Number of epochs at the beginning of the training where none of the noise filters/changes are used'
    )
    parser.add_argument(
        '--quantile_loss',
        type = float,
        default = 0.95,
        help = 'Quantile value for the loss. An instance will be considered removable if its loss exceed this value.'
    )
    parser.add_argument(
        '--prob_threshold',
        type = float,
        default = 0.99,
        help = 'Minimum probability value for an instance to have to consider a change in its class.'
    )
    parser.add_argument(
        '--record_length',
        type = int,
        default = 5,
        help = 'Length of the record dictionary.'
    )
    parser.add_argument(
        '--not_change_epochs',
        type = int,
        default = 5,
        help = 'Number of epochs during which the class of an instance can not be changed after a change.'
    )
    parser.add_argument(
        '--fine_tune',
        default = False,
        help = "Whether to fine tune the whole network or not",
        action = 'store_true'
    )
    parser.add_argument(
        '--save_names',
        type = str,
        default = 'save_names.txt',
        help = "Name of the file in which to store the names of the images removed and the names and new class of the images which have changed class."
    )
    parser.add_argument(
        '--backbone_network',
        type = str,
        default = 'ResNet',
        help = "Which backbone network to use. ResNet or EfficientNet."
    )
    parser.add_argument(
        '--data_set',
        type = str,
        default = 'other',
        help = "Which data set to use. 'cifar10', 'cifar100' or 'other' to read any other data set from a folder."
    )
    parser.add_argument(
        '--noise',
        type = str,
        default = 'RA',
        help = "Noise type when using cifar datasets. 'RA' for symmetric and 'AN' for asymmetric."
    )
    parser.add_argument(
        '--rate',
        type = int,
        default = 20,
        help = "Noise rate when using cifar datasets."
    )
    ARGS, unparsed = parser.parse_known_args()

    acc = train(ARGS.image_dir, ARGS.batch_size, ARGS.epochs, ARGS.epoch_threshold,
                ARGS.prob_threshold, ARGS.record_length,
                ARGS.not_change_epochs, ARGS.fine_tune,
                ARGS.save_names, ARGS.quantile_loss, 0, ARGS.backbone_network,
                ARGS.data_set, ARGS.noise, ARGS.rate)
