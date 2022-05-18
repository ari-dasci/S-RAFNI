import tensorflow as tf
import tensorflow.keras.backend as K
from fit import *
from load_data import *
from dictionaries import *
import os
import argparse
from resnet32preact import *
from scheduler import *
from densenet import *
from CNND2L import *

'''
Function to train the network.
Args:
    image_dir = Path to the directory when the train and test set are located. String.
    batch_size = Batch size. Integer.
    epochs = Total number of epochs to train the network. Integer.
    quantile_prob: Quantile to use for the probability threshold in the relabelling mechanism. Float.
    record_length: Length of the record dictionary. Integer.
    not_change_epochs: Number of epochs after a change during which there is not possible
        to change the label of that instance again nor remove it. Integer.
    fine_tune: Whether to fine-tune the backbone network. Boolean.
    save_names: Output file. String.
    quantile_loss: Quantile to use for the loss threshold in the filtering mechanism. Float.
    fold: Number of fold in the cross-validation. Integer.
    backbone_network: Which backbone network to use. ResNet or EfficientNet. String.
'''
def train(image_dir, batch_size, epochs,
            quantile_prob, record_length, not_change_epochs,
            fine_tune, save_names, quantile_loss, fold,
            backbone_network = 'ResNet', n_type = None, n_rate = 20):

    if image_dir == 'cifar10' or image_dir == 'cifar100':
        labeled_train_ds, labeled_test_ds = load_train_test_cifar('data/' + image_dir,
                                                batch_size, noise = n_type, rate = n_rate, data_aug = True, model = backbone_network)
        if image_dir == 'cifar10':
            num_classes = 10
        else:
            num_classes = 100
    else:
        labeled_train_ds, labeled_test_ds = load_train_test(image_dir + '/train',
                                                            image_dir + '/test',
                                                            batch_size, 2000)
        num_classes = len(os.listdir(image_dir + '/train'))

    changes_dict = dict()
    removals_dict = dict()
    record_dict = dict()
    not_change_dict = dict()

    if backbone_network == 'ResNet':
        resnet50 = tf.keras.applications.ResNet50(include_top = False,
                                                    weights = 'imagenet',
                                                    pooling = 'avg')
    elif backbone_network == 'EfficientNet':
        efficientnetB0 = tf.keras.applications.EfficientNetB0(include_top = False,
                                                    weights = 'imagenet',
                                                    pooling = 'avg')
    elif backbone_network == 'ResNet32':
        model = resnet_preact(5, 0.0001, 10)
    elif backbone_network == 'ResNet44':
        model = resnet_preact(7, 0.001, 100)
    elif backbone_network == 'DenseNet':
        model = densenet(25, 12, num_classes)
    elif backbone_network == 'D2LC10':
        model = eight_layer(num_classes)

    if not fine_tune:
        # Freeze the base model
        if backbone_network == 'ResNet':
            resnet50.trainable = False
        elif backbone_network == 'EfficientNet':
            efficientnetB0.trainable = False

    if backbone_network == 'ResNet' or backbone_network == 'EfficientNet':
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
    elif backbone_network == 'ResNet32':
        opt = tf.keras.optimizers.SGD(learning_rate = CustomScheduleC10(0.1), decay = 0.0,
                                        momentum = 0.9)
    elif backbone_network == 'ResNet44':
        opt = tf.keras.optimizers.SGD(learning_rate = CustomScheduleC100(0.1), decay = 5e-3, #it was 0.0 for patrini
                                        momentum = 0.9)
    elif backbone_network == 'D2LC10':
        opt = tf.keras.optimizers.SGD(learning_rate = CustomScheduleC10(0.1), decay = 1e-4,
                                        momentum = 0.9)
    elif backbone_network == 'DenseNet':
        boundaries = [int(np.ceil(50000/128)*50), int(np.ceil(50000/128)*75)]
        values = [0.1, 0.02, 0.004]
        opt = tf.keras.optimizers.SGD(learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                                        boundaries, values), decay = 0.0,
                                        momentum = 0.9, nesterov = True)

    model, changes_dict, removals_dict = fit(model, labeled_train_ds,
                                                opt, epochs, batch_size,
                                                quantile_prob, changes_dict,
                                                removals_dict, record_dict,
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
    elif backbone_network == 'EfficientNet':
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
        '--quantile_loss',
        type = float,
        default = 0.95,
        help = 'Quantile value for the loss threshold. An instance will be considered removable if its loss exceed this threshold.'
    )
    parser.add_argument(
        '--quantile_prob',
        type = float,
        default = 0.99,
        help = 'Quantile value for the probability threshold. An instance will change its class if the network predicts another one with a probability that excedd this threshold.'
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
        '--n_type',
        type = str,
        default = None,
        help = "Which noise type to use. RA for symmetric or NA for asymmetric."
    )
    parser.add_argument(
        '--n_rate',
        type = int,
        default = 20,
        help = "Which noise rate to use."
    )
    ARGS, unparsed = parser.parse_known_args()

    acc = train(ARGS.image_dir, ARGS.batch_size, ARGS.epochs,
                ARGS.quantile_prob, ARGS.record_length,
                ARGS.not_change_epochs, ARGS.fine_tune,
                ARGS.save_names, ARGS.quantile_loss, 0, ARGS.backbone_network,
                ARGS.n_type, ARGS.n_rate)
