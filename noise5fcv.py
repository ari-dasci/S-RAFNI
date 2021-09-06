import argparse
import os
from training import *

'''
Function that performs the cross-validation
Args:
    image_dir = Path to the directory where the train and test set are located. String.
    batch_size = Batch size. Integer.
    epochs = Total number of epochs to train the network. Integer.
    epoch_threshold: Epoch threshold so that there is no change in the training set before
        that threshold and there is no removal before 1.5*epoch_threshold. Integer.
    quantile_loss: Quantile to use for the loss threshold in the filtering mechanism. Float.
    prob_threshold: Probability threshold to use in the relabelling mechanism. Float.
    record_length: Length of the record dictionary. Integer.
    not_change_epochs: Number of epochs after a change during which there is not possible
        to change the label of that instance again nor remove it. Integer.
    fine_tune: Whether to fine-tune the backbone network. Boolean.
    save_names: Output file. String.
    folds: Total number of folds for the cross-validation.
    backbone_network: Which backbone network to use. ResNet or EfficientNet. String.
    data_set: Which data set to use. 'cifar10', 'cifar100' or 'other'. String.
    noise: Noise type to use when using cifar datasets. 'RA' for symmetric and 'AN' for asymmetric. String.
    rate: Noise rate to use when using cifar dataests. Integer.
'''
def noise5fcv(image_dir, batch_size, epochs, epoch_threshold, quantile_loss,
                prob_threshold, record_length, not_change_epochs,
                fine_tune, save_names, folds, backbone_network = 'ResNet',
                data_set = 'other', noise = 'RA', rate = 20):
    accuracies = []

    for fold in range(folds):
        partition_dir = os.path.join(image_dir, 'partition' + str(fold))
        with open(save_names, 'a') as f:
            f.write('FOLD ' + str(fold) + '\n')

        acc = train(partition_dir, batch_size, epochs, epoch_threshold,
                    prob_threshold, record_length, not_change_epochs, fine_tune,
                    save_names, quantile_loss, fold, data_set, noise, rate)

        accuracies.append(acc)

    accuracies = np.array(accuracies)
    m_acc = np.mean(accuracies)

    with open(save_names, 'a') as f:
        f.write('Mean test accuracy: ' + str(m_acc) + '\n')

    print('Mean test accuracy: ' + str(m_acc))
    return m_acc, accuracies

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type = str,
        default = '',
        help = "Path to folders of labeled images."
    )
    parser.add_argument(
        '--folds',
        type = int,
        default = 5,
        help = 'Number of folds in cross validation'
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
        default = 40,
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
        type = str,
        default = 'ResNet',
        help = "Noise rate when using cifar datasets."
    )
    ARGS, unparsed = parser.parse_known_args()

    m_acc, accuracies = noise5fcv(ARGS.image_dir, ARGS.batch_size, ARGS.epochs,
                        ARGS.epoch_threshold, ARGS.quantile_loss,
                        ARGS.prob_threshold, ARGS.record_length,
                        ARGS.not_change_epochs, ARGS.fine_tune,
                        ARGS.save_names, ARGS.folds, ARGS.backbone_network,
                        ARGS.data_set, ARGS.noise, ARGS.rate)
