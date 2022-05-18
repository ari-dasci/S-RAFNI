import argparse
import os
from training import *

'''
Function that performs the cross-validation
Args:
    image_dir = Path to the directory when the train and test set are located. String.
    batch_size = Batch size. Integer.
    epochs = Total number of epochs to train the network. Integer.
    quantile_loss: Quantile to use for the loss threshold in the filtering mechanism. Float.
    quantile_prob: Quantile to use for the probability threshold in the relabelling mechanism. Float.
    record_length: Length of the record dictionary. Integer.
    not_change_epochs: Number of epochs after a change during which there is not possible
        to change the label of that instance again nor remove it. Integer.
    fine_tune: Whether to fine-tune the backbone network. Boolean.
    save_names: Output file. String.
    folds: Total number of folds for the cross-validation.
    backbone_network: Which backbone network to use. ResNet or EfficientNet. String.
'''
def noise5fcv(image_dir, batch_size, epochs, quantile_loss,
                quantile_prob, record_length, not_change_epochs,
                fine_tune, save_names, folds, backbone_network = 'ResNet'):
    accuracies = []

    for fold in range(folds):
        partition_dir = os.path.join(image_dir, 'partition' + str(fold))
        with open(save_names, 'a') as f:
            f.write('FOLD ' + str(fold) + '\n')

        acc = train(partition_dir, batch_size, epochs, quantile_prob,
                    record_length, not_change_epochs, fine_tune,
                    save_names, quantile_loss, fold)

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
        '--quantile_loss',
        type = float,
        default = 0.95,
        help = 'Quantile value for the loss. An instance will be considered removable if its loss exceed this value.'
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
    ARGS, unparsed = parser.parse_known_args()

    m_acc, accuracies = noise5fcv(ARGS.image_dir, ARGS.batch_size, ARGS.epochs,
                        ARGS.quantile_loss,
                        ARGS.quantile_prob, ARGS.record_length,
                        ARGS.not_change_epochs, ARGS.fine_tune,
                        ARGS.save_names, ARGS.folds, ARGS.backbone_network)
