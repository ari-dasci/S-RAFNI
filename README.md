# A robust approach for deep neural networks in presence of label noise: relabelling and filtering instances during training. The RAFNI (Relabelling And Filtering Noisy Instances) algorithm.

Official TensorFlow implementation of RAFNI.

## 1. Publication
For more information, please read our publication.
If you use this implementation, please cite our publication.

## 2. Requirements
* TensorFlow >= 2.4
* Python >= 3.6
* Numpy >= 1.19.2
* Sklearn >= 0.24.1

## 4. How to use it

### 1. Data sets
This algorithm can be used with the same data sets that we used or with any other data set. To use the same CIFAR10 and CIFAR100 data sets that we used, first download them from  and place them in the same folder as the code. For any other data set, it is necessary that they are in a folder with two sulfolders, train and test, that each contain one subfolder per class.

### 2. Hyperparameters
The algorith has the following hyperparameters:
```
--image_dir = Path to the directory where the train and test set are located (excep cifar10 and cifar100). String
--batch_size = Batch size. Integer.
--epochs = Total number of epochs to train the network. Integer.
--epoch_threshold: Epoch threshold so that there is no change in the training set before that threshold and there is no removal before 1.5*epoch_threshold. Integer.
--prob_threshold: Probability threshold to use in the relabelling mechanism. Float.
--record_length: Length of the record dictionary. Integer.
--not_change_epochs: Number of epochs after a change during which there is not possible to change the label of that instance again nor remove it. Integer.
--fine_tune: Whether to fine-tune the backbone network. Boolean.
--save_names: Output file. String.
--quantile_loss: Quantile to use for the loss threshold in the filtering mechanism. Float.
--backbone_network: Which backbone network to use. ResNet or EfficientNet. String.
--data_set: Which data set to use. 'cifar10', 'cifar100' or 'other'. String.
--noise: Noise type to use when using cifar datasets. 'RA' for symmetric and 'AN' for asymmetric. String.
--rate: Noise rate to use when using cifar dataests. Integer.
```

### 3. Run
To run our algorithm for a hold-out use something similar to:
```
python training.py --save_names=output.txt --epoch_threshold=5 --quantile_loss=0.96 --prob_threshold=0.6 --record_length=4 --not_change_epochs=3 --backbone_network=ResNet --data_set=cifar10 --noise=RA --rate=20 --epochs=10 --batch_size=128 --fine_tune
```

To run a five-fold cross-validation, use the `noise5fcv.py` file.
