# Handwritten number representation with a twist: sum of number predicted with a random number between 0-9

We have built a model based on the classic MNIST data which is trained to not only identify the number in the images, but also add a random number between 0 to 9 to this predicted number.

## Primary Libraries


`import torch` : importing torch library

`import torch.nn as nn` : torch.nn provides basic building blocks for graphs, as per pytorch documentation

`import torch.nn.functional as F` : importing all the functions in torch.nn library

`import torch.optim as optim` : contains various optimization algorithms

`from torchvision import datasets, transforms` : contains popular datasets and commom image transformations for computer vision



## Data representation, generation and usage

We have two data groups as input. First is the standard MNIST data set, with train and test as well as target parameters along with the input features. The second set consists of random numbers generated between 0 and 9 and a target value, which is the sum of random number and target provided by the MNIST dataset. Number of additional inputs is same as the MNIST set, 60,000 for test and 10,000 for train.

We have created additional train and test data loaders for the random numbers namely `random_train_loader` and `random_test_loader`, in addition to regular train and test data loaders. Both the data loaders are passed to the functions for train as well as test where the two sets of input features are passed as inputs to the network.

MNIST data set is passed as input to the convolution layer. The output received after processing the image inputs through various convolutional layers (1x10), is concatinated with the one hot encoded random number input and is passed to a fully connected layer. The final output is a tuple of predicted digit, provided by the process of convolution, and sum of predicted digit and random number provided by the Fully connected layer.

## Evaluation

We are using R2 as our metric for measuring the performance of prediction of sum of the two digits, as we are treating this as a regression problem.

Loss function used for digit prediction is Negative Log Loss and for Summed vaue is MSE. As we are using logSoftmax for predicting the digit in MNIST, we have used NLL for measuring the loss. MSE is a good measure for comparing two continuous values, and hence used.

Total loss is observers as addition of both the losses

## Output

<img width="1364" alt="Screenshot 2021-05-22 at 02 45 05" src="https://user-images.githubusercontent.com/31658286/119199069-0ec51080-baa8-11eb-9416-f8da3da964a8.png">

Complete training and testing log available on notebook.

MNIST Test Accuracy Achieved: `92%`

R2 score over 5 epochs: `0.065`

## Team Members

[Abhiram Gurijala]()

[Arijit Ganguly](https://github.com/Arijit-datascience)

[Rohin Sequeira](https://github.com/RohinSequeira)
