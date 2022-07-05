import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Please read the free response questions before starting to code.
#
# Note: Avoid using nn.Sequential here, as it prevents the test code from
# correctly checking your model architecture and will cause your code to
# fail the tests.

class Digit_Classifier(nn.Module):
    """
    This is the class that creates a neural network for classifying handwritten digits
    from the MNIST dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """
    def __init__(self):
        super(Digit_Classifier, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64,10)

    def forward(self, inputs):
        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        output = self.output(inputs)
        return output
        


class Dog_Classifier_FC(nn.Module):
    """
    This is the class that creates a fully connected neural network for classifying dog breeds
    from the DogSet dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """

    def __init__(self):
        super(Dog_Classifier_FC, self).__init__()
        self.fc1 = nn.Linear(12288, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64,10)

    def forward(self, inputs):
        inputs = inputs.view(-1,64*64*3)
        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        output = self.output(inputs)

        return output




class Dog_Classifier_Conv(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying dog breeds
    from the DogSet dataset.
    Network architecture:
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 10)
    Activation function: ReLU for both hidden layers
    There should be a maxpool after each convolution.
    The sequence of operations looks like this:
        1. Apply convolutional layer with stride and kernel size specified
            - note: uses hard-coded in_channels and out_channels
            - read the problems to figure out what these should be!
        2. Apply the activation function (ReLU)
        3. Apply 2D max pooling with a kernel size of 2
    Inputs:
    kernel_size: list of length 2 containing kernel sizes for the two convolutional layers
                 e.g., kernel_size = [(3,3), (3,3)]
    stride: list of length 2 containing strides for the two convolutional layers
            e.g., stride = [(1,1), (1,1)]
    """

    def __init__(self, kernel_size, stride):
        super(Dog_Classifier_Conv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size[0], stride=stride[0])
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size[1], stride=stride[1])

        h1,w1 = self.calculate_conv_output_size(64, 64, kernel_size[0], stride[0])
        h1_p, w1_p = self.calculate_conv_output_size(h1, w1, (2,2), (2,2))
        h2,w2 = self.calculate_conv_output_size(h1_p,w1_p, kernel_size[1], stride[1])
        h2_p, w2_p = self.calculate_conv_output_size(h2, w2, (2,2), (2,2))

        self.output = nn.Linear(32*int(h2_p)*int(w2_p),10)

    def forward(self, inputs):
        inputs = inputs.permute((0,3,1,2))
        inputs = self.pool(F.relu(self.conv1(inputs)))
        inputs = self.pool(F.relu(self.conv2(inputs)))
        inputs = inputs.view(-1, 32*inputs.shape[2]*inputs.shape[3])
        outputs = self.output(inputs)

        return outputs

    def calculate_conv_output_size(self, height, width, kernel, stride):
        h_out = np.floor(((height - (kernel[0] - 1) - 1)/stride[0])+1)
        w_out = np.floor(((width - (kernel[1] - 1) - 1)/stride[1])+1)
        return h_out, w_out



