import torch
import torch.nn as nn
import numpy as np
from data.my_dataset import MyDataset
from data.load_data import load_mnist_data
from src.run_model import run_model
from src.models import Digit_Classifier, Dog_Classifier_FC, Dog_Classifier_Conv
import time
import matplotlib.pyplot as plt
from data.dogs import DogsDataset


# #1
def training_on_mnist():
    train_size = [500, 1000, 1500, 2000]
    test_size = 1000
    time_arr = []
    test_acc_arr = []

    test_examples_per_class = 100
    _, test_features, _, test_targets = load_mnist_data(10, 0.0, examples_per_class=test_examples_per_class, mnist_folder='data')
    test_dataset = MyDataset(test_features, test_targets)

    #training
    for i in range(0, len(train_size)):
        start = time.time()
        examples_per_class = int(train_size[i]/10)
        train_features, _, train_targets, _ = load_mnist_data(10, 1.0, examples_per_class=examples_per_class, mnist_folder='data')

        train_dataset = MyDataset(train_features, train_targets)

        model = Digit_Classifier()

        final_model, epoch_loss, epoch_acc = run_model(model, running_mode='train', train_set=train_dataset, batch_size=10, learning_rate=0.01, n_epochs=100, shuffle=True)
        duration = time.time() - start
        time_arr.append(duration)

        #testing
        test_loss, test_acc = run_model(model, running_mode='test', test_set=test_dataset, batch_size=10, n_epochs=10, shuffle=True)
        test_acc_arr.append(test_acc)



    plt.plot(train_size, time_arr)
    plt.ylabel('training time (s)')
    plt.xlabel('number of training samples')
    plt.title('training time vs. training samples')
    plt.savefig('time_and_samples')

    plt.clf()

    plt.plot(train_size, test_acc_arr)
    plt.ylabel('testing accuracy %')
    plt.xlabel('number of training samples')
    plt.title('testing accuracy vs. training samples')
    plt.savefig('accuracy_and_samples')




def problem_5():
    dogSet = DogsDataset(path_to_dogsset='data')
    train_features, train_labels = dogSet.get_train_examples()
    valid_features, valid_labels = dogSet.get_validation_examples()
    test_features, test_labels = dogSet.get_test_examples()
    print("training size: {}".format(train_features.shape))
    print("validation size: {}".format(valid_features.shape))
    print("test size: {}".format(test_features.shape))

# train_class, train_count = np.unique(train_labels, return_counts=True)
# print(len(train_class))

# valid_class, valid_count = np.unique(valid_labels, return_counts=True)
# print(len(valid_class))

# test_class, test_count = np.unique(test_labels, return_counts=True)
# print(len(test_class))

# 8.png 9.png 11.png



def problem_7(model):
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            p = parameter.numel()
            print("layer name: {}, #param: {}".format(name, p))


def problem_8():
    dogSet = DogsDataset(path_to_dogsset='data')
    train_features, train_labels = dogSet.get_train_examples()
    valid_features, valid_labels = dogSet.get_validation_examples()
    test_features, test_labels = dogSet.get_test_examples()
    print("training size: {}".format(train_features.shape))
    print("validation size: {}".format(valid_features.shape))
    print("test size: {}".format(test_features.shape))


    train_dataset = MyDataset(train_features, train_labels)
    valid_dataset = MyDataset(valid_features, valid_labels)
    model = Dog_Classifier_FC()
    final_model, epoch_loss, epoch_acc = run_model(model, running_mode='train', train_set=train_dataset, valid_set=valid_dataset, batch_size=10,learning_rate=1e-5,  n_epochs=100, shuffle=True)

    train_loss = epoch_loss['train']
    valid_loss = epoch_loss['valid']

    train_acc = epoch_acc['train']
    valid_acc = epoch_acc['valid']




    plt.plot(np.linspace(1,len(train_loss),len(train_loss)), train_loss, label='training loss')
    plt.plot(np.linspace(1,len(valid_loss),len(valid_loss)), valid_loss, label='validation loss')
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig('dogFCloss')


    plt.clf()
    plt.plot(np.linspace(1,len(train_acc),len(train_acc)), train_acc, label='training accuracy')
    plt.plot(np.linspace(1,len(valid_acc),len(valid_acc)), valid_acc, label='validation accuracy')
    plt.xlabel('epoch number')
    plt.ylabel('accuracy %')
    plt.legend(loc='upper left')
    plt.savefig('dogFCacc')


    print(len(train_loss))
    problem_7(final_model)

    test_dataset = MyDataset(test_features, test_labels)
    test_loss, test_acc = run_model(final_model, running_mode='test', test_set=test_dataset, batch_size=10, n_epochs=1, shuffle=True)
    print(test_acc)


def calculate_conv_output_size(height, width, kernel, stride, padding, dilation):
    h_out = np.floor(((height + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1)/stride[0])+1)
    w_out = np.floor(((width + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1)/stride[1])+1)
    return h_out, w_out


def problem_12():
    dogSet = DogsDataset(path_to_dogsset='data')
    train_features, train_labels = dogSet.get_train_examples()
    valid_features, valid_labels = dogSet.get_validation_examples()
    test_features, test_labels = dogSet.get_test_examples()
    print("training size: {}".format(train_features.shape))
    print("validation size: {}".format(valid_features.shape))
    print("test size: {}".format(test_features.shape))


    train_dataset = MyDataset(train_features, train_labels)
    valid_dataset = MyDataset(valid_features, valid_labels)
    model = Dog_Classifier_Conv([(3,3), (3,3)], [(1,1), (1,1)])
    final_model, epoch_loss, epoch_acc = run_model(model, running_mode='train', train_set=train_dataset, valid_set=valid_dataset, batch_size=10,learning_rate=1e-5,  n_epochs=100, shuffle=True)

    train_loss = epoch_loss['train']
    valid_loss = epoch_loss['valid']

    train_acc = epoch_acc['train']
    valid_acc = epoch_acc['valid']




    plt.plot(np.linspace(1,len(train_loss),len(train_loss)), train_loss, label='training loss')
    plt.plot(np.linspace(1,len(valid_loss),len(valid_loss)), valid_loss, label='validation loss')
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig('dogConvloss')


    plt.clf()
    plt.plot(np.linspace(1,len(train_acc),len(train_acc)), train_acc, label='training accuracy')
    plt.plot(np.linspace(1,len(valid_acc),len(valid_acc)), valid_acc, label='validation accuracy')
    plt.xlabel('epoch number')
    plt.ylabel('accuracy %')
    plt.legend(loc='upper left')
    plt.savefig('dogConvacc')


    print(len(train_loss))
    problem_7(final_model)


    test_dataset = MyDataset(test_features, test_labels)
    test_loss, test_acc = run_model(final_model, running_mode='test', test_set=test_dataset, batch_size=10, n_epochs=1, shuffle=True)
    print(test_acc)



if __name__ == '__main__':
    problem_12()