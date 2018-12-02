import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import utils1 as ut1
import helper as my_help_functions

import sys


#let's define our parameters:
OUTPUT_LAYER_DIM = len(my_help_functions.Dictionary_of_classes)
learning_rate = 0.01
window_size = 5 #2 neighbors + target word
DIM_HIDDEN_LAYER = 100
tags_number = 10
DIM_INPUT = 250 #window x vector_size = 5 x 50 =250
NUM_OF_EPOCHS = 3
VECTOR_EMBEDDINGS_DIM = 50

NUMBER_OF_UNIQUE_WORDS = len(my_help_functions.Dictionary_of_words)
batch_normalization_size = 1024 #denotes the number of samples contained in each generated batch.

def loading_data(train_or_dev_set):
    windows_array, tags = my_help_functions.bring_train_data(train_or_dev_set)
    # Convert a list into a numpy array and set data-type to float32
    windows_array = np.asarray(windows_array, np.float32)

    # Convert a list into a numpy array and set data-type to int32
    tags = np.asarray(tags, np.int32)

    """
    Creates a Tensor from a numpy.ndarray
    The returned tensor and ndarray share the same memory. Modifications to the tensor will
     be reflected in the ndarray and vice versa. The returned tensor is not resizable.
    """
    tags = torch.from_numpy(tags)
    windows_array = torch.from_numpy(windows_array)

    # make sure the words and tag have same size, in order to pass it through TensorDataset
    tags = tags.type(torch.LongTensor)
    windows_array = windows_array.type(torch.LongTensor)

    data = TensorDataset(windows_array, tags)
    return data


def load_training_set(train_file):
    """
    this function create a dataset for the train

    :param train_file:
    :return:train data
    """
    data_train = loading_data(train_file)

    """
    shuffle is set to True, we will get a new order of exploration at each pass.
    Shuffling the order in which examples are fed to the classifier is helpful so
     that batches between epochs do not look alike. Doing so will eventually make our model more robust.
    """
    return DataLoader(shuffle=True,dataset=data_train,batch_size=batch_normalization_size)


def load_dev_set(dev_file):
    """
    this function create a dataset for the train

    :param dev_file:
    :return:train data
    """
    windows_array, tags = my_help_functions.bring_dev_data(dev_file)
    # Convert a list into a numpy array and set data-type to float32
    windows_array = np.asarray(windows_array, np.float32)

    # Convert a list into a numpy array and set data-type to int32
    tags = np.asarray(tags, np.int32)

    """
    Creates a Tensor from a numpy.ndarray
    The returned tensor and ndarray share the same memory. Modifications to the tensor will
     be reflected in the ndarray and vice versa. The returned tensor is not resizable.
    """
    tags = torch.from_numpy(tags)
    windows_array = torch.from_numpy(windows_array)

    # make sure the words and tag have same size, in order to pass it through TensorDataset
    tags = tags.type(torch.LongTensor)
    windows_array = windows_array.type(torch.LongTensor)

    data = TensorDataset(windows_array, tags)

    """
    shuffle is set to True, we will get a new order of exploration at each pass.
    Shuffling the order in which examples are fed to the classifier is helpful so
     that batches between epochs do not look alike. Doing so will eventually make our model more robust.
    """
    return DataLoader(batch_size=1,shuffle=True, dataset=data)



def main(argv):
    """

    :param argument:
    :return: none
    """
    #ner or pos (user input)
    folder_name_input = argv[0]

    if folder_name_input== 'ner':
        learning_rate = 0.05
    path_train = folder_name_input + '/train'
    train_set = load_training_set(path_train)
    print(len(my_help_functions.Dictionary_of_words))
    path_dev = folder_name_input+'/dev'
    dev_set = load_dev_set(path_dev)

    model = NN()
    train_neural_network(model,train=train_set,validation_set=dev_set,mapping_type=folder_name_input)


###############################################################
#Function Name: get_y_tag
#Function input: model
#Function output:return the model prediction tag
#Function Action: the function return the prediction
#by getting  the index of the max log-probability
################################################################

def get_y_tag(model):
    return model.data.max(1, keepdim=True)[1]

###############################################################
#Function Name:print_message_each_epoch
#Function input:kind_of_set,length of set, loss of model, number
#of correct predictions of model and size of batch
#Function output:none
#Function Action:the function print a message to help the user
#follow the network progress
################################################################


def print_message_each_epoch(kind_of_set,set_len,m_loss,m_success,size_of_batch):
    print('\n' + kind_of_set + ': The average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        m_loss, m_success, (set_len * size_of_batch),
        100. * m_success / (set_len * size_of_batch)))

###############################################################
#Function Name: calculate_loss_print
#Function input: size_of_batch,model, set,is_training-boolean varible
#indicates id the set is trainning set or validation set
#Function output:loss
#Function Action:the function calculate the loss of the model
#for each epochs,and print write message
################################################################


def calculate_loss_print(size_of_batch,model,set,is_training,mapping_type):

    #boolean varible indicates id the set is trainning set or validation set
    print_kind_of_set="training set"
    if is_training != 1:
        print_kind_of_set = "validation set"
    #define varibles for loss, and number of correct prediction

    m_loss=0
    m_count = 0
    m_success=0
    #let the model know to switch to eval mode by calling .eval() on the model
    model.eval()
    for data, tag in set:
        #feed model with data
        model_result = model(data)
        #sum the loss
        m_loss = m_loss + F.nll_loss(model_result,tag,size_average=False).item()
        #call help function to get the right prediction
        y_tag = get_y_tag(model_result)
        if mapping_type == 'ner' and is_training!=1 :
            if my_help_functions.Representation_Of_Indexes_By_classes[y_tag.cpu().sum().item()] != 'O' or my_help_functions.Representation_Of_Indexes_By_classes[tag.cpu().sum().item()] != 'O':
                # total of successfull predictions
                m_success += y_tag.eq(tag.data.view_as(y_tag)).cpu().sum().item()
                m_count = m_count+1
        else:
            m_count = m_count + 1
            # total of successfull predictions
            m_success += y_tag.eq(tag.data.view_as(y_tag)).cpu().sum().item()


    # save the len of training set in varible to save calls to len functions
    set_len = len(set)
    #calculate loss
    m_loss = m_loss/(set_len)
    m_accuracy = 100. * m_success / m_count
    #call help function to print message about loss each epoch
    print_message_each_epoch(print_kind_of_set,set_len,m_loss,m_success,size_of_batch)
    return m_accuracy, m_loss


###############################################################
#Function Name:train_neural_network
#Function input:model, trainng and validation sets
#Function output:none
#Function Action:train on the training set and then test the
#network on the test set. This has the network make predictions on data it has never seen
################################################################

def train_neural_network(model,train,validation_set,mapping_type):
    #define 2 empty list to sve there the loss


    validation_set_scores = {}

    dev_accuracy_per_iter = {}


    train_set_scores={}
    #set the optimizer
    optimizer=optim.Adagrad(model.parameters(),lr=learning_rate)
    for epoch in range(NUM_OF_EPOCHS):
        print "epoch number "+ str(epoch)
        model.train()

        for data, labels in train:
            optimizer.zero_grad()
            output = model(data)
            running_los = F.nll_loss(output,labels)
            running_los.backward()
            optimizer.step()
        #calculate the loss by calling calculate_loss_print function

        train_accuracy,running_los = calculate_loss_print(batch_normalization_size, model, train,1,mapping_type)
        train_set_scores[epoch+1]=running_los

        # calculate the loss by calling calculate_loss_print function
        accuracy_dev, loss_dev = calculate_loss_print(1,model,validation_set,0,mapping_type)
        validation_set_scores[epoch+1]=loss_dev

        dev_accuracy_per_iter[epoch+1] = accuracy_dev
    first_lable, = plt.plot(validation_set_scores.keys(), validation_set_scores.values(), "g-", label='validation loss')
    second_lable, = plt.plot(dev_accuracy_per_iter.keys(), dev_accuracy_per_iter.values(), "r-", label='validation accuracy')
    plt.legend(handler_map={first_lable: HandlerLine2D(numpoints=4)})
    plt.show()


class NN(nn.Module):
    """
    Model A: Neural Network with one hidden layer, the first layer should have a size of 250 since each
    embedding vector contains 50 neurons and each window has 5 word
    should be followed by tanh activation function
    """

    def __init__(self):
        super(NN, self).__init__()

        self.E = nn.Embedding(len(my_help_functions.Dictionary_of_words), VECTOR_EMBEDDINGS_DIM)
        self.input_dim = DIM_INPUT
        self.fc0 = nn.Linear(DIM_INPUT, DIM_HIDDEN_LAYER)
        self.fc1 = nn.Linear(DIM_HIDDEN_LAYER, len(my_help_functions.Dictionary_of_classes))
        #create the E matrix


    def forward(self, x):
        x = self.E(x).view(-1, self.input_dim)
        x = F.tanh(self.fc0(x))

        x=self.fc1(x)

        x_softmax = F.log_softmax(x,dim=1)
        return x_softmax




if __name__ == "__main__":
    main(sys.argv[1:])


