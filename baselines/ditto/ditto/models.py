"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""


# Will need development to add all the modules and understand how hydra can help


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
from typing import List
from collections import OrderedDict
# import random
# from matplotlib import pyplot as plt
# from math import comb
# from itertools import combinations
# import flwr as fl
# from flwr.common import Metrics
# Local import

class Net(nn.Module):
    """
    A CNN consisting of (in order):
        A 2D convolutional layer conv1 
        A 2D maxpooling layer pool
        A further 2D convolutional layer conv2
        3 fully connected linear layers, fc1, fc2, fc3 outputing a classification into one of ten bins.
    """
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """ A forward pass through the network. """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_parameters(net) -> List[np.ndarray]:
    """taking state_dict values to numpy (state_dict holds learnable parameters) """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    """ Setting the new parameters in the state_dict from numpy that flower operated on """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, epochs: int, option = None):
    """
    Train the network on the training set.
    
    Inputs:
        net - the instance of the model
        trainloader - a pytorch DataLoader object.
        epochs - the number of local epochs to train over
        option - a flag to enable alternative training regimes such as ditto
    """
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #sum_risks = None # Placeholder for fedminmax return
    def ditto_manual_update(lr, lam, glob):
        """ Manual parameter updates for ditto """
        with torch.no_grad():
            counter = 0
            q = [torch.from_numpy(g).to(DEVICE) for g in glob]
            for p in net.parameters():
                new_p = p - lr*(p.grad + (lam * (p - q[counter])))
                p.copy_(new_p)
                counter += 1
            return

    # def fedminmax_manual_update(lr, risk):
    #     """ Manual parameter updates for FedMinMax strategy"""
    #     with torch.no_grad():
    #         for p in net.parameters():
    #             new_p = p - (lr*(p.grad)*risk)
    #             p.copy_(new_p)
    #         return

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for i, data in enumerate(trainloader,0):
            images, labels = data["img"].to(DEVICE), data["label"].to(DEVICE)
            batch_size = len(labels)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward() # calculated the gradients
            if option is not None:
                if option["opt"] == "ditto":
                    # Ditto personalised updates, used https://discuss.pytorch.org/t/updatation-of-parameters-without-using-optimizer-step/34244/15
                    ditto_manual_update(option["eta"], option["lambda"], option["global_params"])

            else:
                optimizer.step()
            # Train metrics:
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
    return



def test(net, testloader, sensitive_labels=[]):
    """
    Evaluate the network on the inputted test set and determine the equalised odds for each protected group.
    
    Inputs:
        net - the instance of the model
        testloader - a pytorch DataLoader object.
        sensitive_labels - a list of the class indexes associated with the protected groups in question.

    Outputs:
        loss - average loss 
        accuracy - accuracy calculated as the number of correct classificatins out of the total
        group_performance - a list of equalised odds measurers for each protected group given.
    """
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    # group_performance = [[0,0] for label in range(len(sensitive_labels))] # preset for EOD calc, will store the performance
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # Cycles through in batches
            images, labels = data["img"].to(DEVICE), data["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            # Comparing the predicted to the inputs in order to determine EOD
            matched = (predicted == labels)
            # for label in range(len(sensitive_labels)):
            #   labelled = (labels == label)
            #   not_labelled = (labels != label)
            #   group_performance[label][0] += (matched == labelled).sum()
            #   group_performance[label][1] += (matched == not_labelled).sum()
            total += labels.size(0)
            correct += matched.sum().item()
    # for index in range(len(group_performance)):
    #     # Calculating EOD: P(Y.=1|A=1,Y=y) - P(Y.=1|A=0,Y=y) for each:
    #     group_performance[index] = float((group_performance[index][0] - group_performance[index][1]) / total)
    loss /= len(testloader.dataset)
    accuracy = correct / total

    return loss, accuracy #, group_performance