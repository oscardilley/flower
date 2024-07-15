"""Defining our models, and training and eval functions.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List
from collections import OrderedDict


# Need to change to have the different models for the different datasets

class Cifar10Net(nn.Module):
    """
    A CNN consisting of (in order):
        A 2D convolutional layer conv1 
        A 2D maxpooling layer pool
        A further 2D convolutional layer conv2
        3 fully connected linear layers, fc1, fc2, fc3 outputing a classification into one of ten bins.
    """
    def __init__(self) -> None:
        super(Cifar10Net, self).__init__()
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

class FEMNISTNet(nn.Module):
    """
    A CNN consisting of (in order):
        Two 2D convolutional layers, conv1 and conv2 each using leaky relu and followed by maxpooling layer.
        A linear layer with dropout, fcon1
        A fully connected linear layer to output into 62 bins corresponding to the 62 FEMNIST classes.
    """

    # Model needs reconfiguring from herre: https://github.com/litian96/ditto/blob/master/flearn/models/femnist/cnn.py 

    def __init__(self) -> None:
        super(FEMNISTNet, self).__init__() 
        # Prevous model used in research
        # self.fmaps1 = 40
        # self.fmaps2 = 160
        # self.dense = 200
        # self.dropout = 0.4
        # self.batch_size = 32
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=self.fmaps1, kernel_size=5, stride=1, padding='same'),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.fmaps1, out_channels=self.fmaps2, kernel_size=5, stride=1, padding='same'),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        # )
        # self.fcon1 = nn.Sequential(nn.Linear(49*self.fmaps2, self.dense), nn.LeakyReLU())
        # self.fcon2 = nn.Linear(self.dense, 62)
        # self.dropout = nn.Dropout(p=self.dropout)

        # Inferred from the Ditto git repo
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding='same')
        self.fc1 = nn.Linear(7 * 7 * 32, 128)
        self.fc2 = nn.Linear(128, 62)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ A forward pass through the network. """
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = x.view(x.size(0), -1)
        # x = self.dropout(self.fcon1(x))
        # x = self.fcon2(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 7 * 7 * 32)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_parameters(net) -> List[np.ndarray]:
    """taking state_dict values to numpy (state_dict holds learnable parameters).

    Parameters
    ----------
    net: model class
        The instance of the model
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    """ Setting the new parameters in the state_dict from numpy that flower operated on.

    Parameters
    ----------
    net: model class
        The instance of the model
    parameters: model weights
        The parameters to set in the state dictionary.
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, epochs: int, learning_rate: float, option = None):
    """
    Train the network on the training set.
    
    Parameters
    ----------
    net: model class
        The instance of the model
    trainloader: PyTorch DataLoader object.
        Handles the loading of the training dataset.
    epochs: int
        The number of local epochs to train over.
    learning_rate: float
        The learning rate, for both the personal and global objective.
    option: dictionary
        Enables flagging to inform the function that alternative training regimes such as Ditto are to be used.
    """
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def ditto_manual_update(lr, lam, glob):
        """ Manual parameter updates for ditto.

        Parameters
        ----------
        lr: float
            The learning rate.
        lam: float
            Ditto hyperparameter controlling interpolation between local and global models.
        glob: model weights
            The global model parameters.
        """
        with torch.no_grad():
            counter = 0
            q = [torch.from_numpy(g).to(DEVICE) for g in glob]
            for p in net.parameters():
                new_p = p - lr*(p.grad + (lam * (p - q[counter])))
                p.copy_(new_p)
                counter += 1
            return

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
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
                    ditto_manual_update(learning_rate, option["lambda"], option["global_params"])
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



def test(net, testloader):
    """
    Evaluate the network on the inputted test set and determine the equalised odds for each protected group.
    
    Parameters
    ----------
    net: model class
        The instance of the model
    testloader: PyTorch DataLoader object
        Handles loading of the testset.

    Returns
    ----------
    loss: float
        The average loss.
    accuracy: float 
        Accuracy calculated as the number of correct classificatins out of the total
    """
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # Cycles through in batches collecting the results
            images, labels = data["img"].to(DEVICE), data["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            matched = (predicted == labels)
            total += labels.size(0)
            correct += matched.sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total

    return loss, accuracy 