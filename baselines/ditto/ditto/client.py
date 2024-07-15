"""Defining client class and a function to construct such clients.
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import flwr as fl
from typing import Callable, List
import numpy as np
import torch
import models

# Global client personal parameter storage mechanism (would be stored locally at the client if we were not using simulation):
personal_parameters = {}

class FlowerClient(fl.client.NumPyClient):
    """
    Defines the client behaviour for the Ditto strategy.

    Attributes
    ----------
    cid - client id for dynamic client loading and identifying clients without occupying excessive memory.
    net - an instance of the neural net/model.
    trainloader - pytorch DataLoader object for loading the local train dataset.
    valloader - pytorch DataLoader object for loading the local test dataset.
    test - bespoke test function to validate the model.
    train - bespoke function to train the model.

    Methods:
        get_parameters - returns the current parameters of self.net
        fit - detects the strategy used, obtains strategy parameters, trains the model and 
            gathers metrics for fairness analytics.
    """
    def __init__(self, cid, net, trainloader, valloader, test_function, train_function):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.test = test_function
        self.train = train_function
        self.per_params = None

    def get_parameters(self, config):
        """ Return the parameters of self.net"""
        return models.get_parameters(self.net)

    def fit(self, parameters, config):
        """
        Obtain strategy information, orchestrates training and collects data.

        Parameters
        ----------
        parameters: model weights
            The new set of parameters from the aggregating server.
        config: Dict
            Dictionary passed from the strategy indicating the strategy's characteristics.

        Returns
        ----------
        params: model weights
            Updated parameters after E local epochs.
        Length of the trainloader.
        A dictionary containing key measurements required for fairness calculation and strategy behaviour.
        """
        # Unpacking config parameters:
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        if "client_"+str(self.cid) in personal_parameters:
            self.per_params = personal_parameters["client_"+str(self.cid)] # loading saved personal params
        else:
            print(f"Client {self.cid} pers param initialisation")
            self.per_params = parameters
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        models.set_parameters(self.net, parameters)
        # Training and storing the parameters at the end of training.
        self.train(self.net, self.trainloader, epochs=local_epochs)
        params = self.get_parameters(self.net)
        # Additional epochs training the personalised parameters.
        models.set_parameters(self.net, self.per_params)
        opts = {"opt": "ditto", "lambda": config['lambda'], "eta": config['eta'], "global_params": params}
        self.train(self.net, self.trainloader, epochs=int(config['s']), option=opts)                                                         
        # Updating personalised params stored:
        self.per_params = self.get_parameters(self.net)
        personal_parameters["client_"+str(self.cid)] = self.per_params # storing the parameters
        # Performing federated evaluation on the clients that are sampled for training:
        print(f"[Client {self.cid}] evaluate, config: {config}")
        loss, accuracy = self.test(self.net, self.valloader)

        return params, len(self.trainloader), {"cid":int(self.cid), 
                                               "accuracy": float(accuracy), 
                                               "loss": float(loss)}


def gen_client_fn(trainloaders, valloaders, net, test, train) -> Callable[[str], FlowerClient]:
    """
    Instances of clients are only created when required to avoid depleting RAM.

    Parameters
    ----------
    trainloaders: List of PyTorch DataLoader objects.
        The list of trainloaders for each client
    valloaders: List of PyTorch DataLoader objects.
        The corresponding list of vallidation DataLoaders
    net: model class
        The instance of the model to be trained.
    test: function
        The test function for model validation.
    train: function
        The model training function.

    Parameters
    ----------
    client_fn : function
        A function that returns an instance of a Flower client.
    """
    def client_fn(cid:str) -> FlowerClient:
        """ Creating a single flower client.

        Parameters
        ----------
        cid: int
            The unique client ID.

        Returns
        ----------
        An instance of a flower client corresponding to the input cid.
        """
        return FlowerClient(cid, net, trainloaders[int(cid)], valloaders[int(cid)], test, train).to_client()

    return client_fn