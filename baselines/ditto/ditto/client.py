"""Defining client class and a function to construct such clients.
"""
import flwr as fl
from typing import Callable, List
import numpy as np
import torch
import models

# The storage mechanism for the personalised parameters

# Still needs figuring out how to use hydra for this
NUM_CLIENTS = 10
personal_parameters = [None for client in range(NUM_CLIENTS)] 

class FlowerClient(fl.client.NumPyClient):
    """
    Defines the client behaviour for the Ditto strategy.

    Attributes:
        cid - client id for dynamic client loading without occupying excessive memory.
        net - a nn.Module derived object defining an instance of the neural net/model.
        trainloader - pytorch DataLoader object containing the local train dataset.
        valloader - pytorch DataLoader object containing the local test dataset.
        test - bespoke test function for the dataset used, defined in *_net.py files.
        train - bespoke train function for the dataset used, defined in *_net.py files.

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

        Inputs:
        parameters - the new set of parameters from the aggregating server.
        config - a dictionary passed from the strategy indicating the strategy's characteristics.

        Outputs: 
            params - updated parameters after E local epochs.
            len(self.trainloader)
            {...} a dict containing key measurements required for fairness calculation and strategy behaviour.
        """
        # Unpacking config parameters:
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        self.per_params = personal_parameters[int(self.cid)] # loading saved personal params
        if self.per_params == None:
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
        personal_parameters[int(self.cid)] = self.per_params # storing the parameters
        # Performing federated evaluation on the clients that are sampled for training:
        print(f"[Client {self.cid}] evaluate, config: {config}")
        loss, accuracy = self.test(self.net, self.valloader)

        return params, len(self.trainloader), {"cid":int(self.cid), 
                                               "accuracy": float(accuracy), 
                                               "loss": float(loss)}


def gen_client_fn(trainloaders, valloaders, net, test, train) -> Callable[[str], FlowerClient]:
    """
    Instances of clients are only created when required to avoid depleting RAM.
    """
    def client_fn(cid:str) -> FlowerClient:
        """ Creating a single flower client"""
        return FlowerClient(cid, net, trainloaders[int(cid)], valloaders[int(cid)], test, train).to_client()

    return client_fn