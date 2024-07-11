"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

import flwr as fl
from typing import Callable, List
import numpy as np
import torch
import models

class FlowerClient(fl.client.NumPyClient):
    """
    Defines the client behaviour for the Ditto strategy.

    Attributes:
        cid - client id for dynamic client loading without occupying excessive memory.
        net - a nn.Module derived object defining an instance of the neural net/model.
        temp - a copy of net used for data formatting in ditto implementation.
        trainloader - pytorch DataLoader object containing the local train dataset.
        valloader - pytorch DataLoader object containing the local test dataset.
        test - bespoke test function for the dataset used, defined in *_net.py files.
        train - bespoke train function for the dataset used, defined in *_net.py files.
        per_params - parameter attribute for storing personalised parameters for Ditto.
        risks - the risks per protected group for the FedMinMax implementation.

    Methods:
        get_parameters - returns the current parameters of self.net
        fit - detects the strategy used, obtains strategy parameters, trains the model and 
            gathers metrics for fairness analytics.
    """
    def __init__(self, cid, net, trainloader, valloader, test_function, train_function, per_params=None):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.test = test_function
        self.train = train_function
        self.per_params = per_params
        self.risks = None
        # Checking client dataset length without assuming batch size
        dataset_length = 0
        for data in trainloader:
            if "label" in data:
                labels = data["label"]
            else:
                labels = data["class"] # accounts for NSL-KDD
            dataset_length += len(labels)
        for data in valloader:
            if "label" in data:
                labels = data["label"]
            else:
                labels = data["class"] # accounts for NSL-KDD
            dataset_length += len(labels)
        self.dataset_len = dataset_length

    def get_parameters(self, config):
        """ Return the parameters of self.net 

        """
        print(f"[Client {self.cid}] get_parameters")
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
        # Detecting the strategy on the first round and setting a flag:
        print(f"CONFIGGGG: {config}")
        if self.per_params == None:
            print(f"Client {self.cid} pers param initialisation")
            self.per_params = parameters
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        # Training and evaluating
        models.set_parameters(self.net, parameters)
        # Training and storing the parameters at the end of training.
        self.train(self.net, self.trainloader, epochs=local_epochs)
        params = self.get_parameters(self.net)
        models.set_parameters(self.net, self.per_params)
        opts = {"opt": "ditto", "lambda": config['lambda'], "eta": config['eta'], "global_params": params}
        self.train(self.net, self.trainloader, epochs=int(config['s']), option=opts)                                                         
        # Updating personalised params stored:
        self.per_params = self.get_parameters(self.net)

        # FEDERATED EVALUATION SHOULD ALSO BE IN ITS OWN FUNCTION

        # Performing federated evaluation on the clients that are sampled for training:
        print(f"[Client {self.cid}] evaluate, config: {config}")
        loss, accuracy = self.test(self.net, self.valloader)

        return params, len(self.trainloader), {"cid":int(self.cid), 
                                               "personal_parameters": self.per_params,
                                               #"parameters": params, # shouldn't need this anyway as parameters already passed
                                               "accuracy": float(accuracy), 
                                               "loss": float(loss)}



def gen_client_fn(trainloaders, 
                  valloaders, 
                  net,
                  test, 
                  train, 
                  personal_parameters) -> Callable[[str], FlowerClient]:
    """
    Instances of clients are only created when required to avoid depleting RAM.
    """
    def client_fn(cid:str) -> FlowerClient:
        """ Creating a single flower client"""
        return FlowerClient(cid, net, trainloaders[int(cid)], valloaders[int(cid)], test, train, personal_parameters[int(cid)]).to_client()

    return client_fn