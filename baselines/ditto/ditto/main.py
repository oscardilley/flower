"""Creating and connecting the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import flwr as fl
from flwr.common import Metrics
import dataset
import client
import strategy
import models
import server
import json

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # 2. Preparing dataset and data structure
    trainloaders, valloaders, testloader, features = dataset.load_dataset(cfg=cfg)
    data = {"central_eval": [None for i in range(cfg.strategy.num_rounds)],
            "federated_eval":[[None for i in range(int(cfg.strategy.num_clients * cfg.strategy.fraction_fit))] for j in range(cfg.strategy.num_rounds)],
            "federated_eval_mean":[None for i in range(cfg.strategy.num_rounds)]}

    # 3. Defining clients
    if cfg.dataset.set == "cifar10":
        model = models.Cifar10Net()
    elif cfg.dataset.set == "flwrlabs/femnist":
        model = models.FEMNISTNet()
    else:
        raise dataset.ConfigErrorException("Invalid dataset passed through Hydra")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")
    net = model.to(DEVICE)
    client_fn = client.gen_client_fn(trainloaders, valloaders, net, models.test, models.train) 

    # 4. Defining strategy
    def fit_config(server_round: int):
        """
        Return training configuration dict for each round.

        Parameters
        ----------
        server_round : int
            Tracking the current server round.
        """
        config = {
            "server_round": server_round, # The current round of federated learning
            "local_epochs": cfg.strategy.local_epochs}
        return config

    def evaluate(server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """
        Used for centralised evaluation. This is enacted by flower before the Federated Evaluation.
        Runs initially before FL begins as well.

        Parameters
        ----------
        server_round : int
            Tracking the current server round.
        parameters: fl.common.NDArrays
            The global model parameters at the end of the server round.
        config: Dict
            The centralised evaluation configuration dictionary.
        """
        models.set_parameters(net, parameters)
        loss, accuracy = models.test(net, testloader)
        print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
        if server_round > 0:
            data["central_eval"][server_round - 1] = accuracy # saving data to dict for json

        return loss, {"accuracy": accuracy}
    
    def fit_callback(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """ Handling the returns of the client fit method to log federated evaluation results

        Parameters
        ----------
        metrics: list of dict
            The custom structure returned by the client fit method.

        Returns
        ----------
        A custom dictionary of metrics. In this case, the mean federated evaluation accuracy.

        """
        server_round = metrics[0][1]["round"]
        accuracies = np.array([metric["accuracy"] for _,metric in metrics])
        fed_eval_mean = np.mean(accuracies)
        data["federated_eval"][server_round - 1] = accuracies # data logging
        data["federated_eval_mean"][server_round - 1] = fed_eval_mean
        fed_eval_mean = np.mean(accuracies)
        print(f"Mean federated evaluation accuracy {fed_eval_mean}")

        return {"Fed Eval Accuracy": np.mean(accuracies)}

    ditto = strategy.Ditto(
        cfg=cfg.strategy,
        ditto_lambda = cfg.strategy.ditto_lambda, # Recommended value from paper for FEMNIST with no malicious clients, it can be selected locally or tuned
        ditto_eta = cfg.strategy.ditto_eta, # suggested hyperparameter for FEMNIST
        ditto_s = cfg.strategy.ditto_pers_epochs, # the number of personalised fitting epochs
        fraction_fit=cfg.strategy.fraction_fit, # sample all clients for training
        fraction_evaluate=cfg.strategy.fraction_evaluate, # Disabling federated evaluation
        min_fit_clients=int(cfg.strategy.num_clients*cfg.strategy.fraction_fit), # never sample less that this for training
        min_evaluate_clients=int(cfg.strategy.num_clients*cfg.strategy.fraction_evaluate), # never sample less than this for evaluation
        min_available_clients=cfg.strategy.num_clients, # has to wait until all clients are available
        accept_failures = bool(cfg.strategy.accept_failures),
        initial_parameters=fl.common.ndarrays_to_parameters(models.get_parameters(model)),
        fit_metrics_aggregation_fn = fit_callback,
        evaluate_fn=evaluate, # central evaluation function
        on_fit_config_fn=fit_config, # altering client behaviour
    )
    # 5. Start Simulation
    client_resources = None # {"num_cpus": 1, "num_gpus": 0.0}
    if DEVICE.type == "cuda":
        # here we are asigning an entire GPU for each client.
        client_resources = {"num_cpus": 1.0, "num_gpus": 1.0}

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.strategy.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.strategy.num_rounds), 
        strategy=ditto,
        client_resources=client_resources,
    )

    # 6. Saving results
    with open('./ditto/Results/results.json', "w") as outfile:
        data = json.dump(data, outfile)
    # Call plotting functions from util

if __name__ == "__main__":
    main()
