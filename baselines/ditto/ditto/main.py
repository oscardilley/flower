"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# packages:
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


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.

    NEED TO INVESTIGATE USING THE CONFIG TOOLS TO HANDLE THE VARIABLES


    """
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))




    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
    )

    # Key parameter and data storage variables:
    NUM_CLIENTS = 10
    LOCAL_EPOCHS = 2
    NUM_ROUNDS = 5
    BATCH_SIZE = 32
    SELECTION_RATE = 0.3 # what proportion of clients are selected per round
    personal_parameters = [None for client in range(NUM_CLIENTS)] # personal parameters stored globally for simulation - PROBLEMATIC, TRY TO CHANGE
    DITTO_LAMBDA = 0.8
    DITTO_ETA = 0.01
    DITTO_PERS_EPOCHS = LOCAL_EPOCHS

    def fit_config(server_round: int):
        """
        Return training configuration dict for each round.
        """
        config = {
            "server_round": server_round, # The current round of federated learning
            "local_epochs": LOCAL_EPOCHS,
        }
        return config

    def evaluate(server_round: int,
                parameters: fl.common.NDArrays,
                config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        """
        Used for centralised evaluation. This is enacted by flower before the Federated Evaluation.
        Runs initially before FL begins as well.
        """
        net = models.Net().to(DEVICE)
        # shap.aggregatedRoundParams = parameters
        models.set_parameters(net, parameters)
        loss, accuracy = models.test(net, testloader)
        # shap.f_o = accuracy # stored incase the user wants to define orchestrator fairness by the central eval performance, usused by default
        # shap.centralLoss = loss
        # shap.round = server_round
        print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
        return loss, {"accuracy": accuracy}

    def fit_callback(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """
        Called at the end of the clients fit method
        Used to call the Shapley calculation as it is before weight aggregation
        """
        #clients = set()
        parameters = [None for client in range(NUM_CLIENTS)]
        # Why are the parameters we get here the post aggregation ones...?
        for client in metrics:
            cid = client[1]["cid"]
            #clients.add(cid)
            #parameters[cid] = client[1]["parameters"]
            personal_parameters[cid] = np.ndarray(client[1]["personal_parameters"])
        accuracies = np.array([metric["accuracy"] for _,metric in metrics])

        return {"test": 31}










    # 2. Prepare your dataset
    # here you should call a function in datasets.py that returns whatever is needed to:
    trainloaders, valloaders, testloader, _ = dataset.load_iid(NUM_CLIENTS, BATCH_SIZE)
    # 3. Define your clients
    # Define a function that returns another function that will be used during
    # simulation to instantiate each individual client
    net = models.Net().to(DEVICE)
    client_fn = client.gen_client_fn(trainloaders, 
                                     valloaders, 
                                     net,
                                     models.test, 
                                     models.train,
                                     personal_parameters) # this may be problematic as may not update properly

    # 4. Define your strategy
    ditto = strategy.Ditto(
        ditto_lambda = DITTO_LAMBDA, # Recommended value from paper for FEMNIST with no malicious clients, it can be selected locally or tuned
        ditto_eta = DITTO_ETA, # suggested hyperparameter for FEMNIST
        ditto_s = DITTO_PERS_EPOCHS, # the number of personalised fitting epochs
        fraction_fit=SELECTION_RATE, # sample all clients for training
        fraction_evaluate=0.0, # Disabling federated evaluation
        min_fit_clients=int(NUM_CLIENTS*SELECTION_RATE), # never sample less that this for training
        min_evaluate_clients=int(NUM_CLIENTS*SELECTION_RATE), # never sample less than this for evaluation
        min_available_clients=NUM_CLIENTS, # has to wait until all clients are available
        # Passing initial_parameters prevents flower asking a client:
        initial_parameters=fl.common.ndarrays_to_parameters(models.get_parameters(models.Net())),
        # Called whenever fit or evaluate metrics are received from clients:
        fit_metrics_aggregation_fn = fit_callback,
        # Evaluate function is called by flower every round for central evaluation:
        evaluate_fn=evaluate,
        # Altering client behaviour with the config dictionary:
        on_fit_config_fn=fit_config,
        accept_failures = False
    )
    # 5. Start Simulation
    client_resources = None # {"num_cpus": 1, "num_gpus": 0.0}
    if DEVICE.type == "cuda":
        # here we are asigning an entire GPU for each client.
        client_resources = {"num_cpus": 1, "num_gpus": 1.0}

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS), 
        strategy=ditto,
        client_resources=client_resources,
    )

    # 6. Save your results
    # Here you can save the `history` returned by the simulation and include
    # also other buffers, statistics, info needed to be saved in order to later
    # on generate the plots you provide in the README.md. You can for instance
    # access elements that belong to the strategy for example:
    # data = strategy.get_my_custom_data() -- assuming you have such method defined.
    # Hydra will generate for you a directory each time you run the code. You
    # can retrieve the path to that directory with this:
    # save_path = HydraConfig.get().runtime.output_dir


if __name__ == "__main__":
    main()
