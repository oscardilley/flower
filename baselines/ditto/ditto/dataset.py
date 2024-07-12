"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.preprocessor.divider import Divider
from flwr_datasets.partitioner import DirichletPartitioner, NaturalIdPartitioner
from flwr_datasets.visualization import plot_label_distributions
import matplotlib as plt
from flwr.common.logger import log
from logging import WARNING

class ConfigErrorException(Exception):
    pass


def load_dataset(cfg):
    """ Handling config options to return the correct dataloaders
    
    from Ditto paper: "We randomly split local data on each device into 72% train, 8% validation, 
    and 20% test sets, and report all results on test data" For FEMNIST, all results reported
    in the main paper are from the natural partition.
    """
    # Transforms:
    cifar10_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    femnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])

    # Dividers used to handle case that resplitting is required
    dividers = {"flwrlabs/femnist" : Divider(divide_config={"train":0.80, "test": 0.2}, divide_split="train"), 
                "cifar10": None}

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        # can add some better, config based selection in here
        if cfg.dataset.set == "cifar10":
            batch["img"] = [cifar10_transforms(img) for img in batch["img"]]
        elif cfg.dataset.set == "flwrlabs/femnist":
            batch["img"] = [femnist_transforms(img) for img in batch["image"]]
            batch["label"] = [label for label in batch["character"]]
            del batch["character"]
            del batch["image"]
        else:
            log(WARNING, "No data transform applied")
        return batch

    # Partitoning:
    if cfg.dataset.partition == "iid":
        fds = FederatedDataset(dataset=cfg.dataset.set, partitioners={"train": cfg.strategy.num_clients}, preprocessor = dividers[cfg.dataset.set])
    elif cfg.dataset.partition == "dirichlet":
        partitioner = DirichletPartitioner(num_partitions=cfg.strategy.num_clients, partition_by="label",
                                    alpha=cfg.dataset.dirichlet.alpha, min_partition_size=cfg.dataset.dirichlet.min_partition,
                                    self_balancing=bool(cfg.dataset.dirichlet.self_balancing))
        fds = FederatedDataset(dataset=cfg.dataset.set, partitioners={"train": partitioner}, preprocessor = dividers[cfg.dataset.set])
    elif cfg.dataset.partition == "natural":
        partitioner = NaturalIdPartitioner(partition_by="writer_id")
        fds = FederatedDataset(dataset=cfg.dataset.set, partitioners={"train":partitioner}, preprocessor = dividers[cfg.dataset.set])
    else:
        raise ConfigErrorException("Invalid Partition provided from Hydra")
    
    # Creating PyTorch loaders:
    testset = fds.load_split("test") # central testset
    testloader = DataLoader(testset.with_transform(apply_transforms), batch_size=cfg.dataset.batch_size)
    features = testset.features
    trainloaders = []
    valloaders = []
    for c in range(cfg.strategy.num_clients):

        partition = fds.load_partition(c) # this needs randomising for natural ID partitoning

        # Divide data on each node based on the required train:validation split
        partition_train_test = partition.train_test_split(test_size=cfg.dataset.validation_size)
        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloaders.append(DataLoader(partition_train_test["train"], batch_size=cfg.dataset.batch_size, shuffle=True))
        valloaders.append(DataLoader(partition_train_test["test"], batch_size=cfg.dataset.batch_size))

    # Visualisation
    fig, ax, df = plot_label_distributions(
        partitioner=fds.partitioners["train"],
        label_name="character", # need to solve this through config, "label" for cifar10, "character" for femnist
        plot_type="bar",
        size_unit="absolute",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        cmap="tab20b",
        title="Per Partition Labels Distributions"
    )
    fig.savefig("./Images/partitions.png",bbox_inches='tight')

    return trainloaders, valloaders, testloader, features
