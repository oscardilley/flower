"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
import matplotlib.pyplot as plt
import numpy as np
import hydra
import json

@hydra.main(config_path="conf", config_name="base", version_base=None)
def accuracy_plot(cfg):
    ROUNDS = cfg.strategy.num_rounds
    CLIENTS = cfg.strategy.num_clients
    SELECTED_CLIENTS = int(cfg.strategy.num_clients * cfg.strategy.fraction_fit)
    # Spoof data:
    # fed_eval_mean_spoof = np.sort(np.random.uniform(0, 1, size=ROUNDS))
    # fed_eval_spoof = [np.abs(np.tanh(np.random.normal(loc=fed_eval_mean_spoof[j], scale=0.1, size=SELECTED_CLIENTS))) for j in range(ROUNDS)]
    # central_spoof = [(np.mean(i)+np.random.normal(loc=0, scale=0.03)) for i in fed_eval_spoof]
    # data = {"central_eval": central_spoof,
    #     "federated_eval":fed_eval_spoof,
    #     "federated_eval_mean":fed_eval_mean_spoof}
    # Data processing:
    with open('./ditto/Results/results.json', "r") as file:
        data = json.load(file)
    print(data)
    rounds = [i+1 for i in range(ROUNDS)]
    central_eval = data["central_eval"]
    fed_eval_mean = np.array([np.mean(i) for i in data["federated_eval"]])
    fed_eval_std = np.array([np.std(i) for i in data["federated_eval"]])
    # Figure instanciation:
    fig,ax = plt.subplots(1)
    fig.suptitle("Ditto Accuracy Performance", fontsize = 15)
    fig.tight_layout()
    # Plotting average:
    ax.grid(linewidth = 0.5, linestyle = "--")
    ax.set_axisbelow(True)
    ax.plot(rounds, central_eval)
    ax.plot(rounds, fed_eval_mean)
    ax.fill_between(rounds, fed_eval_mean + fed_eval_std, fed_eval_mean - fed_eval_std, alpha=0.3)
    ax.set_xlabel("Round")
    ax.set_xlim([0,ROUNDS])
    ax.set_ylabel("Accuracy")
    ax.set_ylim([0,1])
    ax.set_xticks([i for i in range(0, ROUNDS, 5)],[i for i in range(0, ROUNDS, 5)])
    ax.legend(["Central Evaluation", "Federated Evaluation"])
    ax.set_title("Average Accuracy")
    # Saving:
    fig.savefig("./ditto/Results/Images/average_accuracy.png",bbox_inches='tight', dpi=300)
    return
    


if __name__ == "__main__":
    accuracy_plot()


