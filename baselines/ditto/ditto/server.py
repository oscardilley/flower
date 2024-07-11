"""Create global evaluation function.

Optionally, also define a new Server class (please note this is not needed in most
settings).
"""

# Currently unused as janky to pass testloader across

# import torch
# import flwr as fl
# from typing import Dict, List, Optional, Tuple
# import models

# def evaluate(server_round: int,
#             parameters: fl.common.NDArrays,
#             config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
#     """
#     Used for centralised evaluation. This is enacted by flower before the Federated Evaluation.
#     Runs initially before FL begins as well.
#     """
#     models.set_parameters(net, parameters)
#     loss, accuracy = models.test(net, testloader)
#     print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")

#     return loss, {"accuracy": accuracy}