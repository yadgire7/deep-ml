import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE
    features = torch.tensor(features).float()
    weights = torch.tensor(weights).float()
    labels = torch.tensor(labels)
    output = torch.matmul(features, weights) + torch.tensor(bias).float()
    activated_op = torch.sigmoid(output)
    activated_op_rounded = [round(val.item(), 4) for val in activated_op]
    mse = torch.mean((torch.sigmoid(output) - labels)**2)
    return (activated_op_rounded, round(mse.item(), 4))