import torch
import torch.nn as nn
import numpy as np


# ---- Neural Network for Locomotion Control ----
class NeuralController(nn.Module):
    def __init__(self, input_size, output_size, init_params=False):
        super(NeuralController, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)  # Hidden layer with 16 neurons
        self.fc2 = nn.Linear(16, output_size)  # Output layer
        if init_params: self.apply(initialize_weights)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function
        x = self.fc2(x)  # Output layer
        #return torch.tanh(x) * 100  # Outputs actions for the robot
        return torch.relu(x) * 100  # Outputs actions for the robot

# ---- Convert Weights to NumPy Arrays ----
def get_weights(model, flatten=False):
    """Extract weights from a PyTorch model as NumPy arrays."""
    if flatten:
        return np.concatenate([p.detach().numpy().flatten() for p in model.parameters()])
    else:
        return [p.detach().numpy() for p in model.parameters()]

# ---- Load Weights Back into a Model ----
def set_weights(model, new_weights, reconstruct_weights=False):
    if reconstruct_weights:
        weights = []
        index = 0
        for param in model.parameters():
            shape = param.shape
            size = np.prod(shape)
            chunk = new_weights[index : index+size]
            weights.append(chunk.reshape(shape))
            index += size
        new_weights = weights

    """Update PyTorch model weights from a list of NumPy arrays."""
    for param, new_w in zip(model.parameters(), new_weights):
        param.data = torch.tensor(new_w, dtype=torch.float32)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier initialization
        nn.init.constant_(m.bias, 0.1)  # Small bias


def save_weights(brain, filename='best_weights.pth'):
    torch.save(brain.state_dict(), filename)
    print(f"Saved weights to {filename}")