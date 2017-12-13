import torch.nn as nn

class MLP(nn.Module):
    """
    Module for an MLP with dropout.
    """

    def __init__(self, input_size, layer_size, depth, dropout):
        super(MLP, self).__init__()
        sequence = [nn.Linear(input_size, layer_size)] + [
            nn.Linear(layer_size, layer_size), nn.ReLU(), nn.Dropout(dropout)
        ]
        self.layers = nn.Sequential(
            *sequence
        )

    def forward(self, x):
        x_reshape = x.contiguous().view(-1, x.size(-1)) 
        return self.layers(x_reshape)
