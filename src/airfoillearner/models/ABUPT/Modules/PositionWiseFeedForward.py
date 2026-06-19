import torch.nn as nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4):
        super(PositionWiseFeedForward, self).__init__()
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
    
    