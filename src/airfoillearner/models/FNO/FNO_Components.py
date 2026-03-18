"""

Here, we define the Fourier Layers that are used in the Fourier Neural Operator (FNO) architecture.

"""

import torch
import torch.nn as nn

import sys

from ..auxiliary import get_activation
from ..auxiliary_layers.convNd import *

#region Scripts
# Scripts for complex multiplication in different dimensions.

#region MLPLayer

class ClassicMLP(nn.Module): # Useless for now

    def __init__(self, input_dim:(int), output_dim:(int), hidden_units:(list[int]), act_fn:(str)):
        super().__init__()

        # Initialize MLP parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.act_fn = get_activation(act_fn or "gelu")

        # Add input layer
        self.layers = [nn.Linear(in_features=self.input_dim, out_features=self.hidden_units[0]), self.act_fn]

        # Add hidden layers
        for i in range(len(self.hidden_units)-1):
            self.layers.extend([
                nn.Linear(in_features=self.hidden_units[i], out_features=self.hidden_units[i+1]),
                self.act_fn
                ])

        # Add output layer
        self.layers.append(nn.Linear(in_features=self.hidden_units[-1], out_features=self.output_dim))

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, input):

        # Return the forward prop
        return self.mlp(input)
    

class ChannelMLP(nn.Module): # Used in FNO for P and Q nets

    def __init__(
        self, 
        in_channels:(int), 
        out_channels:(int), 
        hidden_channels:(list[int]), 
        dropout:(float)=0.0, 
        act_fn:(str)='gelu'
    ):
        
        """
        This layer is equivalent to an MLP mathematically, but saves huge amount of memory. It primarily aids in expanding 
        the channel size of an input. This is done with flattening the (Batch, In_Channels, x, y, z, ...) input to get,
        (Batch, In_Channels, x*y*z*...), then applying a Conv1D (kernel=1), and reshaping back to get
        (Batch, Out_Channels, x, y, z, ...).
        
        Args:
            in_channels (int): The in_channels present at the dataset, usually=1
            out_channels (int): The out_channels we want to get, before Fourier Blocks
            hidden_channels (list[int]): The intermediary channels of use
            dropout (float): How much dropout to apply after sigma(Conv1D)
            act_fn (str): Act function of choice
        """

        super().__init__()

        # Take in the inputs
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert len(hidden_channels)>=1, "Hidden channels must be at least1" ; self.hidden_channels = hidden_channels
        self.act_fn = get_activation(act_fn)

        # Build the layers list. For optimization reasons, we will build dropout as a different list. This is because
        # if we keep dropout=0, it still has a lot of computational overhead. Especially for smaller P and Q nets.

        # Dropout list
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(len(self.hidden_channels))])
            if dropout > 0.0
            else None
        )

        # Conv1D layers
        self.conv1d_layers = nn.ModuleList()

        # First layer
        self.conv1d_layers.append(nn.Conv1d(self.in_channels, self.hidden_channels[0], 1))

        # Hidden layers
        for i in range(len(self.hidden_channels) - 1):
            self.conv1d_layers.append(nn.Conv1d(self.hidden_channels[i], self.hidden_channels[i+1], 1))
        
        # Last layer
        self.conv1d_layers.append(nn.Conv1d(self.hidden_channels[-1], self.out_channels, 1))

    def forward(self, x:(torch.tensor)):

        """
        Forward prop for the ChannelMLP super-layer. Takes in the input, flattens spatial dims,
        applies Conv1d (kernel=1)

        Args:
            x (torch.tensor): With size (Batch, In_Channels, x, y, z, ...)
        
        Returns:
            x (torch.tensor): With size (Batch, Out_Channels, x, y, z, ...)
        
        """

        reshaped = False # Set this so we can track if we have reshaped the input tensor

        # If we have multiple spat-dims, reshape x such that the spat-dims are flattened, while B, C_in dims are preserved
        if x.ndim > 3:
            size = list(x.shape)
            x = x.reshape((*size[:2], -1)) # Reshape and not view because it can handle non-contigious too
            reshaped = True # Set this to true, so that we can retrieve the size post layers
        
        for i, conv1d in enumerate(self.conv1d_layers):

            x = conv1d(x) # Input x into the convolutional layer

            if i < len(self.hidden_channels):
                x = self.act_fn(x) # Spare last layer from act_fn
                if self.dropout is not None:
                    x = self.dropout[i](x) # Apply the relevant dropout layer, if it exists

        # Restore the true shape if we have reshaped it
        if reshaped:
            x = x.reshape((size[0], self.out_channels, *size[2:]))
        
        return x

#endregion

