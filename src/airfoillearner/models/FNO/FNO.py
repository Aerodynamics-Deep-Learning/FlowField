
#region Imports

import torch.nn as nn

from .FourierBlock import *
from ..auxiliary import get_activation
from .FNO_Components import *

#endregion


#region FNO Class(es)

class FNO_nD(nn.Module):

    def __init__(
        self, cfg_p:(dict), cfg_q:(dict), 
        fb_hidden_channels:(list[int]), fb_modes:(list[list[int]]), fb_kernel:(list[int]),
        fb_norm_weights:(str)='geom', fb_norm_fft:(str)='ortho', fb_factorization:(str)=None, 
        fb_rank:(any)=None, fb_locallin_type:(str)='conv', fb_act_fn:(str)='gelu', 
        fb_norm_type:(str)='instance', fb_dropout:(float)=0.05, fb_use_mlp:(bool)=True, 
        fb_mlp_expansion:(int)=2, fb_stabilizer:(str)='tanh', fb_use_skip:(bool)=True
    ):
        
        """
        n=Dimensional Fourier Neural Operator. Elevates channel dim using P net, applies Fourier Blocks, lowers channel dim using Q net.
        The dimensions are primarily defined by the len(fb_modes[0]) input to the Fourier Blocks.

        Args:
            cfg_p (dict): Config for the p network, check details at ChannelMLP class
            cfg_q (dict): Config for the q network, check details at CHannelMLP Class

            fb_hidden_channels (list[int]): List of hidden channels for the Fourier Block layers
            fb_modes (list[list[int]]): List of lists of modes for the Fourier Block layers
            fb_kernel (list[int]): List of kernel sizes for the Fourier Block layers
            fb_norm_weights (str): Normalization type for the Fourier Block weights
            fb_norm_fft (str): Normalization type for the Fourier Block FFT
            fb_factorization (str): option to factorize the spectral weights; None (no factorization), "cp" (CANDECOMP/PARAFAC), "tucker" (Tucker decomposition)
            fb_rank (any): rank for the factorization, int for cp, list of ints for tucker
            fb_locallin_type (str): The type of local linearity in Fourier Blocks
            fb_act_fn (str): Activation function for the Fourier Block layers
            fb_norm_type (str): Internal normalization type for the Fourier Blocks
            fb_dropout (float): Internal dropout for the Fourier Blocks
            fb_use_mlp (bool): Option to use an MLP in the Fourier Blocks for local feature mixing
            fb_mlp_expansion (int): Internal channel expansion for the mlp in Fourier Blocks
            fb_stabilizer (str): Stabilizer that neglects high variance inputs such as shocks post registering skip connection
            fb_use_skip (bool): If a skip connection is to be used in Fourier Blocks, usually helpful if the FNO is deep
        """

        super().__init__()

        # Initialize P Network (elevator)
        self.P_Net = ChannelMLP(**cfg_p)

        # Initialize Q Network (reducer)
        self.Q_Net = ChannelMLP(**cfg_q)

        # Initialize Fourier Block
        self.fb_in = cfg_p['out_channels']
        self.fb_out = cfg_q['in_channels']

        assert len(fb_kernel) == len(fb_modes), 'Fourier Block kernel and modes lists have to be same len.'
        assert len(fb_hidden_channels) + 1 == len(fb_kernel), 'Hidden channel list has to be 1 less than kernel list in ken.'

        # Define for easier inputs
        FourierBlock_dict = {
            'norm_weights': fb_norm_weights, 
            'norm_fft': fb_norm_fft,
            'factorization': fb_factorization,
            'rank': fb_rank,
            'locallin_type': fb_locallin_type, 
            'act_fn': fb_act_fn, 
            'norm_type': fb_norm_type,
            'dropout': fb_dropout, 
            'use_mlp': fb_use_mlp, 
            'mlp_expansion': fb_mlp_expansion,
            'stabilizer': fb_stabilizer, 
            'use_skip': fb_use_skip,
            'factorization': fb_factorization,
            'rank': fb_rank
        }

        # Build Fourier Block, init block
        self.FourierBlock = [
            FourierBlockND(
                in_channels=self.fb_in, 
                out_channels=fb_hidden_channels[0], 
                modes=fb_modes[0],
                kernel=fb_kernel[0], 
                **FourierBlock_dict
            )
        ]

        # Additional blocks
        for i in range(len(fb_hidden_channels) - 1):

            self.FourierBlock.append(
                FourierBlockND(
                    in_channels=fb_hidden_channels[i], 
                    out_channels=fb_hidden_channels[i+1], 
                    modes=fb_modes[i+1],
                    kernel=fb_kernel[i+1],
                    **FourierBlock_dict
                )
            )

        # Last block
        self.FourierBlock.append(
                FourierBlockND(
                    in_channels=fb_hidden_channels[-1], 
                    out_channels=self.fb_out, 
                    modes=fb_modes[-1],
                    kernel=fb_kernel[-1],
                    **FourierBlock_dict
                )
            )

        self.FourierBlock = nn.Sequential(*self.FourierBlock)

    def forward(self, input:(torch.tensor)):

        """
        Forward prop for the FNO_nD class. Elevates channel dim using P net, applies Fourier Blocks, lowers channel dim using Q net.
        
        Args:
            input: An input of size (Batch, In_Channels, d1, ..., dn)

        Returns:
            output: An output of size (Batch, Out_Channels, d1, ..., dn)
        """

        # Put it through the p network first
        input = self.P_Net(input)

        # Then through the fourier blocks
        input = self.FourierBlock(input)

        # Lastly through the q network, return the result
        return self.Q_Net(input)

#endregion
