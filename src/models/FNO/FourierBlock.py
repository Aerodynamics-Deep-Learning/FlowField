import torch
import torch.nn as nn
from .FourierBlock_Components import *
from ..auxiliary_layers.convNd import *
from ..auxiliary import get_activation
from .FNO_Components import *

#region Fourier Layer

class FourierBlockND(nn.Module):

    def __init__(
        self, 
        in_channels:(int), out_channels:(int), modes:(list[int]), 
        kernel:(int), norm_weights:(str), norm_fft:(str), 
        factorization:(str), rank:(any), locallin_type:(str), 
        act_fn:(str), norm_type:(str), dropout:(float), 
        use_mlp:(bool), mlp_expansion:(float), stabilizer:(str), 
        use_skip:(bool)
    ):
        
        """
        The complete n-D Fourier Block. Take the input, apply spectral convolution, apply local linear convolution, add these two and that becomes the output. 

        Args:
            in_channels: is how many inputs we have to our channel
            out_channels: is how many outputs we want
            modes: is the number of Fourier modes to keep in the layer (the number of dimensions are defined by the len of this list)
            kernel: kernels for the local linear convolutions 
            norm_weights: option to normalizing weights; "paper" (1/ch1 * ch2), "geom" (1/(ch1 * ch2)**0.5), "xavier" ((2/(ch1 + ch2))**0.5)
            norm_fft: option to normalizing signal; "forward" (none), "backward" (1/n), "ortho" (1/sqrt(n))
            factorization: option to factorize the spectral weights; None (no factorization), "cp" (CANDECOMP/PARAFAC), "tucker" (Tucker decomposition)
            rank: rank for the factorization, int for cp, list of ints for tucker
            locallin_type: the type of local linearity, options are "conv" and "linear", conv uses convolution and linear uses nn.linear
            act_fn: activation function to be used internally, DO NOT APPLY ACT_FN POST THIS BLOCK!!
            norm_type: normalization type to be used internally, options are "instance", "batch", and None
            dropout: dropout probability to be used internally
            use_mlp: option to use an MLP to local feature mix
            mlp_expansion: internal channel expansion for the mlp
            stabilizer: stabilizer that neglects high variance inputs such as shocks post registering skip connection
            use_skip: if a skip connection is to be used, usually helpful if the FNO is deep
        """

        super().__init__()

        # Define the dims, using modes
        self.ndim = len(modes)
        self.use_mlp = use_mlp
        self.use_skip = use_skip

        # 1- Get the spectral convolution path
        self.SpectralConv = SpectralConvolutionND(in_channels, out_channels, modes, norm_weights, norm_fft, factorization, rank)

        # 2- Get the local transformation path
        self.use_linear_permute=False # Initialize permuting
        if locallin_type == "linear":
            if kernel!=1:
                raise ValueError(f"We don't like kernel and locallin option mismatch, if using {locallin_type}, use kernel=1") 
            # Equivalent to conv with kernel=1, works for any dim, fastest with ndim>3
            self.LocalLinear = nn.Linear(in_channels, out_channels)
            self.use_linear_permute=True # Gotta do linear permute if using this, essentially switchinh
                                         # the order of dims from (Batch, Ch, Spat Dim) to (Batch, Spat Dim, Ch)
        
        elif locallin_type == "conv":
            padding = kernel//2
            if self.ndim==1:
                self.LocalLinear = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=padding)
            if self.ndim==2:
                self.LocalLinear = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding)
            if self.ndim==3:
                self.LocalLinear = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, padding=padding)
            if self.ndim>3:
                kernel_list = (kernel,) * self.ndim; padding_list = (padding,) * self.ndim
                self.LocalLinear = convNd(in_channels, out_channels, num_dims=self.ndim, kernel_size=kernel_list, padding=padding_list, stride=1)

        # 3- Define the normalization layers
        self.norm1 = self._get_norm_layer(out_channels, norm_type)
        self.norm2 = self._get_norm_layer(out_channels, norm_type) if use_mlp else nn.Identity()

        # 4- Define the dropout
        if self.ndim == 1:
            self.dropout = nn.Dropout1d(dropout)
        elif self.ndim == 2:
            self.dropout = nn.Dropout2d(dropout)
        elif self.ndim == 3:
            self.dropout = nn.Dropout3d(dropout)
        else: 
            self.dropout = nn.Dropout(dropout) # There's no dropout for ndim>3 unf
        
        # 5- Define the Channel MLP
        if self.use_mlp:
            self.mlp = ChannelMLP(
                in_channels=out_channels,
                out_channels=out_channels,
                hidden_channels=[int(out_channels * mlp_expansion)],
                dropout=dropout,
                act_fn=act_fn
            )
        
        # 6- Get the activation
        self.act_fn = get_activation(act_fn)

        # 7- Get the stabilizer
        # This is insanely needed because we may have some shockwaves or insanely high variance since it is physical data!!
        # Truly gigachad stuff, effectively acts as an input clamper
        assert stabilizer in [None,'identity', 'tanh', 'sigmoid'], f"Stabilizer must be one of: 'identity', 'tanh', 'sigmoid'; {stabilizer} ain't supported" 
        if stabilizer==None: stabilizer='identity'
        self.stabilizer = get_activation(stabilizer)

        # 8- Get the skip connection projector
        if self.use_skip and in_channels != out_channels:
            if self.ndim == 1: self.skip_proj = nn.Conv1d(in_channels, out_channels, 1)
            elif self.ndim == 2: self.skip_proj = nn.Conv2d(in_channels, out_channels, 1)
            elif self.ndim == 3: self.skip_proj = nn.Conv3d(in_channels, out_channels, 1)
            else: self.skip_proj = nn.Linear(in_channels, out_channels)
        else:
            self.skip_proj = nn.Identity()

    def _get_norm_layer(self, channels:(int), norm_type:(str)=None):
        if norm_type is None:
            return nn.Identity()
        elif norm_type == "instance":
            if self.ndim == 1: return nn.InstanceNorm1d(channels)
            if self.ndim == 2: return nn.InstanceNorm2d(channels)
            if self.ndim == 3: return nn.InstanceNorm3d(channels)
            else: return nn.Identity()
        elif norm_type == "batch":
            if self.ndim == 1: return nn.BatchNorm1d(channels)
            if self.ndim == 2: return nn.BatchNorm2d(channels)
            if self.ndim == 3: return nn.BatchNorm3d(channels)
            else: return nn.Identity()
        else: raise ValueError(f'Unknown norm_type: {norm_type}, choose one of: "instance", "batch", None')

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input to the 1D Fourier layer (B, C_in, d1, ..., dn)
        Returns:
            x: Spectrally convoluted and linear convoluted 1D output (B, C_out, d1, ..., dn)
        """

        # 0.0- If using skip connection, allocate it
        if self.use_skip:
            x_skip = x

        # 1- Apply the stabilizer
        x = self.stabilizer(x)

        # 2- Original Fourier block
        # 2.1- Spectral Convolution
        x1 = self.SpectralConv(x)

        # 2.2- Local Linear Transformation
        if self.use_linear_permute:
            # Gotta do (Batch, Ch, d1, ..., dn) -> (Batch, d1, ..., dn, Ch)
            permute_order = [0] + list(range(2, 2 + self.ndim)) + [1]
            inverse_order = [0, self.ndim + 1] + list(range(1, self.ndim + 1))
    
            x_linear = x.permute(*permute_order)
            x_linear = self.LocalLinear(x_linear)
            x2 = x_linear.permute(*inverse_order); del x_linear
        else:
            x2 = self.LocalLinear(x)
        x = x1 + x2

        # 3- Additionals to the original Fourier Block
        x = self.norm1(x)
        x = self.act_fn(x)
        x = self.dropout(x)

        # 4- Apply the channel expansion, for local feature mixing (preserves channel amount)
        if self.use_mlp:
            x_res = x
            # 4.1- Do the MLP
            x = self.mlp(x)
            x = x + x_res
            # 4.2- Additionals to the MLP
            x = self.norm2(x)
            x = self.act_fn(x) # We always apply the last activation function, no matter what, this is because the fourier blocks
                               # will always be followed by a Q network.

        # 0.1- Add the skip if using
        if self.use_skip:
            # Handle projection if dimensions mismatch
            if isinstance(self.skip_proj, nn.Linear): # Gotta switch around channels (like with MLP) if using linear and not conv
                 permute_order = [0] + list(range(2, 2 + self.ndim)) + [1]
                 inverse_order = [0, self.ndim + 1] + list(range(1, self.ndim + 1))
                 x_skip = self.skip_proj(x_skip.permute(*permute_order)).permute(*inverse_order)
            else:
                 x_skip = self.skip_proj(x_skip)

            x += x_skip

        return x

#endregion