import torch
import torch.nn as nn

#region Einsum

class compl_mulnd(nn.Module):

    def __init__(self, ndim:(int)):
        super().__init__()

        spatial_chars = "xyztsruv" # chars pool for dims
        current_dims = spatial_chars[:ndim]
        self.equation = f"bi{current_dims},io{current_dims}->bo{current_dims}"

    def forward(self, a:(torch.Tensor), b:(torch.Tensor)):
        return torch.einsum(self.equation, a, b)

#endregion

#region Spectral Convolution
    
class SpectralConvolutionND(nn.Module):

    """
    Performs the n-D spectral (Fourier) convolution operation.

    Args:
        in_channels: is how many input channels we have
        out_channels: is how many output channels we want
        modes: is the number of Fourier modes to keep in the layer, given as a list of ints of size n
        norm_weights: option to normalizing weights; "paper" (1/ch1 * ch2), "geom" (1/(ch1 * ch2)**0.5), "xavier" ((2/(ch1 + ch2))**0.5)
        norm_fft: option to normalizing signal; "forward" (none), "backward" (1/n), "ortho" (1/sqrt(n))
        factorization: option to factorize the spectral weights; None (no factorization), "cp" (CANDECOMP/PARAFAC), "tucker" (Tucker decomposition)
        rank: rank for the factorization, int for cp, list of ints for tucker

    Returns:
        x: spectrally convoluted n-D signal
    """

    def __init__(self, in_channels:(int), out_channels:(int), modes:(list[int]), norm_weights:(str)="paper", norm_fft:(str)="ortho", factorization:(str)=None, rank:(int)=1):
        super().__init__()

        # 1- Take in inputs
        self.in_channels = in_channels # Total number of in_channels
        self.out_channels = out_channels # Total number of out_channels
        self.norm_weights = norm_weights # Which norm to use for weight initialization
        self.norm_fft = norm_fft # Which norm to apply to the fft and ifft operations

        # How many modes of the FT we want to keep, this is an n-dim list
        self.modes = modes

        # Define the dimensions
        self.ndim = len(modes)


        # 2- Define the weights, depending on the factorization option
        spatial_chars = "xyztsruv"[:self.ndim]

        if factorization is None:
            # Initialize the weights for in-frequency-space linear transformation, the weights are complex numbers, so we use torch.cfloat
            weight_shape = [in_channels, out_channels] + self.modes
            self.weights = nn.Parameter(
                torch.empty(*weight_shape, dtype=torch.cfloat)
            ) # This matrix has dims: (in_ch, out_ch, d1, ..., dn)
            scale = self._get_scale(in_channels, out_channels, norm_weights)
            with torch.no_grad():
                self.weights.copy_(
                    scale * torch.randn(*weight_shape, dtype=torch.cfloat)
                )

        elif factorization == 'cp':
            
            # Equation: "bixyz,ir,or,xr,yr,zr->boxyz"
            lhs = f"bi{spatial_chars},ir,or"
            for i, char in enumerate(spatial_chars):
                lhs += f",{char}r"
            self.equation = f"{lhs}->bo{spatial_chars}"
            
            self.weights = nn.ParameterList()
            
            # 1- Input Factor (In -> Rank)
            # Apply norm_weights logic locally
            scale_in = self._get_scale(in_channels, rank, norm_weights)
            self.weights.append(nn.Parameter(torch.randn(in_channels, rank, dtype=torch.cfloat) * scale_in))
            
            # 2- Output Factor (Rank -> Out)
            # Apply norm_weights logic locally
            scale_out = self._get_scale(rank, out_channels, norm_weights)
            self.weights.append(nn.Parameter(torch.randn(out_channels, rank, dtype=torch.cfloat) * scale_out))
            
            # 3- Spatial Factors (Mode -> Rank)
            # We keep these stable (1/sqrt(mode)) to avoid exploding the signal as dimensions increase
            for m in modes:
                scale_spatial = 1 / (m * rank)**0.5
                self.weights.append(nn.Parameter(torch.randn(m, rank, dtype=torch.cfloat) * scale_spatial))

        elif factorization == 'tucker':
            rank = int(rank)
            
            # --- 1. Dynamic Equation Building (Chain Method) ---
            # We split the operation into 4 steps to save memory:
            # Step 1: Input (bi...) -> Reduced Input (br...)
            # Step 2: Core (rs...)  -> Expanded Core (rs...) (Spatial expansion)
            # Step 3: Mix (br...) + (rs...) -> Reduced Output (bs...)
            # Step 4: Reduced Output (bs...) -> Final Output (bo...)
            
            s_chars = "xyztsruv"[:self.ndim] # Spatial: x, y, z...
            r_spat_chars = "mnpqkj"[:self.ndim] # Rank Spatial: m, n, p...
            
            r_in = 'a' # Rank In
            r_out = 'b' # Rank Out (Careful not to collide with Batch 'batch')
            # Actually, standard 'b' is batch. Let's use 'c' for Rank Out.
            r_out = 'c' 
            
            # Eq 1: Input Project: Batch(b), In(i), Spatial(...) -> Batch(b), RankIn(a), Spatial(...)
            self.tucker_eq_in = f"bi{s_chars},i{r_in}->b{r_in}{s_chars}"
            
            # Eq 2: Core Expand: RankIn(a), RankOut(c), RankSpat(m...)... + SpatialFactors(xm...) -> RankIn(a), RankOut(c), Spatial(x...)
            # Core: acmn...
            # Factors: xm, yn...
            lhs_core = f"{r_in}{r_out}{r_spat_chars}"
            for s, r in zip(s_chars, r_spat_chars):
                lhs_core += f",{s}{r}"
            self.tucker_eq_core = f"{lhs_core}->{r_in}{r_out}{s_chars}"
            
            # Eq 3: Middle Mix: Batch(b), RankIn(a), Spatial(...) + Core(a,c,...) -> Batch(b), RankOut(c), Spatial(...)
            self.tucker_eq_mid = f"b{r_in}{s_chars},{r_in}{r_out}{s_chars}->b{r_out}{s_chars}"
            
            # Eq 4: Output Project: Batch(b), RankOut(c), Spatial(...) -> Batch(b), Out(o), Spatial(...)
            self.tucker_eq_out = f"b{r_out}{s_chars},o{r_out}->bo{s_chars}"

            # --- 2. Initialize Weights ---
            self.weights = nn.ParameterList()
            factor_scale = 1 / (rank**0.5)
            
            # 0: In Factor (In, Rank)
            self.weights.append(nn.Parameter(torch.randn(in_channels, rank, dtype=torch.cfloat) * factor_scale))
            # 1: Out Factor (Out, Rank)
            self.weights.append(nn.Parameter(torch.randn(out_channels, rank, dtype=torch.cfloat) * factor_scale))
            # 2 to N+2: Spatial Factors (Mode, Rank)
            for m in modes:
                self.weights.append(nn.Parameter(torch.randn(m, rank, dtype=torch.cfloat) * factor_scale))
            
            # Last: Core Tensor (Rank, Rank, Rank...)
            core_shape = [rank] * (self.ndim + 2)
            main_scale = self._get_scale(in_channels, out_channels, norm_weights)
            self.weights.append(nn.Parameter(torch.randn(*core_shape, dtype=torch.cfloat) * main_scale))

        else: raise ValueError(f"Unknown factorization {factorization}, please use None, 'cp', or 'tucker'.")


        # 3- Requirements for FFT, einsum, and mode slicing
        # Figure out the dims for fft:
        self.fft_dims = list(range(-self.ndim, 0))

        # Einsum definition
        self.einsum = compl_mulnd(ndim=self.ndim)

        # Slicing defs
        self.slices= [slice(None), slice(None)] # Define the list for slicing, channels are not sliced, so defined as None
        for m in self.modes: self.slices.append(slice(0, m)) 
        self.slices = tuple(self.slices)


    def _get_scale(self, in_ch, out_ch, norm_weights):

        """
        "paper": The initialization found on the paper, causes the model to first improve on the local linear convolution 
        and add small spectral conv perturbations during training, since weights are insanely small initially

        "geom": Geometric scaling, which is a bit less aggressive than the paper scaling

        "xavier": Xavier initialization, which is a standard initialization for neural networks
        """

        if norm_weights == "paper": return 1 / (in_ch * out_ch)
        elif norm_weights == "geom": return 1 / (in_ch * out_ch)**0.5
        elif norm_weights == "xavier": return (2 / (in_ch + out_ch))**0.5
        else: raise ValueError(f"Unknown norm_weights {norm_weights}, please use 'paper', 'geom', or 'xavier'.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes in a signal x (n-D), applies rfft, performs spectral linear transform on selected modes, returns irfft  
        """
        signal_size = x.shape[-self.ndim:]

        # 1- FFT
        x_ft = torch.fft.rfftn(x, dim=self.fft_dims, norm=self.norm_fft)

        # 2- Dynamic Multiplication Logic
        if isinstance(self.weights, nn.ParameterList):
            # --- FACTORIZED PATH ---
            # Check if Tucker (by checking if we have the specific equations)
            if hasattr(self, 'tucker_eq_in'):
                # --- TUCKER CHAIN ---
                # Weights: [In(0), Out(1), Spat1(2), Spat2(3)..., Core(-1)]
                w_in = self.weights[0]
                w_out = self.weights[1]
                w_spatial = [(w) for w in self.weights[2:-1]]
                w_core = self.weights[-1]
                
                # 1- Reduce Input Channels (B, I, X, Y) -> (B, Rin, X, Y)
                x_reduced = torch.einsum(self.tucker_eq_in, x_ft[self.slices], w_in)
                
                # 2- Expand Core (Rin, Rout, Rm, Rn) -> (Rin, Rout, X, Y)
                # This mixes the small rank modes with the spatial modes
                core_expanded = torch.einsum(self.tucker_eq_core, w_core, *w_spatial)
                
                # 3- Convolve (Batch, Rin, X, Y) * (Rin, Rout, X, Y) -> (Batch, Rout, X, Y)
                x_middle = torch.einsum(self.tucker_eq_mid, x_reduced, core_expanded)
                
                # 4- Expand Output Channels (Batch, Rout, X, Y) -> (Batch, Out, X, Y)
                res = torch.einsum(self.tucker_eq_out, x_middle, w_out)
                
            else:
                # --- CP CHAIN (Simple) ---
                args = [x_ft[self.slices]] + [(w) for w in self.weights]
                res = torch.einsum(self.equation, *args)
            
            # Prepare output
            out_ft_shape = list(x_ft.shape)
            out_ft_shape[1] = self.out_channels
            out_ft = torch.zeros(size=out_ft_shape, dtype=x_ft.dtype, device=x.device)
            out_ft[self.slices] = res
            
        else:
            # --- DENSE PATH ---
            out_ft_shape = list(x_ft.shape)
            out_ft_shape[1] = self.out_channels
            out_ft = torch.zeros(size=out_ft_shape, dtype=x_ft.dtype, device=x.device)
            out_ft[self.slices] = self.einsum(x_ft[self.slices], self.weights)

        # 3- Inverse FFT
        x = torch.fft.irfftn(out_ft, s=signal_size, dim=self.fft_dims, norm=self.norm_fft)

        return x
#endregion