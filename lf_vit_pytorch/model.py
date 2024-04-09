import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as cp

from einops import einsum, rearrange
from einops.layers.torch import Rearrange

from typing import Optional

class Config():
    """
    LF-ViT Configuration class

    Args:
        n_labels (int, defaults to 1)
            Defines the number of different classes/labels for images.

        downscale_ratio (int, defaults to 4)
            Image downscale ratio, it will decide the latent image size.

        patch_dim (int, defaults to 16)
            Defines the fixed-size patch dimension on which to split the image.

        d_model (int, defaults to 512)
            Dimension of the patch embedding representation.

        depth (int, defaults to 1)
            Number of blocks the model built upon.

        ffn_dim_mult (int, defaults to 4)
            SwiGLU hidden dimension multiplier (defaults to d_model * ffn_dim_mult)

        n_heads (int, defaults to 8)
            Number of heads in the CrossAttention.

        norm_eps (float, defaults to 1e-5)
            Small value to handle numerical stability (prevents division with zero).
    """

    def __init__(self,
                 n_labels: int = 1,
                 downscale_ratio: int = 4,
                 patch_dim: int = 16,
                 d_model: int = 512,
                 depth: int = 1,
                 ffn_dim_mult: int = 8,
                 n_heads: int = 8,
                 norm_eps: float = 1e-5):
        
        self.n_labels = n_labels
        self.downscale_ratio = downscale_ratio
        self.patch_dim = patch_dim
        self.d_model = d_model
        self.depth = depth
        self.ffn_dim_mult = ffn_dim_mult
        self.n_heads = n_heads
        self.norm_eps = norm_eps

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm)
    https://arxiv.org/pdf/1910.07467.pdf

    A well-known explanation of the success of LayerNorm is its re-centering
    and re-scaling invariance property. However RMSNorm only focuses on
    re-scaling invariance and regularizes the summed inputs simply according
    to the root mean square statistic.

    Intuitively, RMSNorm simplifies LayerNorm by totally removing the
    mean statistic at the cost of sacrificing the invariance that mean
    normalization affords.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    References:
        https://github.com/facebookresearch/llama (Credit)
    """

    def __init__(self, dim: int, eps: Optional[float] = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).to(x.dtype)
        return self.scale * x

class FFT2D(nn.Module):
    """
    Fast-Fourier Transform 2D (FFT2D)
    https://arxiv.org/pdf/2105.03824.pdf

    Decomposes a function into its constituent frequencies. Applies a
    2D FFT to the embeddings vectors (batch_size, seq_len, emb_dim).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.fft.fft(torch.fft.fft(x, dim = -1), dim = -2).real

class XAttn(nn.Module):
    """
    Cross Attention (XAttn)

    Cross-Attention mixes or combines asymmetrically two separate embedding
    sequence. One of the sequences serves as a query input, while the other
    as a key and value inputs.

    References:
        https://vaclavkosar.com/ml/cross-attention-in-transformer-architecture (Description)
    """

    def __init__(self,
                 dim: int,
                 other_dim: int,
                 n_heads: int):
        super().__init__()
        self.head_dim = other_dim // n_heads
        self.n_heads = n_heads

        self.q_proj = nn.Linear(dim, self.head_dim * n_heads, bias = False)
        self.k_proj = nn.Linear(other_dim, self.head_dim * n_heads, bias = False)
        self.v_proj = nn.Linear(other_dim, self.head_dim * n_heads, bias = False)
        self.o_proj = nn.Linear(self.head_dim * n_heads, dim, bias = False)

    def forward(self, x1, x2):
        q, k, v = self.q_proj(x1), self.k_proj(x2), self.v_proj(x2)

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.n_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.n_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.n_heads)

        scores = einsum(q, k, 'b h n d, b h m d -> b h n m')
        attention = F.softmax(scores / math.sqrt(self.head_dim), dim = -1)

        o = einsum(attention, v, 'b h n m, b h m d -> b h n d')
        o = rearrange(o, 'b h n d -> b n (h d)')

        return self.o_proj(o)

class SwiGLU(nn.Module):
    """
    Swish Gated Linear Unit (SwiGLU)
    https://arxiv.org/pdf/2002.05202v1.pdf

    This can be through of as a multiplicative skip connection
    which helps gradients flow across the layers.

    Args:
        dim (int): Input dimension size.
        exp_factor (float): Hidden dimension multiplier. Defaults to 2.
    """

    def __init__(self,
                 dim: int,
                 exp_factor: Optional[int] = 2):
        super().__init__()
        self.dim = dim
        self.exp_factor = exp_factor

        hidden_dim = dim * exp_factor
        self.uv_proj = nn.Linear(dim, hidden_dim * 2, bias = False)
        self.out_proj = nn.Linear(hidden_dim, dim, bias = False)
        self.act = nn.SiLU()

    def forward(self, x):
        u, v = torch.chunk(self.uv_proj(x), 2, dim = -1)
        return self.out_proj(self.act(u) * v)

class LFViTBlock(nn.Module):
    """
    Latent Fourier Vision Transformer Block (LF-ViT Block)

    A block built upon the the listed modules below:
        1. FFT2D
        2. Cross-Attention
        3. SwiGLU

    Each with Residual Connections.
    (2. and 3. with Pre-normalized with RMSNorm)
    """
    
    def __init__(self, config: Config):
        super().__init__()

        ## This will be the downscaled embedding dimension.
        dim = config.d_model // config.downscale_ratio

        self.fft = FFT2D()
        self.attn = XAttn(
            dim = dim,
            other_dim = config.d_model,
            n_heads = config.n_heads
        )
        self.ffn = SwiGLU(
            dim = dim,
            exp_factor = config.ffn_dim_mult
        )

        self.attn_norm = nn.LayerNorm(dim, eps = config.norm_eps)
        self.cross_norm = nn.LayerNorm(config.d_model, eps = config.norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps = config.norm_eps)

    def forward(self, x: torch.Tensor, original: torch.Tensor):
        x = self.fft(x)
        print('FFT: ')
        print(x)
        
        x_norm = self.attn_norm(x)
        cross_x = self.cross_norm(original)
        x = self.attn(x_norm, cross_x) + x
        print('Attn: ')
        print(x)
        
        x_norm = self.ffn_norm(x)
        x = self.ffn(x_norm) + x
        print('GLU: ')
        print(x)

        return x

class LFViT(nn.Module):
    """
    Latent Fourier Vision Transformer (LFViT)

    LF-ViT combines the idea of the latent representation form the paper
    PerceiverIO (https://arxiv.org/pdf/2107.14795.pdf) and FNET's 2D FFT
    to replace Self-Attention.

    The original image is downscaled (scaled_image) and processed by the
    model and when it reaches the Cross-Attention will be fused with the
    original non-scaled image embeddings. This way we eliminate the need
    of huge embedding vectors and enable a significant inference speed up.

    Args:
        pixel_values (torch.Tensor of shape (batch_size, num_channels, height, width))
            Pixel value of the original from the image_preprocessor.
            
        scaled_pixel_values (torch.Tensor of shape (batch_size, num_channels, height // ratio, width // ratio))
            Pixel values of the downscaled image from the image processor.

    Usage:
        model = ...
        
        pixel_values = image_processor(image, ...)['pixel_values']
        scaled_pixel_values = image_processor(image, size = {'height': 224 // ratio, 'width': 224 // ratio, ...})['pixel_values']

        y = model(pixel_values, scaled_pixel_values)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.model_config = config
        self.n_labels = config.n_labels
        self.depth = config.depth

        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = config.patch_dim, p2 = config.patch_dim),
            nn.LayerNorm(config.patch_dim * config.patch_dim * 3),
            nn.Linear(config.patch_dim * config.patch_dim * 3, config.d_model),
            nn.LayerNorm(config.d_model)
        )

        latent_patch_dim = config.patch_dim // config.downscale_ratio
        latent_dim = config.d_model // config.downscale_ratio
        self.to_latent = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = latent_patch_dim, p2 = latent_patch_dim),
            nn.LayerNorm(latent_patch_dim * latent_patch_dim * 3),
            nn.Linear(latent_patch_dim * latent_patch_dim * 3, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        self.blocks = nn.ModuleList([ LFViTBlock(config) for _ in range(config.depth) ])
        self.norm = nn.LayerNorm(config.d_model // config.downscale_ratio)
        self.to_labels = nn.Linear(config.d_model // config.downscale_ratio, config.n_labels, bias = False)

        self.gradient_checkpointing = False

    def forward(self,
                pixel_values: torch.Tensor,
                scaled_pixel_values: torch.Tensor):
                    
        original = self.to_patch(pixel_values)
        x = self.to_latent(scaled_pixel_values)

        for block in self.blocks:
            if self.gradient_checkpointing:
                x = cp(block, x, original, use_reentrant = False)
            else:
                x = block(x, original)
            
        x = self.norm(x)
        x = x.mean(dim = -2)
        x = self.to_labels(x)

        return x
