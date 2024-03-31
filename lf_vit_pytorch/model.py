import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class Config():
    """
    LF-ViT Configuration class

    Args:
        n_classes (int, defaults to 1)
            Defines the number of different classes for images.

        downscale_ratio (int, defaults to 4)
            Image downscale ratio, it will decide the latent image size.

        d_model (int, defaults to 512)
            Dimension of the patch embedding representation.

        depth (int, defaults to 1)
            Number of blocks the model built upon.

        ffn_dim_mult (int, defaults to 2)
            SwiGLU hidden dimension multiplier (defaults to d_model * ffn_dim_mult)

        n_heads (int, defaults to 8)
            Number of heads in the CrossAttention.

        norm_eps (float, defaults to 1e-5)
            Small value to handle numerical stability (prevents division with zero).
    """

    def __init__(self,
                 n_classes: int = 1,
                 downscale_ratio: int = 4,
                 d_model: int = 512,
                 depth: int = 1,
                 ffn_dim_mult: int = 2,
                 n_heads: int = 8,
                 norm_eps: float = 1e-5):
        
        self.n_classes = n_classes
        self.downscale_ratio = downscale_ratio
        self.d_model = d_model
        self.depth = depth
        self.ffn_dim_mult = ffn_dim_mult
        self.n_heads = n_heads
        self.norm_eps = norm_eps

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
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

    Description

    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.fft.fft(torch.fft.fft(x, dim = -1), dim = -2).real

class XAttn(nn.Module):
    """
    Cross Attention (XAttn)

    Description
    """

    def __init__(self,
                 dim: int,
                 n_heads: int):
        super().__init__()
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, self.head_dim * n_heads, bias = True)
        self.k_proj = nn.Linear(dim, self.head_dim * n_heads, bias = True)
        self.v_proj = nn.Linear(dim, self.head_dim * n_heads, bias = True)
        self.o_proj = nn.Linear(self.head_dim * n_heads, dim, bias = True)

    def forward(self, x1, x2):
        q, k, v = self.q_proj(x1), self.k_proj(x2), self.v_proj(x2)

        q = rearrange(q, 'b n h d -> b h n d', h = self.head_dim)
        k = rearrange(k, 'b n h d -> b h n d', h = self.head_dim)
        v = rearrange(v, 'b n h d -> b h n d', h = self.head_dim)

        scores = einsum(q, k, 'b h n d, b h m d -> b h n m')
        attention = F.softmax(scores / math.sqrt(self.head_dim), dim = -1)

        o = einsum(attention, v, 'b h n d, b h n m -> b h m d')
        o = rearrange(o, 'b h n d -> b n h d')

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

class TransformerBlock(nn.Module):
    """
    LF-ViT Block

    Description
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.d_model

        self.fft = FFT2D()
        self.attn = XAttn(
            dim = config.d_model,
            n_heads = config.n_heads
        )
        self.ffn = SwiGLU(
            dim = config.d_model,
            exp_factor = config.ffn_dim_mult
        )

        self.attn_norm = RMSNorm(config.d_model, eps = config.norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps = config.norm_eps)

    def forward(self, x: torch.Tensor):
        x = self.fft(x)

        x_norm = self.attn_norm(x)
        x = self.attn(x_norm) + x

        x_norm = self.ffn_norm(x)
        x = self.ffn(x_norm) + x

        return x
