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