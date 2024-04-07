import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List

class PrepareForTrainer(nn.Module):
    """
    To make the model compatible with HuggingFace's Trainer the model should
    produce a dictionary containing the 'logits' and 'loss'.

    This class implements a custom loss computation and forward pass.
    Also doesn't require any 'label' or 'attention_mask' columns, if any of
    them passed as argument the model will ignore it.
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
