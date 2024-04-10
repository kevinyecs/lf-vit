import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class PrepareForTrainer(nn.Module):
    """
    To make the model compatible with HuggingFace's Trainer the model should
    produce a dictionary containing the 'logits' and 'loss'

    This class implements a custom loss computation and forward pass.
    Also doesn't require to pass the down scaled images as the model wants,
    the resize is happens here in the forward pass with interpolation.

    For interpolation we use 'torch.nn.functional.interpolate' and the mode
    can be changed through 'model.interpolate_mode' attribute.

    Args:
        model (nn.Module):
            LF-ViT model

        classification_type (string):
            Defines the classification type which can be (binary, multiclass)

        interpolate_mode (string):
            Defines the interpolation mode
            For further detailes check:
            https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.interpolate_mode = 'nearest'

        if model.n_labels == 1:
            self.classification_type = 'binary'
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif model.n_labels > 1:
            self.classification_type = 'multiclass'
            self.loss_fn = nn.CrossEntropyLoss()

    def compute_loss(self, logits, labels):
        if self.classification_type == 'binary':  
            return self.loss_fn(logits, labels.float())
        elif self.classification_type == 'multiclass':
            return self.loss_fn(logits.view(-1, self.model.n_labels), labels.view(-1))

    def print_trainable_parameters(self):
        trainable_params, all_params = 0, 0
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f'trainable params: {trainable_params:,} | all params: {all_params:,} | trainable: {100 * trainable_params / all_params}%')
    
    def forward(self,
                pixel_values: torch.Tensor,
                labels: torch.Tensor):

        scale_ratio = self.model.model_config.downscale_ratio
        scaled_shape = (pixel_values.shape[-2] // scale_ratio, pixel_values.shape[-1] // scale_ratio)
        scaled_pixel_values = F.interpolate(pixel_values, size = scaled_shape, mode = self.interpolate_mode)
        
        logits = self.model(pixel_values, scaled_pixel_values)
        return {
            'logits': logits,
            'loss': self.compute_loss(logits.float(), labels)
        }
