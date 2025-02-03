import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Create random embeddings
video_embeddings = torch.rand(4, 20, 128)
text_embeddings = torch.rand(4, 8, 128)

# Define the logit scale parameter
logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

class ContrastiveLoss(nn.Module):
    def __init__(self, logit_scale):
        super().__init__()
        self.logit_scale = logit_scale

    def forward(self, video_embeddings, text_embeddings):
        """
        Calculates the contrastive loss between video and text embeddings.

        Args:
            video_embeddings (torch.Tensor): Tensor of shape [B, N_v, D]
            text_embeddings (torch.Tensor): Tensor of shape [B, N_t, D]

        Returns:
            torch.Tensor: A scalar tensor representing the contrastive loss
        """
        video_embeddings = F.normalize(video_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        # Average pooling over tokens
        video_embeddings = video_embeddings.mean(dim=1)
        text_embeddings = text_embeddings.mean(dim=1)
        
        similarity = torch.matmul(video_embeddings, text_embeddings.transpose(-2, -1))
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * similarity
        
        # Calculate loss
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_video = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.t(), labels)
        breakpoint()
        
        # Return average loss
        return (loss_video + loss_text) / 2

# Create an instance of the ContrastiveLoss
criterion = ContrastiveLoss(logit_scale)

# Calculate the contrastive loss
loss = criterion(video_embeddings, text_embeddings)

print(f"Contrastive Loss: {loss.item():.4f}")