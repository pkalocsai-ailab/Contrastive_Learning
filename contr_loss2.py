import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([temperature]))
        self.sign_projector = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        self.text_projector = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))

    def forward(self, sign_embeddings, text_embeddings):
        # Project embeddings
        Fsign = self.sign_projector(F.avg_pool2d(sign_embeddings, kernel_size=(4, 1)).squeeze(2))
        Ftext = self.text_projector(text_embeddings)

        # Normalize embeddings
        Fsign = F.normalize(Fsign, p=2, dim=-1)
        Ftext = F.normalize(Ftext, p=2, dim=-1)

        # Calculate similarity matrix Z
        Z = torch.matmul(Fsign, Ftext.transpose(-2, -1))

        # Apply softmax to each row of Z
        Z_softmax = F.softmax(Z, dim=-1)

        # Re-weight Z
        Z_hat = torch.matmul(Z_softmax, Z.transpose(-2, -1))

        # Calculate global similarity
        M = Z_hat.sum(dim=1).mean(dim=1)

        # Reshape to batch size x batch size
        M = M.view(sign_embeddings.size(0), -1)

        # Calculate InfoNCE loss
        labels = torch.arange(M.size(0), device=M.device)
        loss_i = -torch.log(torch.exp(M.diag() / self.temperature) / torch.exp(M / self.temperature).sum(dim=1))
        loss_t = -torch.log(torch.exp(M.diag() / self.temperature) / torch.exp(M / self.temperature).sum(dim=0))
        loss = (loss_i.mean() + loss_t.mean()) / 2

        return loss

# Create random embeddings
sign_embeddings = torch.rand(4, 20, 128)
text_embeddings = torch.rand(4, 8, 128)

# Initialize the loss function
criterion = ContrastiveLoss()

# Calculate the loss
loss = criterion(sign_embeddings, text_embeddings)

print(f"Contrastive Loss: {loss.item():.4f}")