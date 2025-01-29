import torch
import torch.nn as nn
import torch.nn.functional as F

class FineGrainedSignTextSimilarity:
    def __init__(self, embedding_dim, temperature=0.07):
        """
        Initialize the Fine-Grained Sign-Text Similarity module.
        
        Args:
            embedding_dim (int): Dimension of the shared embedding space.
            temperature (float): Temperature coefficient for contrastive loss.
        """
        self.temperature = temperature
        self.projection_sign = nn.Linear(embedding_dim, embedding_dim)
        self.projection_text = nn.Linear(embedding_dim, embedding_dim)

    def compute_similarity_matrix(self, sign_embeddings, text_embeddings):
        """
        Compute the similarity matrix between sign and text embeddings without loops.
        
        Args:
            sign_embeddings (torch.Tensor): Sign embeddings of shape (batch_size, seq_len_sign, embedding_dim).
            text_embeddings (torch.Tensor): Text embeddings of shape (batch_size, seq_len_text, embedding_dim).
        
        Returns:
            torch.Tensor: Similarity matrix of shape (batch_size, batch_size).
        """
        # Project embeddings into shared latent space
        sign_proj = self.projection_sign(sign_embeddings)  # Shape: (batch_size, seq_len_sign, embedding_dim)
        text_proj = self.projection_text(text_embeddings)  # Shape: (batch_size, seq_len_text, embedding_dim)

        # Normalize embeddings
        sign_proj = F.normalize(sign_proj, dim=-1)  # Shape: (batch_size, seq_len_sign, embedding_dim)
        text_proj = F.normalize(text_proj, dim=-1)  # Shape: (batch_size, seq_len_text, embedding_dim)

        # Compute pairwise token similarities for all samples in the batch
        Z = torch.matmul(sign_proj, text_proj.transpose(2, 1))  # Shape: (batch_size, seq_len_sign, seq_len_text)

        # Apply softmax row-wise for each sample in the batch
        Z_softmax = F.softmax(Z, dim=-1)  # Shape: (batch_size, seq_len_sign, seq_len_text)

        # Re-weighted similarity matrix
        Z_reweighted = torch.matmul(Z_softmax, Z.transpose(2, 1))  # Shape: (batch_size, seq_len_sign, seq_len_sign)

        # Global similarity score for each pair of samples in the batch
        global_similarity = Z_reweighted.mean(dim=(1, 2))  # Shape: (batch_size,)
        
        # Compute similarity matrix for all pairs in the batch
        similarity_matrix = global_similarity.unsqueeze(0) - global_similarity.unsqueeze(1)
        
        return similarity_matrix

    def compute_contrastive_loss(self, similarity_matrix):
        """
        Compute contrastive loss using InfoNCE.
        
        Args:
            similarity_matrix (torch.Tensor): Similarity matrix of shape (batch_size, batch_size).
        
        Returns:
            torch.Tensor: Contrastive loss value.
        """
        batch_size = similarity_matrix.size(0)
        
        # Apply temperature scaling and softmax normalization
        logits = similarity_matrix / self.temperature
        labels = torch.arange(batch_size).to(similarity_matrix.device)

        # Compute InfoNCE loss for both directions (sign-to-text and text-to-sign)
        loss_s2t = F.cross_entropy(logits, labels)
        loss_t2s = F.cross_entropy(logits.transpose(0, 1), labels)

        # Total loss is the average of both directions
        total_loss = (loss_s2t + loss_t2s) / 2.0
        return total_loss

# Example Usage
if __name__ == "__main__":
    # Example input data: batch size = 4, sequence lengths vary for sign and text
    batch_size = 4
    seq_len_sign = 10
    seq_len_text = 8
    embedding_dim = 512

    # Randomly generated embeddings for demonstration purposes
    sign_embeddings = torch.rand(batch_size, seq_len_sign, embedding_dim)
    text_embeddings = torch.rand(batch_size, seq_len_text, embedding_dim)

    # Initialize the module and compute loss
    fgsts_module = FineGrainedSignTextSimilarity(embedding_dim=embedding_dim)
    similarity_matrix = fgsts_module.compute_similarity_matrix(sign_embeddings, text_embeddings)
    contrastive_loss = fgsts_module.compute_contrastive_loss(similarity_matrix)

    print("Similarity Matrix:\n", similarity_matrix)
    print("Contrastive Loss:", contrastive_loss.item())