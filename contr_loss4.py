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
        Compute the similarity matrix between sign and text embeddings.
        
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
        sign_proj = F.normalize(sign_proj, dim=-1)
        text_proj = F.normalize(text_proj, dim=-1)

        # Compute fine-grained similarity matrix Z
        batch_size = sign_proj.size(0)
        similarity_matrix = torch.zeros(batch_size, batch_size).to(sign_embeddings.device)

        for i in range(batch_size):
            for j in range(batch_size):
                # Compute pairwise similarity between all tokens in sign and text sequences
                Z = torch.matmul(sign_proj[i], text_proj[j].transpose(0, 1))  # Shape: (seq_len_sign, seq_len_text)
                Z_softmax = F.softmax(Z, dim=-1)  # Apply softmax row-wise
                reweighted_Z = torch.matmul(Z_softmax, Z.transpose(0, 1))   # Re-weighted similarity matrix
                global_similarity = reweighted_Z.sum() / reweighted_Z.numel()  # Average over all elements
                similarity_matrix[i, j] = global_similarity

        return similarity_matrix

    def compute_contrastive_loss(self, similarity_matrix):
        """
        Compute the contrastive loss using InfoNCE.
        
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