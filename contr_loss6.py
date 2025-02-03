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

    def compute_contrastive_loss(self, sign_embeddings, text_embeddings):
        """
        Compute the similarity matrix between sign and text embeddings without loops.
        
        Args:
            sign_embeddings (torch.Tensor): Sign embeddings of shape (batch_size, seq_len_sign, embedding_dim).
            text_embeddings (torch.Tensor): Text embeddings of shape (batch_size, seq_len_text, embedding_dim).
        
        Returns:
            torch.Tensor: Similarity matrix of shape (batch_size, batch_size).
        """
        sign_proj = self.projection_sign(sign_embeddings)  
        text_proj = self.projection_text(text_embeddings)  

        sign_proj = F.normalize(sign_proj, dim=-1)  
        text_proj = F.normalize(text_proj, dim=-1)  

        Z = torch.matmul(sign_proj, text_proj.transpose(2, 1)) 
        Z_softmax = F.softmax(Z, dim=-1)  
        Z_reweighted = torch.matmul(Z_softmax, Z.transpose(2, 1)) 

        global_similarity = Z_reweighted.mean(dim=(1, 2)) 
        similarity_matrix = global_similarity.unsqueeze(0) - global_similarity.unsqueeze(1)
        
        batch_size = similarity_matrix.size(0)
        
        logits = similarity_matrix / self.temperature
        labels = torch.arange(batch_size).to(similarity_matrix.device)

        loss_s2t = F.cross_entropy(logits, labels)
        loss_t2s = F.cross_entropy(logits.transpose(0, 1), labels)

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
    contrastive_loss = fgsts_module.compute_contrastive_loss(sign_embeddings, text_embeddings)

    #print("Similarity Matrix:\n", similarity_matrix)
    print("Contrastive Loss:", contrastive_loss.item())