def contrastive_loss(self, video_embeddings, text_embeddings):
        """
        Calculates the contrastive loss between video and text embeddings.

        Args:
            video_embeddings (torch.Tensor): Tensor of shape [B, N_v, D] where: torch.Size([1, 50, 2048])
                B is the batch size
                N_v is the number of video tokens
                D is the embedding dimension
            text_embeddings (torch.Tensor): Tensor of shape [B, N_t, D] where: (torch.Size([1, 8, 2048]))
                B is the batch size
                N_t is the number of text tokens
                D is the embedding dimension

        Returns:
            torch.Tensor: A scalar tensor representing the contrastive loss

        Note:
            This method assumes that the video and text embeddings are from the same batch,
        """
        video_embeddings = F.normalize(video_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        similarity = torch.matmul(video_embeddings, text_embeddings.transpose(-2, -1))
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * similarity
        
        # Calculate loss
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_video = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.t(), labels)
        
        # Return average loss
        return (loss_video + loss_text) / 2