def contrastive_loss_split(self, hand_sign_embeddings, non_hand_sign_embeddings, text_embeddings):
        """
        Calculates separate contrastive losses for hand sign embeddings and non-hand sign embeddings with text embeddings.

        Args:
            hand_sign_embeddings (torch.Tensor): Tensor of shape [B, N_h, D] where:
                B is the batch size
                N_h is the number of hand sign tokens
                D is the embedding dimension.
            non_hand_sign_embeddings (torch.Tensor): Tensor of shape [B, N_nh, D] where:
                B is the batch size
                N_nh is the number of non-hand sign tokens
                D is the embedding dimension.
            text_embeddings (torch.Tensor): Tensor of shape [B, N_t, D] where:
                B is the batch size
                N_t is the number of text tokens
                D is the embedding dimension.

        Returns:
            tuple: Two scalar tensors representing the contrastive loss for hand sign embeddings and non-hand sign embeddings.
        """

        # Project and normalize hand sign embeddings
        hand_sign_proj = self.hparams.projection_sign(hand_sign_embeddings)
        hand_sign_proj = F.normalize(hand_sign_proj, dim=-1)

        # Project and normalize non-hand sign embeddings
        non_hand_sign_proj = self.hparams.projection_sign(non_hand_sign_embeddings)
        non_hand_sign_proj = F.normalize(non_hand_sign_proj, dim=-1)

        # Project and normalize text embeddings
        text_proj = self.hparams.projection_text(text_embeddings)
        text_proj = F.normalize(text_proj, dim=-1)

        # Compute similarity matrices for hand-sign and non-hand-sign separately
        Z_hand = torch.matmul(hand_sign_proj, text_proj.transpose(2, 1))  # [B, N_h, N_t]
        Z_non_hand = torch.matmul(non_hand_sign_proj, text_proj.transpose(2, 1))  # [B, N_nh, N_t]

        # Apply softmax and reweighting for hand-sign embeddings
        Z_hand_softmax = F.softmax(Z_hand, dim=-1)  # [B, N_h, N_t]
        Z_hand_reweighted = torch.matmul(Z_hand_softmax, Z_hand.transpose(2, 1))  # [B, N_h, N_h]

        global_similarity_hand = Z_hand_reweighted.mean(dim=(1, 2))  # [B]
        similarity_matrix_hand = global_similarity_hand.unsqueeze(0) - global_similarity_hand.unsqueeze(1)  # [B, B]

        # Apply softmax and reweighting for non-hand-sign embeddings
        Z_non_hand_softmax = F.softmax(Z_non_hand, dim=-1)  # [B, N_nh, N_t]
        Z_non_hand_reweighted = torch.matmul(Z_non_hand_softmax, Z_non_hand.transpose(2, 1))  # [B, N_nh, N_nh]

        global_similarity_non_hand = Z_non_hand_reweighted.mean(dim=(1, 2))  # [B]
        similarity_matrix_non_hand = global_similarity_non_hand.unsqueeze(0) - global_similarity_non_hand.unsqueeze(1)  # [B, B]

        batch_size = similarity_matrix_hand.size(0)

        logits_hand = similarity_matrix_hand / self.hparams.temperature
        logits_non_hand = similarity_matrix_non_hand / self.hparams.temperature

        labels = torch.arange(batch_size).to(similarity_matrix_hand.device)

        # Compute losses for hand-sign-to-text and text-to-hand-sign directions
        loss_s2t_hand = F.cross_entropy(logits_hand, labels)
        loss_t2s_hand = F.cross_entropy(logits_hand.transpose(0, 1), labels)
        
        loss_s2t_non_hand = F.cross_entropy(logits_non_hand, labels)
        loss_t2s_non_hand = F.cross_entropy(logits_non_hand.transpose(0, 1), labels)

        # Return separate contrastive losses
        loss_hand = (loss_s2t_hand + loss_t2s_hand) / 2.0
        loss_non_hand = (loss_s2t_non_hand + loss_t2s_non_hand) / 2.0

        return loss_hand, loss_non_hand