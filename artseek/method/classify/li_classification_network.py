import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateInteractionClassificationNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_tasks,
        activation="relu",
        num_encoder_layers=2,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        output_dim=None,
        single_encoder=False,
        single_embedding=False,
        single_projection=False,
    ):
        """
        Initialize the Late Interaction Classification Network.

        Args:
            embedding_dim: Dimension of input embeddings
            num_tasks: Number of classification tasks
            activation: Activation function to use  (relu, gelu, etc.)
            num_encoder_layers: Number of transformer encoder layers
            nhead: Number of attention heads in transformer
            dim_feedforward: Dimension of feedforward network in transformer
            dropout: Dropout rate
            output_dim: Output dimension for text task embeddings (None means no projection)
            single_encoder: If True, use a single transformer encoder for both text and visual modalities
            single_embedding: If True, use a single embedding for both text and visual modalities
            single_projection: If True, use a single projection for both text and visual modalities
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_tasks = num_tasks
        self.output_dim = output_dim or embedding_dim

        # Create learnable task embeddings
        self.task_embeddings_text = nn.Parameter(torch.zeros(num_tasks, embedding_dim))
        nn.init.normal_(self.task_embeddings_text, std=0.02)
        if single_embedding:
            self.task_embeddings_visual = self.task_embeddings_text
        else:
            self.task_embeddings_visual = nn.Parameter(
                torch.zeros(num_tasks, embedding_dim)
            )
            nn.init.normal_(self.task_embeddings_visual, std=0.02)

        # Create transformer encoders for each modality
        encoder_layer_text = nn.TransformerEncoderLayer(
            activation=activation,
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(
            encoder_layer_text,
            num_layers=num_encoder_layers,
        )

        # If single_encoder is True, use the same encoder for both text and visual
        if single_encoder:
            encoder_layer_visual = encoder_layer_text
            self.visual_encoder = nn.TransformerEncoder(
                encoder_layer_visual,
                num_layers=num_encoder_layers,
            )
        # Otherwise, create a separate encoder for visual modality
        else:
            encoder_layer_visual = nn.TransformerEncoderLayer(
                activation=activation,
                d_model=embedding_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.visual_encoder = nn.TransformerEncoder(
                encoder_layer_visual,
                num_layers=num_encoder_layers,
            )

        # Create linear projection for text task embeddings
        self.text_projection = nn.Linear(embedding_dim, self.output_dim, bias=False)
        if single_projection:
            self.visual_projection = self.text_projection
        else:
            self.visual_projection = nn.Linear(
                embedding_dim, self.output_dim, bias=False
            )

    def forward(
        self,
        text_embeddings=None,
        visual_embeddings=None,
        text_mask=None,
        visual_mask=None,
        task_idx=0,
    ):
        """
        Forward pass of the Late Interaction Classification Network.

        Args:
            text_embeddings: Tensor of shape (N, D) or (batch_size, N, D) or None
            visual_embeddings: Tensor of shape (M, D) or (batch_size, M, D) or None
            text_mask: Tensor of shape (batch_size, N) or None
            visual_mask: Tensor of shape (batch_size, M) or None
            task_idx: Index of the classification task to use

        Returns:
            Dictionary containing task embeddings for each provided modality
        """
        results = {}

        # Check that at least one modality is provided
        if text_embeddings is None and visual_embeddings is None:
            raise ValueError(
                "At least one of text_embeddings or visual_embeddings must be provided"
            )
        # Check that if masks are provided, they are of the correct shape
        if text_mask is not None and text_mask.shape != text_embeddings.shape[:2]:
            raise ValueError("text_mask must have shape (batch_size, N)")
        if visual_mask is not None and visual_mask.shape != visual_embeddings.shape[:2]:
            raise ValueError("visual_mask must have shape (batch_size, M)")
        # If masks are not provided, create them with all ones
        if text_embeddings is not None and text_mask is None:
            text_mask = torch.ones(text_embeddings.shape[:2], dtype=torch.bool).to(text_embeddings.device)
        if visual_embeddings is not None and visual_mask is None:
            visual_mask = torch.ones(visual_embeddings.shape[:2], dtype=torch.bool).to(visual_embeddings.device)

        # Get the task embedding for the specified task (only for text)
        task_emb_text = self.task_embeddings_text[task_idx]  # Shape: (D)

        # Process text embeddings if provided
        if text_embeddings is not None:
            # Add batch dimension if not present
            if text_embeddings.dim() == 2:
                text_embeddings = text_embeddings.unsqueeze(0)

            batch_size = text_embeddings.shape[0]

            # Create task embeddings for each item in batch
            batch_task_embs = (
                task_emb_text.unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
            )
            batch_task_mask = torch.ones(batch_size, 1, dtype=torch.bool).to(
                text_mask.device
            )

            # Prepend task embedding to text embeddings
            text_input = torch.cat([batch_task_embs, text_embeddings], dim=1)
            text_mask = torch.cat([batch_task_mask, text_mask], dim=1)

            # Process through text encoder
            text_output = self.text_encoder(text_input, src_key_padding_mask=~text_mask)

            # Extract task token outputs (first token of each sequence in batch)
            # Apply linear projection to increase dimension
            results["text_task_embeddings"] = self.text_projection(text_output[:, 0])

        # Process visual embeddings if provided
        if visual_embeddings is not None:
            # Add batch dimension if not present
            if visual_embeddings.dim() == 2:
                visual_embeddings = visual_embeddings.unsqueeze(0)

            batch_size = visual_embeddings.shape[0]

            # Create task embeddings for each item in batch
            batch_task_embs = self.task_embeddings_visual.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            batch_task_mask = torch.ones(
                batch_size, self.num_tasks, dtype=torch.bool
            ).to(visual_mask.device)

            # Prepend task embedding to visual embeddings
            visual_input = torch.cat([batch_task_embs, visual_embeddings], dim=1)
            visual_mask = torch.cat([batch_task_mask, visual_mask], dim=1)

            # Process through visual encoder
            visual_output = self.visual_encoder(
                src=visual_input, src_key_padding_mask=~visual_mask
            )

            # Extract task token outputs (first tokens of each sequence in batch)
            task_tokens = visual_output[:, : self.num_tasks]
            # Apply linear projection to increase dimension
            results["visual_task_embeddings"] = self.visual_projection(task_tokens)

        return results


class SigmoidLoss(nn.Module):
    def __init__(self, t_prime=math.log(10), b=-10.0, num_tasks=1):
        """
        The Sigmoid Loss function from the SigLIP paper.
        """
        super().__init__()
        self.t_prime = nn.Parameter(torch.tensor(t_prime))
        self.b = nn.Parameter(torch.tensor(b))
        self.loss_w = nn.Parameter(torch.ones(num_tasks))

    def forward(
        self, img_emb, txt_emb, labels=None, txt_weights=None, return_logits=False
    ):
        """
        Compute the sigmoid loss given image and text embeddings and multi-label ground truth.

        Args:
            img_emb: Tensor of shape [B, D] (image embeddings)
            txt_emb: Tensor of shape [C, D] (text embeddings)
            labels: Tensor of shape [B, C] (multi-label ground truth, with 1 for relevant pairs and 0 otherwise)
            txt_weights: Tensor of shape [C] (weights for each text label)
            return_logits: If True, return logits together with the loss

        Returns:
            Scalar loss value.
        """
        if labels is None and not return_logits:
            raise ValueError("labels must be provided if return_logits is False")
        if txt_weights is not None:
            # check that txt_weights has the correct shape
            if txt_weights.shape[0] != txt_emb.shape[0]:
                raise ValueError("txt_weights must have shape [C]")

        t = torch.exp(self.t_prime)  # Learnable temperature
        zimg = F.normalize(img_emb, p=2, dim=-1)  # L2-normalize image embeddings
        ztxt = F.normalize(txt_emb, p=2, dim=-1)  # L2-normalize text embeddings

        logits = t * (zimg @ ztxt.T) + self.b  # Compute logits

        if labels is None:
            return logits

        # Ignore rows where labels are all zeros
        valid_rows = labels.sum(dim=1) > 0  # Mask for rows with at least one label
        if valid_rows.any():  # Avoid computing loss if no valid samples exist
            loss = F.binary_cross_entropy_with_logits(
                logits[valid_rows], labels[valid_rows].float(), reduction="none"
            )
            if txt_weights is not None:
                loss = loss * txt_weights
            loss = loss.mean()
        else:
            loss = torch.tensor(0.0, device=img_emb.device)

        if return_logits:
            return loss, logits
        return loss
