import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionModule(nn.Module):
    """
    Dynamic Inter-Object Interaction Module untuk MI-STN.

    Input:
        interaction_features:
            [B, T, N, K, F]

        interaction_mask:
            [B, T, N, K]

    Output:
        interaction_embedding:
            [B, T, N, D_out]
    """

    def __init__(
        self,
        input_dim=9,
        hidden_dim=64,
        output_dim=64,
        dropout=0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # --------------------------------------------------
        # 1. Project raw pairwise interaction features
        # --------------------------------------------------

        self.feature_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        # --------------------------------------------------
        # 2. Attention over neighbors
        # --------------------------------------------------

        self.query = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        self.key = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )

        self.value = nn.Linear(
            hidden_dim,
            output_dim,
            bias=False,
        )

        # --------------------------------------------------
        # 3. Output projection
        # --------------------------------------------------

        self.output_projection = nn.Linear(
            output_dim,
            output_dim,
        )

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        interaction_features,
        interaction_mask=None,
    ):
        """
        Args:
            interaction_features:
                [B, T, N, K, 9]

            interaction_mask:
                [B, T, N, K]

        Returns:
            interaction_embedding:
                [B, T, N, 64]
        """

        if interaction_features.dim() != 5:
            raise ValueError(
                "interaction_features harus memiliki bentuk "
                "[B, T, N, K, F], "
                f"tetapi mendapatkan {interaction_features.shape}"
            )

        B, T, N, K, F_dim = interaction_features.shape

        if F_dim != self.input_dim:
            raise ValueError(
                f"Dimensi interaction feature tidak sesuai. "
                f"Diharapkan {self.input_dim}, "
                f"tetapi mendapatkan {F_dim}"
            )

        # --------------------------------------------------
        # 1. Feature projection
        # --------------------------------------------------

        x = self.feature_projection(
            interaction_features
        )

        # [B, T, N, K, hidden_dim]

        # --------------------------------------------------
        # 2. Query, Key, Value
        # --------------------------------------------------

        q = self.query(x)

        k = self.key(x)

        v = self.value(x)

        # --------------------------------------------------
        # 3. Query aggregation
        # --------------------------------------------------

        # Gunakan mean query dari seluruh neighbor
        # sebagai representasi sumber object.

        q_global = q.mean(dim=3, keepdim=True)

        # [B, T, N, 1, hidden_dim]

        # --------------------------------------------------
        # 4. Attention score antar neighbor
        # --------------------------------------------------

        scores = torch.sum(
            q_global * k,
            dim=-1,
        )

        scores = scores / (
            self.hidden_dim ** 0.5
        )

        # [B, T, N, K]

        # --------------------------------------------------
        # 5. Mask padding neighbors
        # --------------------------------------------------

        if interaction_mask is not None:

            if interaction_mask.shape != (
                B,
                T,
                N,
                K,
            ):
                raise ValueError(
                    "interaction_mask harus memiliki bentuk "
                    f"[B,T,N,K], tetapi mendapatkan "
                    f"{interaction_mask.shape}"
                )

            mask = interaction_mask.bool()

            scores = scores.masked_fill(
                ~mask,
                torch.finfo(scores.dtype).min,
            )

        else:
            mask = None

        # --------------------------------------------------
        # 6. Attention weights
        # --------------------------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1,
        )

        if mask is not None:
            attention_weights = (
                attention_weights * mask.to(
                    attention_weights.dtype
                )
            )

            denominator = attention_weights.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-6)

            attention_weights = (
                attention_weights / denominator
            )

        attention_weights = self.dropout(
            attention_weights
        )

        # --------------------------------------------------
        # 7. Aggregate neighbors
        # --------------------------------------------------

        interaction = torch.sum(
            attention_weights.unsqueeze(-1) * v,
            dim=3,
        )

        # [B, T, N, output_dim]

        # --------------------------------------------------
        # 8. Output projection
        # --------------------------------------------------

        interaction = self.output_projection(
            interaction
        )

        interaction = self.norm(interaction)

        return interaction
