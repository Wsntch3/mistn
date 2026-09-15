import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionModule(nn.Module):
    """
    Dynamic Inter-Object Interaction Module untuk MI-STN.

    Input:
        object_features:
            [B, T, N, D]

        B = batch size
        T = timestep
        N = jumlah objek
        D = dimensi feature objek

    Output:
        interaction_features:
            [B, T, N, D_out]
    """

    def __init__(
        self,
        input_dim=64,
        hidden_dim=64,
        output_dim=64,
        dropout=0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Transformasi feature objek
        self.feature_projection = nn.Linear(
            input_dim,
            hidden_dim
        )

        # Attention layer
        self.query = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False
        )

        self.key = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False
        )

        self.value = nn.Linear(
            hidden_dim,
            output_dim,
            bias=False
        )

        # Output projection
        self.output_projection = nn.Linear(
            output_dim,
            output_dim
        )

        self.norm = nn.LayerNorm(output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, object_features, object_mask=None):
        """
        Args:
            object_features:
                Tensor [B, T, N, D]

            object_mask:
                Optional tensor [B, T, N]
                True  = objek valid
                False = padding

        Returns:
            interaction_features:
                Tensor [B, T, N, output_dim]
        """

        if object_features.dim() != 4:
            raise ValueError(
                "object_features harus memiliki bentuk "
                "[B, T, N, D], "
                f"tetapi mendapatkan {object_features.shape}"
            )

        B, T, N, D = object_features.shape

        if D != self.input_dim:
            raise ValueError(
                f"Dimensi feature tidak sesuai. "
                f"Diharapkan {self.input_dim}, "
                f"tetapi mendapatkan {D}"
            )

        # --------------------------------------------------
        # 1. Project object features
        # --------------------------------------------------

        x = self.feature_projection(object_features)

        # x:
        # [B, T, N, hidden_dim]

        # --------------------------------------------------
        # 2. Query, Key, Value
        # --------------------------------------------------

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # --------------------------------------------------
        # 3. Pairwise attention antar objek
        # --------------------------------------------------

        # q:
        # [B, T, N, D]

        # k:
        # [B, T, N, D]

        # attention_scores:
        # [B, T, N, N]

        attention_scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        attention_scores = attention_scores / (
            self.hidden_dim ** 0.5
        )

        # --------------------------------------------------
        # 4. Mask objek padding jika tersedia
        # --------------------------------------------------

        if object_mask is not None:

            if object_mask.shape != (B, T, N):
                raise ValueError(
                    "object_mask harus memiliki bentuk "
                    f"[B, T, N], tetapi mendapatkan "
                    f"{object_mask.shape}"
                )

            # Key mask
            key_mask = object_mask.unsqueeze(-2)

            attention_scores = attention_scores.masked_fill(
                ~key_mask,
                float("-inf")
            )

        # --------------------------------------------------
        # 5. Softmax attention
        # --------------------------------------------------

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        attention_weights = self.dropout(
            attention_weights
        )

        # --------------------------------------------------
        # 6. Aggregate informasi objek lain
        # --------------------------------------------------

        interaction = torch.matmul(
            attention_weights,
            v
        )

        # --------------------------------------------------
        # 7. Output projection
        # --------------------------------------------------

        interaction = self.output_projection(
            interaction
        )

        # Residual connection
        # hanya jika dimensi sama
        if interaction.shape[-1] == object_features.shape[-1]:
            interaction = interaction + object_features

        interaction = self.norm(interaction)

        return interaction