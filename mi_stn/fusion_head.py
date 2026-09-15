import torch
import torch.nn as nn


class MISTNFusionHead(nn.Module):
    """
    Fusion Head untuk MI-STN.

    Menggabungkan:
        1. Motion embedding
        2. Interaction embedding

    Kemudian melakukan temporal modeling menggunakan GRU
    dan menghasilkan prediksi risiko kecelakaan pada
    beberapa temporal horizon.

    Input:
        motion_features:
            [B, T, N, D]

        interaction_features:
            [B, T, N, D]

    Output:
        risk:
            [B, T, H]

        B = batch size
        T = timestep
        N = jumlah objek
        D = feature dimension
        H = jumlah temporal horizon
    """

    def __init__(
        self,
        motion_dim=64,
        interaction_dim=64,
        fusion_dim=128,
        temporal_dim=128,
        num_horizons=3,
        dropout=0.1
    ):
        super().__init__()

        self.motion_dim = motion_dim
        self.interaction_dim = interaction_dim
        self.fusion_dim = fusion_dim
        self.temporal_dim = temporal_dim
        self.num_horizons = num_horizons

        # --------------------------------------------------
        # 1. Fusion
        # --------------------------------------------------

        self.fusion = nn.Sequential(
            nn.Linear(
                motion_dim + interaction_dim,
                fusion_dim
            ),
            nn.ReLU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout)
        )

        # --------------------------------------------------
        # 2. Temporal modeling
        # --------------------------------------------------

        self.gru = nn.GRU(
            input_size=fusion_dim,
            hidden_size=temporal_dim,
            num_layers=1,
            batch_first=True
        )

        # --------------------------------------------------
        # 3. Risk prediction
        # --------------------------------------------------

        self.risk_predictor = nn.Sequential(
            nn.Linear(
                temporal_dim,
                temporal_dim // 2
            ),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                temporal_dim // 2,
                num_horizons
            )
        )

    def forward(
        self,
        motion_features,
        interaction_features
    ):
        """
        Args:
            motion_features:
                [B, T, N, D]

            interaction_features:
                [B, T, N, D]

        Returns:
            risk:
                [B, T, H]
        """

        if motion_features.dim() != 4:
            raise ValueError(
                "motion_features harus memiliki bentuk "
                "[B, T, N, D], "
                f"tetapi mendapatkan {motion_features.shape}"
            )

        if interaction_features.dim() != 4:
            raise ValueError(
                "interaction_features harus memiliki bentuk "
                "[B, T, N, D], "
                f"tetapi mendapatkan {interaction_features.shape}"
            )

        if motion_features.shape != interaction_features.shape:
            raise ValueError(
                "motion_features dan interaction_features "
                "harus memiliki bentuk yang sama. "
                f"Motion: {motion_features.shape}, "
                f"Interaction: {interaction_features.shape}"
            )

        B, T, N, D = motion_features.shape

        # --------------------------------------------------
        # 1. Gabungkan motion + interaction
        # --------------------------------------------------

        x = torch.cat(
            [
                motion_features,
                interaction_features
            ],
            dim=-1
        )

        # [B, T, N, motion_dim + interaction_dim]

        x = self.fusion(x)

        # [B, T, N, fusion_dim]

        # --------------------------------------------------
        # 2. Aggregate object dimension
        # --------------------------------------------------

        # Saat ini menggunakan mean pooling.
        # Setiap timestep direpresentasikan oleh
        # seluruh objek yang tersedia.

        x = x.mean(dim=2)

        # [B, T, fusion_dim]

        # --------------------------------------------------
        # 3. Temporal modeling
        # --------------------------------------------------

        x, _ = self.gru(x)

        # [B, T, temporal_dim]

        # --------------------------------------------------
        # 4. Risk prediction
        # --------------------------------------------------

        risk = self.risk_predictor(x)

        # [B, T, num_horizons]

        return risk