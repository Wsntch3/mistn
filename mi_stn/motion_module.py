import torch
import torch.nn as nn


class MotionModule(nn.Module):
    """
    Motion Deviation Module untuk MI-STN.

    Motion features:
        x, y,
        vx, vy,
        speed,
        ax, ay,
        accel,
        direction

    Input:
        [B, T, N, 9]

        B = batch size
        T = jumlah timestep/frame
        N = jumlah objek
        9 = jumlah motion features

    Output:
        [B, T, N, D]
    """

    def __init__(
        self,
        input_dim=9,
        hidden_dim=64,
        output_dim=64,
        dropout=0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.motion_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.LayerNorm(hidden_dim),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, motion_features):
        """
        Args:
            motion_features:
                Tensor [B, T, N, 9]

        Returns:
            motion_embedding:
                Tensor [B, T, N, output_dim]
        """

        if motion_features.dim() != 4:
            raise ValueError(
                "motion_features harus memiliki bentuk "
                "[B, T, N, F], "
                f"tetapi mendapatkan {motion_features.shape}"
            )

        B, T, N, F = motion_features.shape

        if F != self.input_dim:
            raise ValueError(
                f"Jumlah motion feature tidak sesuai. "
                f"Diharapkan {self.input_dim}, tetapi mendapatkan {F}"
            )

        # Gabungkan dimensi batch, waktu, dan objek
        # agar setiap object-frame diproses oleh encoder.
        x = motion_features.reshape(B * T * N, F)

        # Encode motion features
        x = self.motion_encoder(x)

        # Kembalikan bentuk temporal-object
        x = x.reshape(B, T, N, self.output_dim)

        return x