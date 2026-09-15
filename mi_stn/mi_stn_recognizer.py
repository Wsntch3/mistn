import torch
import torch.nn as nn

from mmengine.registry import MODELS

from .motion_module import MotionModule
from .interaction_module import InteractionModule
from .fusion_head import MISTNFusionHead


@MODELS.register_module()
class MISTNRecognizer(nn.Module):
    """
    Motion-Interaction Spatio-Temporal Network (MI-STN).

    Pipeline:
        Motion Features
            ↓
        MotionModule
            ↓
        Motion Embedding
            ↓
        InteractionModule
            ↓
        Interaction Embedding
            ↓
        MISTNFusionHead
            ↓
        Temporal Modeling (GRU)
            ↓
        Risk Prediction

    Input:
        motion_features:
            [B, T, N, 9]

    Output:
        risk:
            [B, T, H]

        B = batch size
        T = temporal sequence length
        N = number of objects
        9 = motion features
        H = number of temporal horizons
    """

    def __init__(
        self,
        motion_input_dim=9,
        motion_hidden_dim=64,
        motion_output_dim=64,
        interaction_hidden_dim=64,
        interaction_output_dim=64,
        fusion_dim=128,
        temporal_dim=128,
        num_horizons=3,
        dropout=0.1,
    ):
        super().__init__()

        # --------------------------------------------------
        # 1. Motion Module
        # --------------------------------------------------

        self.motion_module = MotionModule(
            input_dim=motion_input_dim,
            hidden_dim=motion_hidden_dim,
            output_dim=motion_output_dim,
            dropout=dropout,
        )

        # --------------------------------------------------
        # 2. Interaction Module
        # --------------------------------------------------

        self.interaction_module = InteractionModule(
            input_dim=motion_output_dim,
            hidden_dim=interaction_hidden_dim,
            output_dim=interaction_output_dim,
            dropout=dropout,
        )

        # --------------------------------------------------
        # 3. Fusion + Temporal + Risk Prediction
        # --------------------------------------------------

        self.fusion_head = MISTNFusionHead(
            motion_dim=motion_output_dim,
            interaction_dim=interaction_output_dim,
            fusion_dim=fusion_dim,
            temporal_dim=temporal_dim,
            num_horizons=num_horizons,
            dropout=dropout,
        )

    def forward(
        self,
        motion_features,
        object_mask=None,
    ):
        """
        Args:
            motion_features:
                Tensor [B, T, N, 9]

            object_mask:
                Optional tensor [B, T, N]

        Returns:
            risk:
                Tensor [B, T, H]
        """

        # --------------------------------------------------
        # Motion representation
        # --------------------------------------------------

        motion_embedding = self.motion_module(
            motion_features
        )

        # --------------------------------------------------
        # Dynamic interaction representation
        # --------------------------------------------------

        interaction_embedding = self.interaction_module(
            motion_embedding,
            object_mask=object_mask,
        )

        # --------------------------------------------------
        # Fusion + temporal modeling + risk prediction
        # --------------------------------------------------

        risk = self.fusion_head(
            motion_embedding,
            interaction_embedding,
        )

        return risk

