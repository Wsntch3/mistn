import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModel
from mmaction.registry import MODELS

from .motion_module import MotionModule
from .interaction_module import InteractionModule
from .fusion_head import MISTNFusionHead


@MODELS.register_module()
class MISTNRecognizer(BaseModel):
    """
    Motion-Interaction Spatio-Temporal Network (MI-STN).

    MISTN hanya menyediakan:
        - MotionModule
        - InteractionModule
        - FusionHead

    Training, validation, testing, checkpoint, metric,
    dan hooks tetap menggunakan framework TAA / RiskProp.

    Input dari MISTNDataset:
        motion:
            [B, T, N, 9]

        interaction:
            [B, T, N, K, 9]

        motion_mask:
            [B, T, N]

        neighbor_mask:
            [B, T, N, K]

        labels:
            [B, T, H]

    Output:
        risk logits:
            [B, T, H]
    """

    def __init__(
        self,
        motion_input_dim=9,
        motion_hidden_dim=64,
        motion_output_dim=64,

        interaction_input_dim=9,
        interaction_hidden_dim=64,
        interaction_output_dim=64,

        fusion_dim=128,
        temporal_dim=128,

        num_horizons=3,
        dropout=0.1,

        init_cfg=None,
    ):
        super().__init__(
            init_cfg=init_cfg
        )

        # ==================================================
        # 1. Motion Module
        # ==================================================

        self.motion_module = MotionModule(
            input_dim=motion_input_dim,
            hidden_dim=motion_hidden_dim,
            output_dim=motion_output_dim,
            dropout=dropout,
        )

        # ==================================================
        # 2. Interaction Module
        # ==================================================

        self.interaction_module = InteractionModule(
            input_dim=interaction_input_dim,
            hidden_dim=interaction_hidden_dim,
            output_dim=interaction_output_dim,
            dropout=dropout,
        )

        # ==================================================
        # 3. Fusion Head
        # ==================================================

        self.fusion_head = MISTNFusionHead(
            motion_dim=motion_output_dim,
            interaction_dim=interaction_output_dim,
            fusion_dim=fusion_dim,
            temporal_dim=temporal_dim,
            num_horizons=num_horizons,
            dropout=dropout,
        )

        # ==================================================
        # Compatibility dengan TAA EpochHook
        # ==================================================

        self.cls_head = self.fusion_head

        # ==================================================
        # Configuration
        # ==================================================

        self.num_horizons = num_horizons

    # ======================================================
    # Feature Extraction
    # ======================================================

    def _extract_features(
        self,
        motion,
        interaction,
        motion_mask=None,
        neighbor_mask=None,
    ):
        """
        Menghasilkan motion dan interaction embedding.
        """

        # --------------------------------------------------
        # Motion branch
        # --------------------------------------------------

        motion_embedding = self.motion_module(
            motion
        )

        # [B, T, N, motion_output_dim]

        # --------------------------------------------------
        # Interaction branch
        # --------------------------------------------------

        interaction_embedding = self.interaction_module(
            interaction,
            interaction_mask=neighbor_mask,
        )

        # [B, T, N, interaction_output_dim]

        return (
            motion_embedding,
            interaction_embedding,
        )

    # ======================================================
    # Forward Tensor
    # ======================================================

    def _forward(
        self,
        inputs,
        data_samples=None,
        **kwargs,
    ):
        """
        Forward tensor untuk mode tensor.
        """

        motion = inputs["motion"]
        interaction = inputs["interaction"]

        motion_mask = inputs.get(
            "motion_mask",
            None,
        )

        neighbor_mask = inputs.get(
            "neighbor_mask",
            None,
        )

        motion_embedding, interaction_embedding = (
            self._extract_features(
                motion=motion,
                interaction=interaction,
                motion_mask=motion_mask,
                neighbor_mask=neighbor_mask,
            )
        )

        risk_logits = self.fusion_head(
            motion_embedding,
            interaction_embedding,
        )

        return risk_logits

    # ======================================================
    # Loss
    # ======================================================

    def loss(
        self,
        inputs,
        data_samples=None,
        **kwargs,
    ):
        """
        Training loss.

        labels:
            [B, T, H]
        """

        labels = inputs["labels"].float()

        risk_logits = self._forward(
            inputs=inputs,
            data_samples=data_samples,
        )

        if risk_logits.shape != labels.shape:
            raise ValueError(
                "Ukuran output model dan label tidak sama. "
                f"risk_logits={risk_logits.shape}, "
                f"labels={labels.shape}"
            )

        loss = F.binary_cross_entropy_with_logits(
            risk_logits,
            labels,
        )

        return {
            "loss": loss,
        }

    # ======================================================
    # Prediction
    # ======================================================

    def predict(
        self,
        inputs,
        data_samples=None,
        **kwargs,
    ):
        """
        Menghasilkan output yang kompatibel dengan
        taa.AnticipationMetric.

        Output setiap sample berbentuk dict yang memiliki:

            pred_score
            target
            frame_inds
            abnormal_start_frame
            accident_frame
            video_id
            dataset
            frame_dir
            filename_tmpl
            type
            is_val
            is_test
        """

        risk_logits = self._forward(
            inputs=inputs,
            data_samples=data_samples,
        )

        # Probabilitas risiko
        pred_score = torch.sigmoid(
            risk_logits
        )

        batch_size = pred_score.shape[0]

        results = []

        video_ids = inputs["video_id"]

        targets = inputs["target"]

        abnormal_start_frames = inputs[
            "abnormal_start_frame"
        ]

        is_val_list = inputs[
            "is_val"
        ]

        is_test_list = inputs[
            "is_test"
        ]

        frames = inputs[
            "frames"
        ]

        accident_frames = inputs[
            "accident_frame"
        ]

        for i in range(batch_size):

            result = {
                # ------------------------------------------
                # Prediction
                # ------------------------------------------

                "pred_score": pred_score[i],

                # ------------------------------------------
                # Target
                # ------------------------------------------

                "target": bool(
                    targets[i]
                ),

                # ------------------------------------------
                # Temporal information
                # ------------------------------------------

                "frame_inds": frames[i].detach().cpu(),

                "abnormal_start_frame": int(
                    abnormal_start_frames[i]
                ),

                "accident_frame": int(
                    accident_frames[i].item()
                ),

                # ------------------------------------------
                # Identity
                # ------------------------------------------

                "video_id": video_ids[i],

                "dataset": "DADA2000",

                "frame_dir": video_ids[i],

                "filename_tmpl": "{:04d}.png",

                "type": "DADA2000",

                # ------------------------------------------
                # RiskProp metadata
                # ------------------------------------------

                "is_val": bool(
                    is_val_list[i]
                ),

                "is_test": bool(
                    is_test_list[i]
                ),
            }

            results.append(
                result
            )

        return results

    # ======================================================
    # Simple forward alias
    # ======================================================

    def forward(
        self,
        inputs,
        data_samples=None,
        mode="tensor",
        **kwargs,
    ):
        """
        Tidak perlu override BaseModel secara agresif.
        mode diteruskan ke implementasi BaseModel.
        """

        return super().forward(
            inputs=inputs,
            data_samples=data_samples,
            mode=mode,
            **kwargs,
        )
