import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mmaction.registry import DATASETS
from mmengine.registry import FUNCTIONS


@DATASETS.register_module()
class MISTNDataset(Dataset):
    """
    Dataset MI-STN yang kompatibel dengan framework TAA / RiskProp.

    Satu item dataset merepresentasikan satu video.

    Setiap video diubah menjadi sequence temporal tetap:

        [T, N, 9]          motion
        [T, N, K, 9]       interaction

    T = sequence_length
    N = max_objects
    K = max_neighbors

    Output batch selanjutnya digunakan oleh MISTNRecognizer.
    Metadata is_val / is_test dibuat mengikuti kebutuhan
    AnticipationMetric milik TAA.
    """

    def __init__(
        self,
        split,
        annotation_file="/content/data_text_annotation.xlsx",
        feature_root="/content/MI-STN/features",
        data_root=None,
        sequence_length=30,
        max_objects=50,
        max_neighbors=5,
        horizons=(1, 2, 3),
        fps=30,
        stride=None,
        test_mode=False,
    ):
        super().__init__()

        self.split = split

        # --------------------------------------------------
        # Feature root
        # --------------------------------------------------

        if data_root is not None:
            feature_root = data_root

        self.feature_root = Path(feature_root) / split

        if not self.feature_root.exists():
            raise FileNotFoundError(
                f"Feature directory tidak ditemukan: "
                f"{self.feature_root}"
            )

        # --------------------------------------------------
        # Configuration
        # --------------------------------------------------

        self.sequence_length = sequence_length
        self.max_objects = max_objects
        self.max_neighbors = max_neighbors
        self.horizons = tuple(horizons)
        self.fps = fps
        self.stride = stride
        self.test_mode = test_mode

        # --------------------------------------------------
        # Annotation
        # --------------------------------------------------

        annotation_file = Path(annotation_file)

        if not annotation_file.exists():
            raise FileNotFoundError(
                f"Annotation file tidak ditemukan: "
                f"{annotation_file}"
            )

        df = pd.read_excel(
            annotation_file,
            sheet_name=1,
            header=None,
        )

        self.annotations = {}

        for _, row in df.iterrows():
            try:
                video_id = (
                    f"{int(row[5])}_"
                    f"{str(row[0]).zfill(3)}"
                )

                self.annotations[video_id] = {
                    "accident": int(row[6]),
                    "abnormal_start": int(row[7]),
                    "accident_frame": int(row[8]),
                    "abnormal_end": int(row[9]),
                    "total_frames": int(row[10]),
                }

            except (TypeError, ValueError, IndexError):
                continue

        # --------------------------------------------------
        # Build video samples
        # --------------------------------------------------

        self.samples = []

        matched_videos = 0
        skipped_videos = 0

        video_dirs = sorted(
            p for p in self.feature_root.iterdir()
            if p.is_dir()
        )

        for video_dir in video_dirs:

            video_id = video_dir.name

            # ----------------------------------------------
            # Annotation harus tersedia
            # ----------------------------------------------

            if video_id not in self.annotations:
                skipped_videos += 1
                continue

            motion_path = video_dir / "motion.json"
            interaction_path = video_dir / "interaction.json"

            if not motion_path.exists():
                skipped_videos += 1
                continue

            if not interaction_path.exists():
                skipped_videos += 1
                continue

            # ----------------------------------------------
            # Load motion untuk memperoleh daftar frame
            # ----------------------------------------------

            try:
                with open(
                    motion_path,
                    "r",
                    encoding="utf-8",
                ) as f:
                    motion = json.load(f)

            except Exception:
                skipped_videos += 1
                continue

            frames = sorted(
                {
                    item["frame"]
                    for obj in motion.values()
                    for item in obj.get("features", [])
                }
            )

            if len(frames) < self.sequence_length:
                skipped_videos += 1
                continue

            ann = self.annotations[video_id]

            accident = ann["accident"]
            accident_frame = ann["accident_frame"]

            # ----------------------------------------------
            # Hanya gunakan frame sebelum kecelakaan
            # ----------------------------------------------

            if accident == 1:
                valid_frames = [
                    frame
                    for frame in frames
                    if frame < accident_frame
                ]
            else:
                valid_frames = frames

            if len(valid_frames) < self.sequence_length:
                skipped_videos += 1
                continue

            # ----------------------------------------------
            # Ambil sequence tetap.
            #
            # TAA menggunakan num_clips=30.
            # Karena itu default MI-STN juga menggunakan
            # sequence_length=30.
            #
            # Satu video = satu sample.
            # ----------------------------------------------

            indices = np.linspace(
                0,
                len(valid_frames) - 1,
                num=self.sequence_length,
                dtype=int,
            )

            selected_frames = [
                valid_frames[i]
                for i in indices
            ]

            self.samples.append(
                {
                    "video_id": video_id,
                    "frames": selected_frames,
                    "accident": accident,
                    "accident_frame": accident_frame,
                    "abnormal_start": ann["abnormal_start"],
                    "abnormal_end": ann["abnormal_end"],
                    "total_frames": ann["total_frames"],
                }
            )

            matched_videos += 1

        print(
            f"[MISTNDataset] "
            f"split={split} "
            f"videos={matched_videos} "
            f"samples={len(self.samples)} "
            f"skipped={skipped_videos}"
        )

        # --------------------------------------------------
        # JSON cache
        # --------------------------------------------------

        self._motion_cache = {}
        self._interaction_cache = {}

    # ======================================================
    # Length
    # ======================================================

    def __len__(self):
        return len(self.samples)

    # ======================================================
    # Motion loader
    # ======================================================

    def _load_motion(self, video_id):

        if video_id not in self._motion_cache:

            path = (
                self.feature_root
                / video_id
                / "motion.json"
            )

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:

                self._motion_cache[video_id] = json.load(f)

        return self._motion_cache[video_id]

    # ======================================================
    # Interaction loader
    # ======================================================

    def _load_interaction(self, video_id):

        if video_id not in self._interaction_cache:

            path = (
                self.feature_root
                / video_id
                / "interaction.json"
            )

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:

                self._interaction_cache[
                    video_id
                ] = json.load(f)

        return self._interaction_cache[video_id]

    # ======================================================
    # Get Item
    # ======================================================

    def __getitem__(self, idx):

        sample = self.samples[idx]

        video_id = sample["video_id"]
        target_frames = sample["frames"]

        accident = sample["accident"]
        accident_frame = sample["accident_frame"]

        motion = self._load_motion(video_id)
        interaction = self._load_interaction(video_id)

        T = self.sequence_length
        N = self.max_objects
        K = self.max_neighbors
        F = 9

        # --------------------------------------------------
        # Tensor allocation
        # --------------------------------------------------

        motion_tensor = torch.zeros(
            T,
            N,
            F,
            dtype=torch.float32,
        )

        interaction_tensor = torch.zeros(
            T,
            N,
            K,
            F,
            dtype=torch.float32,
        )

        motion_mask = torch.zeros(
            T,
            N,
            dtype=torch.bool,
        )

        neighbor_mask = torch.zeros(
            T,
            N,
            K,
            dtype=torch.bool,
        )

        labels = torch.zeros(
            T,
            len(self.horizons),
            dtype=torch.float32,
        )

        # --------------------------------------------------
        # Object mapping
        # --------------------------------------------------

        object_ids = list(motion.keys())[:N]

        object_to_idx = {
            str(object_id): i
            for i, object_id in enumerate(object_ids)
        }

        frame_to_t = {
            frame: t
            for t, frame in enumerate(target_frames)
        }

        # ==================================================
        # MOTION
        # ==================================================

        for object_id in object_ids:

            object_idx = object_to_idx[
                str(object_id)
            ]

            for item in motion[
                object_id
            ].get("features", []):

                frame = item["frame"]

                if frame not in frame_to_t:
                    continue

                t = frame_to_t[frame]

                motion_tensor[
                    t,
                    object_idx,
                ] = torch.tensor(
                    [
                        item["x"],
                        item["y"],
                        item["vx"],
                        item["vy"],
                        item["speed"],
                        item["ax"],
                        item["ay"],
                        item["accel"],
                        item["direction"],
                    ],
                    dtype=torch.float32,
                )

                motion_mask[
                    t,
                    object_idx,
                ] = True

        # ==================================================
        # INTERACTION
        # ==================================================

        for t, frame in enumerate(target_frames):

            frame_data = interaction.get(
                str(frame),
                {},
            )

            for source_id, neighbors in frame_data.items():

                source_idx = object_to_idx.get(
                    str(source_id)
                )

                if source_idx is None:
                    continue

                for k, neighbor in enumerate(
                    neighbors[:K]
                ):

                    interaction_tensor[
                        t,
                        source_idx,
                        k,
                    ] = torch.tensor(
                        [
                            neighbor["distance"],
                            neighbor["dx"],
                            neighbor["dy"],
                            neighbor["relative_vx"],
                            neighbor["relative_vy"],
                            neighbor["relative_speed"],
                            neighbor["closing_speed"],
                            neighbor["direction_difference"],
                            neighbor["interaction_strength"],
                        ],
                        dtype=torch.float32,
                    )

                    neighbor_mask[
                        t,
                        source_idx,
                        k,
                    ] = True

        # ==================================================
        # LABEL
        # ==================================================

        if accident == 1:

            for t, frame in enumerate(
                target_frames
            ):

                delta = (
                    accident_frame - frame
                )

                for h, horizon in enumerate(
                    self.horizons
                ):

                    if (
                        0 < delta <= self.fps * horizon
                    ):
                        labels[t, h] = 1.0

        # ==================================================
        # TAA-compatible metadata
        # ==================================================

        # IMPORTANT:
        #
        # Validation:
        #   is_val=True
        #   is_test=False
        #
        # Test:
        #   is_val=True
        #   is_test=True
        #
        # Test tetap is_val=True secara sengaja agar
        # AnticipationMetric RiskProp menghitung bagian "@"
        # saat tools/test.py dijalankan.

        is_test = self.split == "test"

        is_val = (
            self.split in ("val", "test")
        )

        return {
            # ------------------------------------------------
            # Model inputs
            # ------------------------------------------------

            "motion": motion_tensor,

            "interaction": interaction_tensor,

            "motion_mask": motion_mask,

            "neighbor_mask": neighbor_mask,

            "labels": labels,

            # ------------------------------------------------
            # Temporal information
            # ------------------------------------------------

            "frames": torch.tensor(
                target_frames,
                dtype=torch.long,
            ),

            "accident_frame": torch.tensor(
                accident_frame,
                dtype=torch.long,
            ),

            # ------------------------------------------------
            # Video metadata
            # ------------------------------------------------

            "video_id": video_id,

            "target": bool(accident),

            "abnormal_start_frame": int(
                sample["abnormal_start"]
            ),

            "dataset": "DADA2000",

            "frame_dir": video_id,

            "filename_tmpl": "{:04d}.png",

            "type": "DADA2000",

            "start_index": 1,

            "total_frames": int(
                sample["total_frames"]
            ),

            # ------------------------------------------------
            # TAA metadata
            # ------------------------------------------------

            "is_val": is_val,

            "is_test": is_test,
        }


# ============================================================
# Collate Function
# ============================================================

@FUNCTIONS.register_module()
def mistn_collate_fn(batch):
    """
    Collate batch MI-STN.

    Format dibuat kompatibel dengan BaseModel MMEngine:

        {
            "inputs": {...},
            "data_samples": [...]
        }

    sehingga tools/train.py dan tools/test.py
    dapat tetap digunakan.
    """

    return {
        "inputs": {
            "motion": torch.stack(
                [
                    item["motion"]
                    for item in batch
                ]
            ),

            "interaction": torch.stack(
                [
                    item["interaction"]
                    for item in batch
                ]
            ),

            "motion_mask": torch.stack(
                [
                    item["motion_mask"]
                    for item in batch
                ]
            ),

            "neighbor_mask": torch.stack(
                [
                    item["neighbor_mask"]
                    for item in batch
                ]
            ),

            "labels": torch.stack(
                [
                    item["labels"]
                    for item in batch
                ]
            ),

            "frames": torch.stack(
                [
                    item["frames"]
                    for item in batch
                ]
            ),

            "accident_frame": torch.stack(
                [
                    item["accident_frame"]
                    for item in batch
                ]
            ),

            # Metadata masih berada di inputs karena
            # MISTNRecognizer membutuhkan informasi ini
            # saat predict().
            "video_id": [
                item["video_id"]
                for item in batch
            ],

            "target": [
                item["target"]
                for item in batch
            ],

            "abnormal_start_frame": [
                item["abnormal_start_frame"]
                for item in batch
            ],

            "dataset": [
                item["dataset"]
                for item in batch
            ],

            "frame_dir": [
                item["frame_dir"]
                for item in batch
            ],

            "filename_tmpl": [
                item["filename_tmpl"]
                for item in batch
            ],

            "type": [
                item["type"]
                for item in batch
            ],

            "start_index": [
                item["start_index"]
                for item in batch
            ],

            "total_frames": [
                item["total_frames"]
                for item in batch
            ],

            "is_val": [
                item["is_val"]
                for item in batch
            ],

            "is_test": [
                item["is_test"]
                for item in batch
            ],
        },

        "data_samples": [
            {
                "video_id": item["video_id"],
                "target": item["target"],
                "is_val": item["is_val"],
                "is_test": item["is_test"],
            }
            for item in batch
        ],
    }
