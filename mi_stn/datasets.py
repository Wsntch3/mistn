import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from mmaction.registry import DATASETS
from mmengine.registry import FUNCTIONS
from taa.splits import dada_test


@DATASETS.register_module()
class MISTNDataset(Dataset):
    """
    MI-STN dataset dengan pembagian video dan semantics sampling
    yang mengikuti RiskProp/TAA.

    Prinsip utama:
        - RiskProp menentukan video mana yang masuk train/test.
        - MI-STN hanya menggunakan feature dari video tersebut.
        - Satu video dapat menghasilkan:
            target=True
            target=False
        - Sehingga jumlah SAMPLE tidak sama dengan jumlah VIDEO.

    DADA-2000:
        FPS             = 30
        Sampling        = 10 Hz
        Frame interval  = 3 frame
        Sequence length = 30 timestep
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

        if data_root is not None:
            feature_root = data_root

        self.feature_root = Path(feature_root) / split

        if not self.feature_root.exists():
            raise FileNotFoundError(
                f"Feature directory tidak ditemukan: {self.feature_root}"
            )

        self.sequence_length = int(sequence_length)
        self.max_objects = int(max_objects)
        self.max_neighbors = int(max_neighbors)
        self.horizons = tuple(horizons)
        self.fps = int(fps)
        self.stride = stride
        self.test_mode = test_mode

        if self.fps % 10 != 0:
            raise ValueError(
                f"FPS {self.fps} tidak kompatibel dengan sampling TAA 10 Hz."
            )

        # DADA 30 FPS -> 10 Hz
        self.frame_interval = self.fps // 10

        # Sama dengan RiskProp
        self.start_index = 1

        annotation_file = Path(annotation_file)

        if not annotation_file.exists():
            raise FileNotFoundError(
                f"Annotation file tidak ditemukan: {annotation_file}"
            )

        df = pd.read_excel(
            annotation_file,
            sheet_name=1,
            header=None,
        )

        self.annotations = {}

        for _, row in df.iterrows():
            try:
                video_id = f"{int(row[5])}_{str(row[0]).zfill(3)}"

                self.annotations[video_id] = {
                    "accident": int(row[6]),
                    "abnormal_start": int(row[7]),
                    "accident_frame": int(row[8]),
                    "abnormal_end": int(row[9]),
                    "total_frames": int(row[10]),
                }

            except (TypeError, ValueError, IndexError):
                continue
        self.samples = []

        matched_videos = 0
        skipped_videos = 0

        # Statistik khusus untuk debugging
        skipped_not_in_split = 0
        skipped_no_annotation = 0
        skipped_no_motion = 0
        skipped_no_interaction = 0
        skipped_non_accident = 0
        skipped_not_enough_frames = 0

        video_dirs = sorted(
            p for p in self.feature_root.iterdir()
            if p.is_dir()
        )
        for video_dir in video_dirs:

            video_id = video_dir.name

            if video_id not in self.annotations:
                skipped_videos += 1
                skipped_no_annotation += 1
                continue

            in_dada_test = video_id in dada_test

            if split == "test":

                if not in_dada_test:
                    skipped_videos += 1
                    skipped_not_in_split += 1
                    continue

            elif split == "train":

                if in_dada_test:
                    skipped_videos += 1
                    skipped_not_in_split += 1
                    continue

            elif split == "val":

                # Jika val feature memang berasal dari subset
                # RiskProp, hanya gunakan video yang ada di
                # dada_test.
                if not in_dada_test:
                    skipped_videos += 1
                    skipped_not_in_split += 1
                    continue

            motion_path = video_dir / "motion.json"

            if not motion_path.exists():
                skipped_videos += 1
                skipped_no_motion += 1
                continue

            interaction_path = video_dir / "interaction.json"

            if not interaction_path.exists():
                skipped_videos += 1
                skipped_no_interaction += 1
                continue

            ann = self.annotations[video_id]

            if ann["accident"] == -1:
                skipped_videos += 1
                skipped_non_accident += 1
                continue

            try:
                category_id = int(video_id.split("_")[0])
            except (ValueError, IndexError):
                skipped_videos += 1
                continue

            if not 1 <= category_id <= 18:
                skipped_videos += 1
                continue

            accident_frame = ann["accident_frame"]

            positive_available = (
                accident_frame - self.start_index
                >= self.fps * 2
            )

            if positive_available:

                positive_frames = self._build_frame_sequence(
                    accident_frame
                )

                self.samples.append(
                    {
                        "video_id": video_id,
                        "frames": positive_frames,
                        "target": True,
                        "accident_frame": accident_frame,
                        "abnormal_start": ann["abnormal_start"],
                        "abnormal_end": ann["abnormal_end"],
                        "total_frames": ann["total_frames"],
                    }
                )

            negative_available = (
                accident_frame - self.start_index
                >= self.fps * 3.5
            )

            if negative_available:

                negative_end_frame = (
                    accident_frame - self.fps * 3
                )

                negative_frames = self._build_frame_sequence(
                    negative_end_frame
                )

                self.samples.append(
                    {
                        "video_id": video_id,
                        "frames": negative_frames,
                        "target": False,
                        "accident_frame": accident_frame,
                        "abnormal_start": ann["abnormal_start"],
                        "abnormal_end": ann["abnormal_end"],
                        "total_frames": ann["total_frames"],
                    }
                )

            if positive_available or negative_available:
                matched_videos += 1
            else:
                skipped_videos += 1
                skipped_not_enough_frames += 1

        # ========================================================
        # Statistik
        # ========================================================

        positive_count = sum(
            1
            for sample in self.samples
            if sample["target"] is True
        )

        negative_count = sum(
            1
            for sample in self.samples
            if sample["target"] is False
        )

        print()
        print("=" * 60)
        print("[MISTNDataset]")
        print(f"split             : {split}")
        print(f"feature_root      : {self.feature_root}")
        print(f"RiskProp test IDs : {len(dada_test)}")
        print(f"available folders : {len(video_dirs)}")
        print(f"matched videos    : {matched_videos}")
        print(f"total samples     : {len(self.samples)}")
        print(f"positive samples  : {positive_count}")
        print(f"negative samples  : {negative_count}")
        print(f"skipped videos    : {skipped_videos}")
        print()
        print("Skipped detail:")
        print(f"  not in split     : {skipped_not_in_split}")
        print(f"  no annotation    : {skipped_no_annotation}")
        print(f"  no motion        : {skipped_no_motion}")
        print(f"  no interaction   : {skipped_no_interaction}")
        print(f"  non accident     : {skipped_non_accident}")
        print(f"  insufficient     : {skipped_not_enough_frames}")
        print("=" * 60)


        print()
        print(f"[MISTNDataset] Sample list ({split}):")

        for i, sample in enumerate(self.samples):

            print(
                f"  [{i:03d}] "
                f"{sample['video_id']} "
                f"target={sample['target']} "
                f"accident_frame={sample['accident_frame']}"
            )

        print()

        self._motion_cache = {}
        self._interaction_cache = {}

    def _build_frame_sequence(self, end_frame):
        """
        Sampling 10 Hz seperti TAA/RiskProp.

        DADA:
            FPS = 30
            frame_interval = 3
            sequence_length = 30

        Frame terakhir:
            positive -> accident_frame
            negative -> accident_frame - 90
        """

        clip_inds = (
            torch.arange(
                self.sequence_length,
                dtype=torch.long,
            )
            * self.frame_interval
        )

        clip_inds_max = int(
            clip_inds[-1].item()
        )

        clip_inds = (
            clip_inds.numpy()
            + end_frame
            - self.start_index
            - clip_inds_max
        )

        frame_inds = (
            clip_inds.clip(min=0)
            + self.start_index
        )

        return [
            int(frame)
            for frame in frame_inds
        ]


    def __len__(self):
        return len(self.samples)

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
                self._interaction_cache[video_id] = json.load(f)

        return self._interaction_cache[video_id]

    # ============================================================
    # Get item
    # ============================================================

    def __getitem__(self, idx):

        sample = self.samples[idx]

        video_id = sample["video_id"]
        target_frames = sample["frames"]

        target = sample["target"]
        accident_frame = sample["accident_frame"]

        motion = self._load_motion(video_id)
        interaction = self._load_interaction(video_id)

        T = self.sequence_length
        N = self.max_objects
        K = self.max_neighbors
        F = 9

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

        object_ids = list(motion.keys())[:N]

        object_to_idx = {
            str(object_id): i
            for i, object_id in enumerate(object_ids)
        }

        frame_to_t = {
            frame: t
            for t, frame in enumerate(target_frames)
        }

        # ========================================================
        # Motion
        # ========================================================

        for object_id in object_ids:

            object_idx = object_to_idx[
                str(object_id)
            ]

            for item in motion[
                object_id
            ].get("features", []):

                frame = int(item["frame"])

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

        # ========================================================
        # Interaction
        # ========================================================

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

        # ========================================================
        # Horizon labels
        # ========================================================

        if target is True:

            for t, frame in enumerate(target_frames):

                delta = accident_frame - frame

                for h, horizon in enumerate(self.horizons):

                    if (
                        0 < delta <= self.fps * horizon
                    ):
                        labels[t, h] = 1.0

        # ========================================================
        # TAA metadata
        # ========================================================

        is_test = self.split == "test"

        is_val = self.split in (
            "val",
            "test",
        )

        return {
            "motion": motion_tensor,
            "interaction": interaction_tensor,
            "motion_mask": motion_mask,
            "neighbor_mask": neighbor_mask,
            "labels": labels,

            "frames": torch.tensor(
                target_frames,
                dtype=torch.long,
            ),

            "accident_frame": torch.tensor(
                accident_frame,
                dtype=torch.long,
            ),

            "video_id": video_id,
            "target": bool(target),

            "abnormal_start_frame": int(
                sample["abnormal_start"]
            ),

            "dataset": "DADA2000",
            "frame_dir": video_id,
            "filename_tmpl": "{:04d}.png",
            "type": "DADA2000",
            "start_index": self.start_index,

            "total_frames": int(
                sample["total_frames"]
            ),

            "is_val": is_val,
            "is_test": is_test,
        }


# ================================================================
# Collate
# ================================================================

@FUNCTIONS.register_module()
def mistn_collate_fn(batch):

    return {
        "inputs": {
            "motion": torch.stack(
                [item["motion"] for item in batch]
            ),

            "interaction": torch.stack(
                [item["interaction"] for item in batch]
            ),

            "motion_mask": torch.stack(
                [item["motion_mask"] for item in batch]
            ),

            "neighbor_mask": torch.stack(
                [item["neighbor_mask"] for item in batch]
            ),

            "labels": torch.stack(
                [item["labels"] for item in batch]
            ),

            "frames": torch.stack(
                [item["frames"] for item in batch]
            ),

            "accident_frame": torch.stack(
                [item["accident_frame"] for item in batch]
            ),

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
