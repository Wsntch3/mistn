import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mmaction.registry import DATASETS
from mmengine.registry import FUNCTIONS

from taa.splits import dada_test


@DATASETS.register_module()
class MISTNDataset(Dataset):
    """
    MI-STN dataset.

    Split:
        TRAIN:
            mengikuti RiskProp/TAA:
            video yang ada di dada_test dikeluarkan dari train.

        VAL:
            menggunakan seluruh feature yang tersedia di folder val
            dan tidak difilter dengan dada_test.

        TEST:
            menggunakan seluruh feature yang tersedia di folder test
            dan tidak difilter dengan dada_test.

    Satu video dapat menghasilkan:
        - positive sample
        - negative sample

    Frame window sampling:
        Mengikuti RiskProp `SampleFramesBeforeAccident`:

        - test_mode=True (val/test):
            positive -> window fixed, berakhir tepat di accident_frame
            negative -> window fixed, berakhir tepat 3 detik sebelum accident

        - test_mode=False (train):
            positive -> window berakhir di accident_frame + jitter kecil
                        (0 s/d frame_interval), supaya model tidak
                        menghafal titik akhir yang selalu persis sama
            negative -> window diacak di SELURUH rentang aman
                        (dari awal video sampai 3 detik sebelum accident),
                        bukan fixed di satu titik saja

        Randomisasi window negative saat training ini penting: tanpa ini,
        model bisa "menghafal" jarak waktu tetap terhadap accident_frame
        alih-alih belajar sinyal risiko yang sesungguhnya dari motion dan
        interaction (shortcut learning).
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

        # test_mode menentukan strategi sampling window (lihat docstring).
        # Kalau tidak diberikan eksplisit, ikuti split (train -> False,
        # val/test -> True), supaya tidak bergantung pada config eksternal.
        self.test_mode = bool(test_mode) or (split != "train")

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

        # ========================================================
        # Load annotation DADA
        # ========================================================

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

        # ========================================================
        # Build samples
        #
        # PENTING: sekarang HANYA metadata yang disimpan di sini.
        # Frame window (positif/negatif) TIDAK dihitung di __init__
        # lagi -- itu dipindah ke __getitem__ lewat _sample_frame_inds(),
        # supaya bisa diacak setiap epoch saat training.
        # ========================================================

        self.samples = []

        matched_videos = 0
        skipped_videos = 0

        # Statistik debugging
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

        # ========================================================
        # LOOP VIDEO
        # ========================================================

        for video_dir in video_dirs:

            video_id = video_dir.name

            # ----------------------------------------------------
            # 1. Harus ada annotation
            # ----------------------------------------------------

            if video_id not in self.annotations:
                skipped_videos += 1
                skipped_no_annotation += 1
                continue

            # ----------------------------------------------------
            # 2. SPLIT
            #
            # HANYA TRAIN menggunakan dada_test.
            # ----------------------------------------------------

            if split == "train":

                in_dada_test = video_id in dada_test

                if in_dada_test:
                    skipped_videos += 1
                    skipped_not_in_split += 1
                    continue

            # ----------------------------------------------------
            # 3. Feature motion harus tersedia
            # ----------------------------------------------------

            motion_path = video_dir / "motion.json"

            if not motion_path.exists():
                skipped_videos += 1
                skipped_no_motion += 1
                continue

            # ----------------------------------------------------
            # 4. Feature interaction harus tersedia
            # ----------------------------------------------------

            interaction_path = video_dir / "interaction.json"

            if not interaction_path.exists():
                skipped_videos += 1
                skipped_no_interaction += 1
                continue

            ann = self.annotations[video_id]

            # ----------------------------------------------------
            # 5. Hanya accident video
            # ----------------------------------------------------

            if ann["accident"] == -1:
                skipped_videos += 1
                skipped_non_accident += 1
                continue

            # ----------------------------------------------------
            # 6. RiskProp: category 1-18
            # ----------------------------------------------------

            try:
                category_id = int(video_id.split("_")[0])
            except (ValueError, IndexError):
                skipped_videos += 1
                continue

            if not 1 <= category_id <= 18:
                skipped_videos += 1
                continue

            accident_frame = ann["accident_frame"]

            # ====================================================
            # POSITIVE SAMPLE (metadata saja)
            #
            # Sama seperti RiskProp:
            # accident_frame - start_index >= fps * 2
            # ====================================================

            positive_available = (
                accident_frame - self.start_index
                >= self.fps * 2
            )

            if positive_available:

                self.samples.append(
                    {
                        "video_id": video_id,
                        "target": True,
                        "accident_frame": accident_frame,
                        "abnormal_start": ann["abnormal_start"],
                        "abnormal_end": ann["abnormal_end"],
                        "total_frames": ann["total_frames"],
                    }
                )

            # ====================================================
            # NEGATIVE SAMPLE (metadata saja)
            #
            # Sama seperti RiskProp:
            # accident_frame - start_index >= fps * 3.5
            # ====================================================

            negative_available = (
                accident_frame - self.start_index
                >= self.fps * 3.5
            )

            if negative_available:

                self.samples.append(
                    {
                        "video_id": video_id,
                        "target": False,
                        "accident_frame": accident_frame,
                        "abnormal_start": ann["abnormal_start"],
                        "abnormal_end": ann["abnormal_end"],
                        "total_frames": ann["total_frames"],
                    }
                )

            # ----------------------------------------------------
            # Video berhasil menghasilkan sample
            # ----------------------------------------------------

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
        print(f"test_mode         : {self.test_mode}")
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

        # ========================================================
        # Print sample list
        # ========================================================

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

    # ============================================================
    # Sample frame indices (mengikuti RiskProp SampleFramesBeforeAccident)
    # ============================================================

    def _sample_frame_inds(self, accident_frame, target, total_frames):

        clip_inds = (
            np.arange(self.sequence_length, dtype=np.int64)
            * self.frame_interval
        )
        clip_inds_max = int(clip_inds[-1])

        safe_limit = (
            accident_frame - self.start_index - self.fps * 3
        )

        if self.test_mode:

            if target is True:
                # Fixed: window berakhir tepat di accident_frame
                clip_inds = (
                    clip_inds
                    + accident_frame
                    - self.start_index
                    - clip_inds_max
                )
            else:
                # Fixed: window berakhir tepat 3 detik sebelum accident
                if safe_limit > clip_inds_max:
                    clip_inds = clip_inds + 0
                else:
                    clip_inds = (
                        clip_inds + safe_limit - clip_inds_max
                    )

        else:

            if target is True:
                # Jitter kecil di sekitar accident_frame
                jitter = int(
                    np.random.randint(0, self.frame_interval)
                )
                accident_ind = (
                    accident_frame - self.start_index + jitter
                )
                accident_ind = min(
                    accident_ind, total_frames - 1
                )
                clip_inds = clip_inds + accident_ind - clip_inds_max
            else:
                # Diacak di seluruh rentang aman (bukan fixed 1 titik)
                if safe_limit > clip_inds_max:
                    offset = int(
                        np.random.randint(
                            0, safe_limit - clip_inds_max
                        )
                    )
                    clip_inds = clip_inds + offset
                else:
                    clip_inds = (
                        clip_inds + safe_limit - clip_inds_max
                    )

        frame_inds = np.maximum(clip_inds, 0) + self.start_index

        return [int(frame) for frame in frame_inds]

    # ============================================================
    # Length
    # ============================================================

    def __len__(self):
        return len(self.samples)

    # ============================================================
    # Load motion
    # ============================================================

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

    # ============================================================
    # Load interaction
    # ============================================================

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
        target = sample["target"]
        accident_frame = sample["accident_frame"]
        total_frames = sample["total_frames"]

        # Frame window dihitung DI SINI, tiap kali dipanggil,
        # bukan sekali di __init__.
        target_frames = self._sample_frame_inds(
            accident_frame, target, total_frames
        )

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
# Collate (tidak berubah)
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
            "video_id": [item["video_id"] for item in batch],
            "target": [item["target"] for item in batch],
            "abnormal_start_frame": [
                item["abnormal_start_frame"] for item in batch
            ],
            "dataset": [item["dataset"] for item in batch],
            "frame_dir": [item["frame_dir"] for item in batch],
            "filename_tmpl": [
                item["filename_tmpl"] for item in batch
            ],
            "type": [item["type"] for item in batch],
            "start_index": [item["start_index"] for item in batch],
            "total_frames": [item["total_frames"] for item in batch],
            "is_val": [item["is_val"] for item in batch],
            "is_test": [item["is_test"] for item in batch],
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
