_base_ = [
    "_base_/schedules/sgd_50e.py",
    "_base_/default_runtime.py",
]

# ============================================================
# IMPORT
# TAA = framework, evaluator, hooks, runtime
# MI-STN = dataset + model
# ============================================================

custom_imports = dict(
    imports=[
        "taa",
        "mi_stn",
    ],
    allow_failed_imports=False,
)

default_scope = "mmaction"


# ============================================================
# MI-STN SETTINGS
# ============================================================

annotation_file = "/content/data_text_annotation.xlsx"
feature_root = "/content/MI-STN/features"

sequence_length = 20
max_objects = 50
max_neighbors = 5

motion_input_dim = 9
motion_hidden_dim = 64
motion_output_dim = 64

interaction_input_dim = 9
interaction_hidden_dim = 64
interaction_output_dim = 64

fusion_dim = 128
temporal_dim = 128

num_horizons = 3
horizons = (1, 2, 3)

fps = 30
stride = 10

dropout = 0.1


# ============================================================
# TRAIN DATALOADER
# ============================================================

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type="DefaultSampler",
        shuffle=True,
    ),
    dataset=dict(
        type="MISTNDataset",

        annotation_file=annotation_file,
        feature_root=feature_root,

        sequence_length=sequence_length,
        max_objects=max_objects,
        max_neighbors=max_neighbors,

        horizons=horizons,
        fps=fps,
        stride=stride,

        split="train",
        test_mode=False,
    ),

    collate_fn=dict(
        type="mistn_collate_fn",
    ),
)


# ============================================================
# VALIDATION DATALOADER
# ============================================================

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=dict(
        type="MISTNDataset",

        annotation_file=annotation_file,
        feature_root=feature_root,

        sequence_length=sequence_length,
        max_objects=max_objects,
        max_neighbors=max_neighbors,

        horizons=horizons,
        fps=fps,
        stride=stride,

        split="val",
        test_mode=True,
    ),

    collate_fn=dict(
        type="mistn_collate_fn",
    ),
)


# ============================================================
# TEST DATALOADER
# ============================================================

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=dict(
        type="MISTNDataset",

        annotation_file=annotation_file,
        feature_root=feature_root,

        sequence_length=sequence_length,
        max_objects=max_objects,
        max_neighbors=max_neighbors,

        horizons=horizons,
        fps=fps,
        stride=stride,

        split="test",
        test_mode=True,
    ),

    collate_fn=dict(
        type="mistn_collate_fn",
    ),
)


# ============================================================
# EVALUATION
# Tetap menggunakan evaluator asli TAA / RiskProp
# ============================================================

val_evaluator = dict(
    type="AnticipationMetric",
    fpr_max=0.1,
    vis_list=[],
    output_dir="visualizations_mi_stn",
)

test_evaluator = val_evaluator


# ============================================================
# TRAINING
# 50 EPOCHS
# Optimizer + scheduler berasal dari TAA:
# _base_/schedules/sgd_50e.py
# ============================================================

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=50,
    val_begin=1,
    val_interval=1,
)

val_cfg = dict(
    type="ValLoop",
)

test_cfg = dict(
    type="TestLoop",
)

resume = False


# ============================================================
# CHECKPOINT
# Mengikuti pola TAA
# ============================================================

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=5,
        save_best="mAUC@",
        rule="greater",
    )
)


# ============================================================
# TAA HOOKS
# Jangan diubah.
# ============================================================

custom_hooks = [
    dict(type="EpochHook"),
    dict(type="AnticipationMetricHook"),
]


# ============================================================
# MI-STN MODEL
# ============================================================

model = dict(
    type="MISTNRecognizer",

    # --------------------------------------------------------
    # Motion branch
    # --------------------------------------------------------
    motion_input_dim=motion_input_dim,
    motion_hidden_dim=motion_hidden_dim,
    motion_output_dim=motion_output_dim,

    # --------------------------------------------------------
    # Interaction branch
    # --------------------------------------------------------
    interaction_input_dim=interaction_input_dim,
    interaction_hidden_dim=interaction_hidden_dim,
    interaction_output_dim=interaction_output_dim,

    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------
    fusion_dim=fusion_dim,

    # --------------------------------------------------------
    # Temporal modeling
    # --------------------------------------------------------
    temporal_dim=temporal_dim,

    # --------------------------------------------------------
    # Risk prediction
    # --------------------------------------------------------
    num_horizons=num_horizons,

    # --------------------------------------------------------
    # Regularization
    # --------------------------------------------------------
    dropout=dropout,
)


# ============================================================
# WORK DIRECTORY
# ============================================================

work_dir = "/content/RiskProp/work_dirs/mi_stn"
