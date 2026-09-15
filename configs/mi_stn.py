_base_ = ["_base_/default_runtime.py"]

# ============================================================
# Custom MI-STN package
# ============================================================

custom_imports = dict(
    imports="mi_stn",
    allow_failed_imports=False,
)


# ============================================================
# MI-STN Model Settings
# ============================================================

motion_input_dim = 9

motion_hidden_dim = 64
motion_output_dim = 64

interaction_hidden_dim = 64
interaction_output_dim = 64

fusion_dim = 128
temporal_dim = 128

num_horizons = 3
dropout = 0.1


# ============================================================
# Dataset Settings
# ============================================================

data_root = "data/MI-STN/features"


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
        data_root=data_root,
        split="train",
        test_mode=False,
    ),
)


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
        data_root=data_root,
        split="val",
        test_mode=True,
    ),
)


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
        data_root=data_root,
        split="test",
        test_mode=True,
    ),
)


# ============================================================
# Evaluation
# ============================================================

val_evaluator = dict(
    type="AnticipationMetric",
    fpr_max=0.1,
    vis_list=[],
    output_dir="visualizations_mi_stn",
)

test_evaluator = val_evaluator


# ============================================================
# Training
# ============================================================

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=50,
    val_begin=1,
    val_interval=1,
)


# ============================================================
# Checkpoint
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
# Custom Hooks
# ============================================================

custom_hooks = [
    dict(type="EpochHook"),
    dict(type="AnticipationMetricHook"),
]


# ============================================================
# MI-STN Model
# ============================================================

model = dict(
    type="MISTNRecognizer",

    # Motion branch
    motion_input_dim=motion_input_dim,
    motion_hidden_dim=motion_hidden_dim,
    motion_output_dim=motion_output_dim,

    # Interaction branch
    interaction_hidden_dim=interaction_hidden_dim,
    interaction_output_dim=interaction_output_dim,

    # Fusion
    fusion_dim=fusion_dim,

    # Temporal modeling
    temporal_dim=temporal_dim,

    # Risk prediction
    num_horizons=num_horizons,

    # Regularization
    dropout=dropout,
)
