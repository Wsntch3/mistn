# MI-STN: Motion-Interaction Spatio-Temporal Network for Traffic Accident Anticipation

## Overview

This repository contains the additional implementation developed for **MI-STN (Motion-Interaction Spatio-Temporal Network)**, a research model for early traffic accident anticipation.

MI-STN is developed as an extension for experimental comparison with **RiskProp**, which is used as the baseline framework. The original RiskProp repository is maintained by its original authors and can be accessed here:

**Original RiskProp repository:**
https://github.com/xingyueye5/RiskProp

This repository is **not the original RiskProp repository**. It contains additional code and configuration required to implement and evaluate MI-STN while reusing the RiskProp framework and environment.

The proposed MI-STN models traffic risk by jointly considering:

* individual object motion,
* dynamic interactions between surrounding objects,
* temporal dependencies, and
* multi-horizon accident risk prediction.

The overall architecture is:

```text
Motion Features
      │
      ▼
 Motion Module
      │
      ▼
Motion Embedding ─────────────┐
                             │
Interaction Features          │
      │                      ▼
      ▼                  Fusion
Interaction Module             │
      │                      ▼
      ▼                     GRU
Interaction Embedding          │
                             ▼
                       Risk Prediction
```

## Relationship to RiskProp

RiskProp is used as the **baseline model** for this research.

The original RiskProp framework provides the training, evaluation, dataset handling, and accident anticipation infrastructure. This repository adds the MI-STN-specific implementation without changing the original research repository.

```text
Original RiskProp
https://github.com/xingyueye5/RiskProp
        │
        │ baseline framework
        ▼
   RiskProp environment
        │
        ├── Original RiskProp code
        │
        └── Additional MI-STN implementation
                    │
                    ├── Motion Module
                    ├── Interaction Module
                    ├── Fusion Head
                    └── MI-STN Recognizer
```

For reproducibility, the original RiskProp repository should be consulted for its installation requirements, dataset preparation, and baseline training procedures.

## Repository Structure

The additional implementation in this repository is organized as follows:

```text
mistn/
├── configs/
│   ├── predict_anomaly_frame.py
│   └── mi_stn.py
│
└── mi_stn/
    ├── __init__.py
    ├── motion_module.py
    ├── interaction_module.py
    ├── fusion_head.py
    └── mi_stn_recognizer.py
```

The `mi_stn/` directory contains the implementation of the proposed model, while `configs/mi_stn.py` contains the configuration used to run MI-STN experiments.

The original RiskProp source code remains part of the baseline environment when this repository is used alongside RiskProp.

## MI-STN Components

### Motion Module

The Motion Module processes object-level motion features extracted from tracked trajectories.

The current motion representation contains:

```text
x
y
vx
vy
speed
ax
ay
accel
direction
```

These features are transformed into a latent motion representation before being passed to the interaction modeling stage.

### Interaction Module

The Interaction Module models relationships between surrounding traffic objects.

The interaction representation is derived from object trajectories and includes information such as:

```text
distance
dx
dy
relative_vx
relative_vy
relative_speed
closing_speed
direction_difference
interaction_strength
```

The purpose of this module is to capture dynamic interactions that may indicate an increasing accident risk.

### Fusion and Temporal Modeling

Motion and interaction representations are combined through the Fusion Head.

The fused representation is then processed using a GRU to model temporal dependencies before producing accident-risk predictions.

### Risk Prediction

MI-STN is designed to predict accident risk at multiple temporal horizons before the accident.

The exact temporal horizon configuration follows the evaluation setting used for comparison with the RiskProp baseline.

## Dataset

The experiments use the **DADA-2000** traffic accident anticipation dataset.

DADA-2000 provides dashcam videos and accident-related annotations. Object trajectories are generated separately using an object detection and tracking pipeline.

The preprocessing pipeline used in this research is:

```text
DADA-2000
    │
    ▼
Object Detection
    │
    ▼
ByteTrack
    │
    ▼
Object Trajectories
    │
    ├───────────────┐
    ▼               ▼
Motion Features   Interaction Features
    │               │
    └───────┬───────┘
            ▼
          MI-STN
```

The extracted features are stored separately from the model source code and are not included in this repository because of their large storage requirements.

## Installation

MI-STN is intended to run within the **RiskProp environment**.

First, clone the original RiskProp repository:

```bash
git clone https://github.com/xingyueye5/RiskProp.git
cd RiskProp
```

Install the dependencies according to the original RiskProp repository.

Then copy or clone the MI-STN implementation into the RiskProp environment.

The MI-STN configuration can then be used with:

```bash
tools/train.py configs/mi_stn.py
```

> The exact installation procedure and dependency versions should follow the original RiskProp repository.

## Training

The RiskProp baseline can be trained using its original configuration and training procedure.

MI-STN uses the additional configuration:

```text
configs/mi_stn.py
```

The model is specified as:

```python
model = dict(
    type="MISTNRecognizer",
    ...
)
```

The custom model is automatically registered through:

```python
custom_imports = dict(
    imports="mi_stn",
    allow_failed_imports=False,
)
```

## Evaluation

MI-STN is evaluated using the same accident anticipation evaluation framework as the RiskProp baseline where applicable.

The main evaluation metrics include:

| Metric     | Description                        |
| ---------- | ---------------------------------- |
| `mAUC@`    | Mean partial AUC with FPR ≤ 0.1    |
| `mAUC`     | Mean full-curve AUC                |
| `mAP`      | Mean Average Precision             |
| `mTTA@0.1` | Mean Time-To-Accident at FPR ≤ 0.1 |

The same dataset split and evaluation conditions should be used when comparing MI-STN with RiskProp to ensure a fair comparison.

## Baseline Comparison

The main experimental comparison is:

```text
RiskProp
   │
   │ Baseline
   ▼
DADA-2000
   │
   ├───────────────┐
   │               │
   ▼               ▼
RiskProp         MI-STN
Baseline        Proposed Model
   │               │
   └───────┬───────┘
           ▼
      Same Evaluation
           │
           ▼
       Comparison
```

The comparison focuses on whether explicitly modeling object motion and dynamic inter-object interactions improves early accident anticipation performance.

## Research Status

This repository is intended for research and experimental purposes.

The current implementation is under development and may be modified as the MI-STN architecture, preprocessing pipeline, and experimental configuration are refined.

## Acknowledgement

This work uses the **RiskProp** framework as the baseline for experimental comparison.

Please refer to the original RiskProp repository for the original implementation, documentation, and citation:

https://github.com/xingyueye5/RiskProp

## Citation

If you use the original RiskProp framework, please cite the original authors:

```bibtex
@InProceedings{Zou_2026_CVPR,
    author    = {Zou, Yiyang and Zhao, Tianhao and Xiao, Peilun and Jin, Hongyu and Qi, Longyu and Li, Yuxuan and Liang, Liyin and Qian, Yifeng and Lai, Chunbo and Lin, Yutian and Li, Zhihui and Wu, Yu},
    title     = {RiskProp: Collision-Anchored Self-Supervised Risk Propagation For Early Accident Anticipation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {2768-2777}
}
```

## License

The original RiskProp project is released under the Apache 2.0 License.

Please refer to the original repository for the complete license and attribution information:

https://github.com/xingyueye5/RiskProp

```
```
