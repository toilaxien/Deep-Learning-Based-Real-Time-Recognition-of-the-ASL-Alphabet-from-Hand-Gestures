# ASL Alphabet Recognition Technical Re-examination Repository

Private editor-facing repository for the paper:

**Deep Learning Based Real-Time Recognition of the ASL Alphabet from Hand Gestures**

This repository is organized for technical re-examination and correction. It keeps the canonical training code, model configs, final checkpoints, technical re-examination notebooks, final evidence, dataset documentation, validation utilities, and the webcam demo.

## Repository Layout

```text
configs/                         Model configs and hyperparameters.
dataset_docs/                    Dataset documentation and metadata templates.
evaluation/technical_reexamination/
                                  Canonical technical re-examination notebooks.
inference/                       Webcam demo only; checkpoints are loaded from weights/.
result/                          Final editor evidence and audit outputs.
sample_cross_domain/             Representative samples only.
tools/                           Repository validation utilities.
training/                        Training notebooks for the eight target models.
weights/                         Canonical checkpoints only.
```

## Evaluated Target Models

The benchmark comparison contains exactly eight target models.

| Branch | Model | Weight |
|---|---|---|
| Classification | Custom CNN | `weights/custom_cnn_asl.keras` |
| Classification | EfficientNetB0 | `weights/efficientnetb0_asl.keras` |
| Classification | ResNet50V2 | `weights/resnet50v2_asl.keras` |
| Classification | Vision Transformer | `weights/ViT.pth` |
| Classification | YOLOv11m-cls | `weights/YoLo11m_csf.pt` |
| Detection | YOLOv11m | `weights/YoLo11m_dectection.pt` |
| Detection | YOLOv12x | `weights/YoLo12x.pt` |
| Detection | RT-DETR-L | `weights/rtdetr-l.pt` |

## Supporting Stage-1 Model

`weights/yolov8s_stage1_hand_detector.pt` is a supporting YOLOv8s hand localizer used only for Stage-1 real-time hand detection.

- It is not a ninth benchmark model.
- Real-time Stage-1 settings: `imgsz=640`, `conf=0.25`.
- Its config is `configs/yolov8s_stage1_hand_detector.yaml`.
- YOLOv8-generated boxes must not be used as ground truth for detection benchmark or cross-domain mAP.

## Technical Re-examination Code

Canonical notebooks are under `evaluation/technical_reexamination/`:

- `01_dataset_audit/ijies-asl-benchmark-data-split-verification.ipynb`
- `02_classification/IJIES_Technical_Reexamination_5Model_Classification_Benchmark_CrossDomain_Evaluation.ipynb`
- `03_detection/ijies-benchmark-crossdomain-sequential-reevaluation.ipynb`
- `04_realtime/IJIES_Technical_Reexamination_RealTime_Video_Evaluation.ipynb`

Old evaluation scripts have been removed from the canonical tree and quarantined outside the repo for final human review before deletion.

## Final Evidence

Canonical editor evidence is under `result/`:

- `result/01_Benchmark_CrossDomain/Classification/IJIES_EDITOR_CLASSIFICATION_BENCHMARK_CROSSDOMAIN/`
- `result/01_Benchmark_CrossDomain/Detection/IJIES_EDITOR_DETECTOR_BENCHMARK_CROSSDOMAIN/`
- `result/02_RealTime_Video/IJIES_EDITOR_REALTIME_EVIDENCE/`
- `result/03_Audits_and_Checksums/`

Review `result/03_Audits_and_Checksums/overlap_checks/SUBMISSION_STATUS_REVIEW_REQUIRED.md` before sending. The evidence file reports benchmark/cross-domain SHA256 overlap and must be reconciled with the manuscript text.

## Data Availability and Privacy

The full independent cross-domain dataset and real-time videos are private because they contain participant images collected under informed consent. The public-facing repository includes representative samples, dataset documentation, manifests, and evidence outputs, but not the complete private image/video data.

Current canonical cross-domain protocol:

- 870 images.
- 29 classes.
- 30 images per class.
- Representative samples are in `sample_cross_domain/`.

## Installation

Python 3.10 or 3.11 is recommended for TensorFlow/Keras compatibility.

```powershell
python -m pip install -r requirements.txt
```

## Webcam Demo

```powershell
python inference\webcam_app.py
```

The demo loads the YOLOv8s Stage-1 hand detector and ViT classifier from `weights/`.

## Repository Validation

Before the canonical commit for editor submission, run:

## Demo

![Recording-2025-11-16-154042 (1)](https://github.com/user-attachments/assets/9d13d8e9-d0a7-4ad8-b97a-18d086f27c32)

```powershell
python tools\validate_repository.py
```

The validator checks target checkpoints, supporting YOLOv8s checkpoint, label order, configs, canonical evidence folders, manifests, SHA256 files, and duplicate checkpoint ambiguity.
