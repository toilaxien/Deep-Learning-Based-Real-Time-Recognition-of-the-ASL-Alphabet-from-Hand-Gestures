# ASL Cross-Domain Recognition

Repository for the paper:

**Deep Learning Based Real-Time Recognition of the ASL Alphabet from Hand Gestures**

This repository contains the training code, evaluation code, model configuration files, hyperparameters, inference scripts, trained weights, annotation format, dataset documentation, and representative cross-domain samples for the evaluated ASL alphabet recognition models.

## Study Summary

The paper evaluates eight deep learning architectures for static ASL alphabet recognition under a three-tier protocol:

1. Internal benchmark testing on ASL Alphabet.
2. Independent cross-domain validation under unconstrained conditions.
3. Real-time video inference.

The main goal is to quantify the gap between near-perfect benchmark performance and real-world deployment robustness.

## Models

| Branch | Model | Weight File |
|---|---|---|
| Classification | Custom CNN | `weights/custom_cnn_asl.keras` |
| Classification | EfficientNetB0 | `weights/efficientnetb0_asl.keras` |
| Classification | ResNet50V2 | `weights/resnet50v2_asl.keras` |
| Classification | Vision Transformer | `weights/ViT.pth` |
| Classification | YOLOv11m-cls | `weights/YoLo11m_csf.pt` |
| Detection | YOLOv11m | `weights/YoLo11m_dectection.pt` |
| Detection | YOLOv12x | `weights/YoLo12x.pt` |
| Detection | RT-DETR-L | `weights/rtdetr-l.pt` |

## Repository Structure

```text
configs/                 YAML model configurations and hyperparameters.
dataset_docs/            Dataset documentation, metadata schema, annotation notes.
evaluation/              Scripts for classification, detection, and realtime evaluation.
inference/               Webcam app and deployment model folder.
sample_cross_domain/     Public representative cross-domain samples, one image per class.
training/                Training notebooks for all eight architectures.
weights/                 Trained model weights used in the study.
```

## Installation

Recommended:

- Python 3.10 or 3.11 for full TensorFlow/Keras compatibility.
- CUDA-capable GPU for practical runtime, although CPU execution is possible for small smoke tests.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If TensorFlow is not available for your current Python version, use Python 3.10 or 3.11 for Keras model evaluation.

## Data Availability

Benchmark dataset:

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Private independent cross-domain dataset:

- 870 RGB images.
- 29 ASL classes.
- 30 images per class.
- 5 signers.
- Multiple cameras, lighting conditions, and backgrounds.

The full cross-domain set is not publicly released because it contains participant images collected under informed consent. Representative samples and metadata documentation are provided:

- `sample_cross_domain/`
- `dataset_docs/class_distribution.csv`
- `dataset_docs/cross_domain_metadata_template.csv`
- `dataset_docs/annotation_guidelines.md`
- `dataset_docs/inter_annotator_agreement.md`

Researchers requiring access to the complete cross-domain dataset may contact the corresponding author:

```text
lethivinhthanh@iuh.edu.vn
```

Requests should include a brief description of the intended use.

## Expected Dataset Layout

Evaluation scripts expect class-folder data:

```text
dataset_root/
  A/
  B/
  C/
  ...
  Z/
  del/
  nothing/
  space/
```

## Quick Smoke Test

Use the public representative samples to verify that scripts can run:

```powershell
python evaluation\eval_classification.py --data sample_cross_domain --output results\cross_domain_sample
python evaluation\eval_detection.py --data sample_cross_domain --output results\detection_sample
```

The sample folder has only one image per class, so these smoke-test outputs are not the paper results.

## Reproducing Evaluations

| Evaluation | Script | Required Data |
|---|---|---|
| Classification benchmark/cross-domain | `evaluation/eval_classification.py` | Class-folder image dataset |
| Detection benchmark/cross-domain | `evaluation/eval_detection.py` | Class-folder image dataset; script creates YOLO full-image labels |
| Realtime performance | `evaluation/eval_realtime_video.ipynb` | Original labeled videos and model paths |

Classification cross-domain:

```powershell
python evaluation\eval_classification.py --data path\to\cross_domain_870 --output results\cross_domain
```

Detection cross-domain:

```powershell
python evaluation\eval_detection.py --data path\to\cross_domain_870 --output results\cross_domain_detection
```

Realtime evaluation is provided in `evaluation/eval_realtime_video.ipynb`. The notebook evaluates labeled videos frame by frame: it detects the hand with a YOLO hand detector, crops the largest hand ROI, runs the selected ASL model, records frame-level predictions, computes frame accuracy, generates a classification report and confusion matrix, and reports average latency/FPS. Update the Google Drive paths in the notebook before rerunning.

## Webcam Demo

```powershell
python inference\webcam_app.py
```

The current webcam application uses:

- YOLO12x for hand localization.
- ViT for ASL class prediction on the cropped hand region.

Press `Esc` to exit the OpenCV window.

## Repository Validation

Before upload or review, run:

```powershell
python tools\validate_repository.py
```

This checks required files, folders, configs, weights, labels, representative samples, and reported result CSV schemas.
This checks required files, folders, configs, weights, labels, and representative samples.

## Important Reproducibility Note

Exact reproduction of the manuscript numbers requires the exact private cross-domain dataset, the exact realtime videos, and the released trained weights. The public sample set is provided for transparency and smoke testing, not as a replacement for the private evaluation data.

The public repository is designed to satisfy the reproducibility items requested by reviewers: training/evaluation code, configuration files, hyperparameters, inference scripts, trained weights, annotation format, anonymized representative samples, metadata distributions, annotation guidelines, and access instructions for the private full cross-domain dataset.
