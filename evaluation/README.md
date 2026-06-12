# Evaluation Scripts

This folder contains reproducibility scripts for the paper's evaluation protocol.

## Files

| File | Purpose | Related Paper Table |
|---|---|---|
| `eval_realtime_video.ipynb` | Original Colab notebook for realtime video evaluation with YOLO hand crop and multiple model types | Table 5 |
| `eval_classification.py` | Computes Accuracy, macro Precision, macro Recall, and macro F1 for classification models | Table 2 |
| `eval_detection.py` | Computes Precision, Recall, mAP50, and mAP50-95 for detection models through Ultralytics validation | Table 4 |
| `video_manifest_template.csv` | Template manifest for realtime evaluation videos | Table 5 |

## Classification Evaluation

Input format:

```text
dataset_root/
  A/
  B/
  ...
  Z/
  del/
  nothing/
  space/
```

Run:

```powershell
python evaluation\eval_classification.py --data path\to\dataset_root --output results\cross_domain
```

Outputs:

- `classification_summary.csv`
- `classification_summary.md`
- `{model}_predictions.csv`
- `{model}_classification_report.txt`
- `{model}_confusion_matrix.csv`

Use the private 870-image cross-domain dataset to reproduce Table 2. Use `sample_cross_domain/` only for smoke testing.

## Detection Evaluation

The detection models were trained with YOLO labels. If the input is class-folder data, the script creates a temporary YOLO validation dataset with a full-image box for every image:

```text
class_id 0.5 0.5 1.0 1.0
```

Run:

```powershell
python evaluation\eval_detection.py --data path\to\dataset_root --output results\cross_domain_detection
```

Optional: keep the generated YOLO-format data for inspection:

```powershell
python evaluation\eval_detection.py --data path\to\dataset_root --output results\cross_domain_detection --keep-yolo-data results\generated_yolo_eval
```

Outputs:

- `detection_summary.csv`
- `detection_summary.md`
- Ultralytics validation artifacts.

Use the private 870-image cross-domain dataset to reproduce Table 4.

## Real-Time Evaluation

Real-time evaluation is provided in `eval_realtime_video.ipynb`. This is the original Colab-style notebook used for the paper's video-based evaluation.

### Input

The notebook expects a folder of labeled videos. In the original code, the class label is read from the video filename:

```text
Video/
  A.mp4
  B.mp4
  ...
  Z.mp4
  delete.mp4
  nothing.mp4
  space.mp4
```

The labels used in the notebook are:

```text
A-Z, delete, nothing, space
```

Note: the training/inference labels elsewhere in this repository use `del`, while this realtime notebook uses `delete`. Keep the label naming consistent with the video filenames when rerunning the notebook.

### Pipeline

For each frame in each video, the notebook performs:

1. Read a frame with OpenCV.
2. Detect the hand using a YOLO hand detector.
3. Select the largest detected hand bounding box.
4. Crop the hand region of interest.
5. Resize the crop to `224 x 224` for TensorFlow and ViT models; YOLO models receive the crop directly.
6. Run the selected ASL model.
7. Convert the model output to a predicted class label.
8. Store frame-level predictions and inference latency.

The model loader supports three model families:

- `tf`: TensorFlow/Keras classifiers such as Custom CNN, EfficientNetB0, and ResNet50V2.
- `yolo`: Ultralytics YOLO classification or detection models.
- `vit`: PyTorch/timm Vision Transformer checkpoint.

### Metrics and Outputs

The notebook computes and saves:

- Frame-level predictions: `predictions.csv`.
- Overall frame accuracy from `true_label == pred_label`.
- `classification_report` from scikit-learn.
- Confusion matrix image: `confusion_matrix.png`.
- Average per-frame model latency in milliseconds: `avg_latency_ms`.
- Average FPS computed as `1000 / avg_latency_ms`.
- Cropped ROI samples every 100 frames under `samples/`.

In the original notebook, outputs are saved to:

```text
/content/drive/MyDrive/AllModel/RealTime_Results/{model_name}
```

Before rerunning, update these Google Drive paths:

- `VIDEO_DIR`
- `MODEL_PATH`
- `MODEL_TYPE`
- `MODEL_NAME`
- `hand_detector` path
- `save_dir`

Reproducing the full Table 5 requires the original labeled video set and per-model paths.
