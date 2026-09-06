# Inference Demo

`webcam_app.py` is a local webcam demo.

The app loads canonical checkpoints from `../weights/`:

- Stage-1 hand localization: `yolov8s_stage1_hand_detector.pt`
- ASL classification: `ViT.pth`

Stage-1 runtime settings:

- `imgsz=640`
- `conf=0.25`

No model checkpoint should be stored under `inference/`.
