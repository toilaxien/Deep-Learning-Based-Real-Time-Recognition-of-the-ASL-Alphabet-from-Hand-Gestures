# Canonical Checkpoints

This folder is the only canonical checkpoint location in the repository. The webcam app and evaluation notebooks should load checkpoints from here.

## Eight Evaluated Target Models

| File | Model | Task | Size Bytes | SHA256 |
|---|---|---|---:|---|
| `custom_cnn_asl.keras` | Custom CNN | Classification | 148680943 | `4B7596B07A2F5DC798C57B535576C1D35021A27719AF48797BB04EFF868F6475` |
| `efficientnetb0_asl.keras` | EfficientNetB0 | Classification | 49704964 | `2877979DAC0ACCC796F066BA171620487A6E18675434AC93D3FF5CC7F9634661` |
| `resnet50v2_asl.keras` | ResNet50V2 | Classification | 283939880 | `25A8D26894E139DF3048614119EE3020843CC881F91B93D1F8E1F5ED61358EEC` |
| `ViT.pth` | Vision Transformer | Classification | 343345878 | `F3DF543566BBAFF65654F979C3B384C9373DD8DDD96CA49C6827AA29AAB33BCB` |
| `YoLo11m_csf.pt` | YOLOv11m-cls | Classification | 20946639 | `E03FE2CFEABBD677CA70A137DD5CDFD2AF15AEF618812AD9BD7439CCA7A7AF24` |
| `YoLo11m_dectection.pt` | YOLOv11m | Detection | 40507173 | `58197E6B500AB02F9DF6CBDA3272EF06E638D7C104CBFFE2E76264C3213EBDCA` |
| `YoLo12x.pt` | YOLOv12x | Detection | 119100218 | `37E67D095D46276B0E9FA24D71F1FCB7AF1D94F17EE7BC4326DD14B353D26A76` |
| `rtdetr-l.pt` | RT-DETR-L | Detection | 66248192 | `51ED37D1799832D1934CE277F3BCEC5DB4517BE0603F7A0B81BD1ED4CBFA3971` |

## Supporting Stage-1 Model

| File | Model | Role | Size Bytes | SHA256 |
|---|---|---|---:|---|
| `yolov8s_stage1_hand_detector.pt` | YOLOv8s | Stage-1 hand localization for real-time inference | 22507643 | `70B540063FBC385736D8258970744A4AFBC4CBF7932134BAE3B24CDADEADEC06` |

The YOLOv8s Stage-1 detector is a supporting pipeline component and is not counted as a ninth evaluated target model.

The checkpoint used in this study was obtained from the publicly released `hand_yolov8s.pt` model in the Bingsu/adetailer Hugging Face repository. Its SHA-256 hash matches the publicly released checkpoint. The provider reports mAP50 = 79.4% and mAP50–95 = 52.7%; separate Precision and Recall values are not reported.
