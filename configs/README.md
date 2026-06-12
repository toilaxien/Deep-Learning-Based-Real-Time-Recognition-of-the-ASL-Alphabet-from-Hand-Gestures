# Model Configuration Files

Each YAML file records the model identity, task type, framework, weight path, input resolution, major hyperparameters, preprocessing, augmentation, and reported metric family.

| File | Model | Task |
|---|---|---|
| `custom_cnn.yaml` | Custom CNN | Classification |
| `efficientnetb0.yaml` | EfficientNetB0 | Classification |
| `resnet50v2.yaml` | ResNet50V2 | Classification |
| `vit.yaml` | Vision Transformer | Classification |
| `yolo11m_cls.yaml` | YOLOv11m-cls | Classification |
| `yolo11m_detection.yaml` | YOLOv11m | Detection |
| `yolo12x.yaml` | YOLOv12x | Detection |
| `rtdetr_l.yaml` | RT-DETR-L | Detection |

These files are intended to make the hyperparameters used in the notebooks easier to audit without opening every notebook.
