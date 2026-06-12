# Training Notebooks

This folder contains the notebooks used to train the eight architectures evaluated in the paper.

| Notebook | Model | Framework | Notes |
|---|---|---|---|
| `train_cnn.ipynb` | Custom CNN | TensorFlow/Keras | Classification on class folders |
| `train_efficientnetb0.ipynb` | EfficientNetB0 | TensorFlow/Keras | Classification on class folders |
| `train_Resnet50v2.ipynb` | ResNet50V2 | TensorFlow/Keras | Classification on class folders |
| `train_VIT.ipynb` | ViT | PyTorch/timm | Classification on class folders |
| `train_YoLov11m_CLS.ipynb` | YOLOv11m-cls | Ultralytics | Classification |
| `train_YoLo11m_Detection.ipynb` | YOLOv11m | Ultralytics | Detection with full-image boxes |
| `train_YoLo12x.ipynb` | YOLOv12x | Ultralytics | Detection with YOLO labels |
| `train_RTDETR.ipynb` | RT-DETR-L | Ultralytics | Detection with YOLO labels |

Most notebooks were originally executed on Kaggle/Colab and contain platform-specific paths such as `/kaggle/input/...`. Replace these paths with local dataset paths before rerunning.

The consolidated hyperparameters are also recorded in `configs/`.
