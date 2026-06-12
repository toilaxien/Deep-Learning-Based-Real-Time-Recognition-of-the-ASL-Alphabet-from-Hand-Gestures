import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix


CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space",
]


def list_images(data_dir):
    data_dir = Path(data_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    items = []
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in exts:
                items.append((path, class_name))
    if not items:
        raise FileNotFoundError(f"No class-folder images found under {data_dir}")
    return items


def compute_metrics(y_true, y_pred):
    labels = list(range(len(CLASS_NAMES)))
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "accuracy": acc * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
    }


def save_outputs(output_dir, model_name, y_true, y_pred, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    (output_dir / f"{model_name}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    np.savetxt(output_dir / f"{model_name}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    with open(output_dir / f"{model_name}_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "true_label", "pred_label", "confidence"])
        writer.writeheader()
        writer.writerows(rows)


def eval_keras(model_path, items, output_dir, model_name):
    try:
        import tensorflow as tf
    except Exception as exc:
        return {"model": model_name, "status": f"SKIPPED: TensorFlow unavailable: {exc}"}

    model = tf.keras.models.load_model(model_path, compile=False)
    y_true, y_pred, rows = [], [], []
    start = time.perf_counter()
    for path, true_label in items:
        img = Image.open(path).convert("RGB").resize((224, 224))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        pred = model.predict(arr[None, ...], verbose=0)[0]
        pred_id = int(np.argmax(pred))
        true_id = CLASS_NAMES.index(true_label)
        y_true.append(true_id)
        y_pred.append(pred_id)
        rows.append({
            "path": str(path),
            "true_label": true_label,
            "pred_label": CLASS_NAMES[pred_id],
            "confidence": float(pred[pred_id]),
        })
    metrics = compute_metrics(y_true, y_pred)
    metrics.update({"model": model_name, "status": "OK", "seconds": time.perf_counter() - start})
    save_outputs(output_dir, model_name, y_true, y_pred, rows)
    return metrics


def eval_vit(model_path, items, output_dir):
    try:
        import torch
        import timm
        from torchvision import transforms
    except Exception as exc:
        return {"model": "ViT", "status": f"SKIPPED: PyTorch/timm unavailable: {exc}"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASS_NAMES))
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    y_true, y_pred, rows = [], [], []
    start = time.perf_counter()
    with torch.no_grad():
        for path, true_label in items:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            probs = torch.softmax(model(tensor), dim=1)[0]
            conf, pred_id = torch.max(probs, dim=0)
            pred_id = int(pred_id)
            true_id = CLASS_NAMES.index(true_label)
            y_true.append(true_id)
            y_pred.append(pred_id)
            rows.append({
                "path": str(path),
                "true_label": true_label,
                "pred_label": CLASS_NAMES[pred_id],
                "confidence": float(conf),
            })
    metrics = compute_metrics(y_true, y_pred)
    metrics.update({"model": "ViT", "status": "OK", "seconds": time.perf_counter() - start})
    save_outputs(output_dir, "ViT", y_true, y_pred, rows)
    return metrics


def eval_yolo_cls(model_path, items, output_dir):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        return {"model": "YOLOv11m-cls", "status": f"SKIPPED: ultralytics unavailable: {exc}"}

    model = YOLO(str(model_path))
    names = model.names
    y_true, y_pred, rows = [], [], []
    start = time.perf_counter()
    for path, true_label in items:
        result = model.predict(str(path), imgsz=224, verbose=False)[0]
        pred_id_model = int(result.probs.top1)
        conf = float(result.probs.top1conf)
        pred_label = names[pred_id_model]
        true_id = CLASS_NAMES.index(true_label)
        pred_id = CLASS_NAMES.index(pred_label) if pred_label in CLASS_NAMES else -1
        y_true.append(true_id)
        y_pred.append(pred_id)
        rows.append({
            "path": str(path),
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": conf,
        })
    metrics = compute_metrics(y_true, y_pred)
    metrics.update({"model": "YOLOv11m-cls", "status": "OK", "seconds": time.perf_counter() - start})
    save_outputs(output_dir, "YOLOv11m-cls", y_true, y_pred, rows)
    return metrics


def write_summary(output_dir, results):
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "status", "accuracy", "precision", "recall", "f1", "seconds"]
    with open(output_dir / "classification_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    lines = ["| Model | Status | Accuracy | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(
            f"| {r.get('model', '')} | {r.get('status', '')} | "
            f"{r.get('accuracy', 0):.2f} | {r.get('precision', 0):.2f} | "
            f"{r.get('recall', 0):.2f} | {r.get('f1', 0):.2f} |"
            if r.get("status") == "OK" else
            f"| {r.get('model', '')} | {r.get('status', '')} |  |  |  |  |"
        )
    (output_dir / "classification_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate classification models on a class-folder ASL dataset.")
    parser.add_argument("--data", required=True, help="Dataset root with one folder per class.")
    parser.add_argument("--weights", default="weights", help="Weights directory.")
    parser.add_argument("--output", default="results/cross_domain", help="Output directory.")
    args = parser.parse_args()

    items = list_images(args.data)
    weights = Path(args.weights)
    output_dir = Path(args.output)

    jobs = [
        ("CustomCNN", weights / "custom_cnn_asl.keras", eval_keras),
        ("EfficientNetB0", weights / "efficientnetb0_asl.keras", eval_keras),
        ("ResNet50V2", weights / "resnet50v2_asl.keras", eval_keras),
    ]

    results = []
    for model_name, model_path, fn in jobs:
        if model_path.exists():
            results.append(fn(model_path, items, output_dir, model_name))
        else:
            results.append({"model": model_name, "status": f"SKIPPED: missing {model_path}"})

    vit_path = weights / "ViT.pth"
    results.append(eval_vit(vit_path, items, output_dir) if vit_path.exists() else {"model": "ViT", "status": f"SKIPPED: missing {vit_path}"})

    yolo_cls_path = weights / "YoLo11m_csf.pt"
    results.append(eval_yolo_cls(yolo_cls_path, items, output_dir) if yolo_cls_path.exists() else {"model": "YOLOv11m-cls", "status": f"SKIPPED: missing {yolo_cls_path}"})

    write_summary(output_dir, results)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
