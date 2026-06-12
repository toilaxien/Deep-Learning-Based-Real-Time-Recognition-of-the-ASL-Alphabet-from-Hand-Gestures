import argparse
import csv
import shutil
import tempfile
from pathlib import Path

import yaml


CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space",
]


def build_temp_yolo_dataset(data_dir, tmp_root):
    data_dir = Path(data_dir)
    tmp_root = Path(tmp_root)
    images_dir = tmp_root / "val" / "images"
    labels_dir = tmp_root / "val" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    count = 0
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for src in sorted(class_dir.rglob("*")):
            if src.suffix.lower() not in exts:
                continue
            dst_name = f"{class_name}_{count:06d}{src.suffix.lower()}"
            shutil.copy2(src, images_dir / dst_name)
            (labels_dir / f"{Path(dst_name).stem}.txt").write_text(
                f"{class_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8"
            )
            count += 1

    if count == 0:
        raise FileNotFoundError(f"No images found under {data_dir}")

    data_yaml = {
        "path": str(tmp_root.resolve()),
        "train": "val/images",
        "val": "val/images",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    yaml_path = tmp_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return yaml_path, count


def evaluate_model(model_name, model_path, data_yaml, output_dir):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        return {"model": model_name, "status": f"SKIPPED: ultralytics unavailable: {exc}"}

    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml), imgsz=224, split="val", verbose=False, plots=True, project=str(output_dir), name=model_name)
    box = metrics.box
    return {
        "model": model_name,
        "status": "OK",
        "precision": float(box.mp) * 100,
        "recall": float(box.mr) * 100,
        "map50": float(box.map50) * 100,
        "map50_95": float(box.map) * 100,
    }


def write_summary(output_dir, results):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "detection_summary.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "status", "precision", "recall", "map50", "map50_95"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    lines = ["| Model | Status | Precision | Recall | mAP50 | mAP50-95 |", "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        if r.get("status") == "OK":
            lines.append(
                f"| {r['model']} | OK | {r['precision']:.2f} | {r['recall']:.2f} | "
                f"{r['map50']:.2f} | {r['map50_95']:.2f} |"
            )
        else:
            lines.append(f"| {r.get('model', '')} | {r.get('status', '')} |  |  |  |  |")
    (output_dir / "detection_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection models using YOLO-format full-image labels.")
    parser.add_argument("--data", required=True, help="Class-folder dataset root.")
    parser.add_argument("--weights", default="weights", help="Weights directory.")
    parser.add_argument("--output", default="results/cross_domain_detection", help="Output directory.")
    parser.add_argument("--keep-yolo-data", default=None, help="Optional folder to keep generated YOLO dataset.")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights)

    if args.keep_yolo_data:
        tmp_dir = Path(args.keep_yolo_data)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        data_yaml, count = build_temp_yolo_dataset(args.data, tmp_dir)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory()
        data_yaml, count = build_temp_yolo_dataset(args.data, cleanup.name)

    print(f"Prepared YOLO validation dataset with {count} images: {data_yaml}")

    models = [
        ("YOLOv11m", weights / "YoLo11m_dectection.pt"),
        ("YOLOv12x", weights / "YoLo12x.pt"),
        ("RT-DETR-L", weights / "rtdetr-l.pt"),
    ]
    results = []
    for model_name, model_path in models:
        if model_path.exists():
            results.append(evaluate_model(model_name, model_path, data_yaml, output_dir))
        else:
            results.append({"model": model_name, "status": f"SKIPPED: missing {model_path}"})

    write_summary(output_dir, results)
    for result in results:
        print(result)

    if cleanup is not None:
        cleanup.cleanup()


if __name__ == "__main__":
    main()
