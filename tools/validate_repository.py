import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "inference/webcam_app.py",
    "inference/label.txt",
    "evaluation/eval_realtime_video.ipynb",
    "evaluation/eval_classification.py",
    "evaluation/eval_detection.py",
    "evaluation/video_manifest_template.csv",
    "dataset_docs/README.md",
    "dataset_docs/class_distribution.csv",
    "dataset_docs/cross_domain_metadata_template.csv",
    "dataset_docs/annotation_guidelines.md",
    "dataset_docs/inter_annotator_agreement.md",
]

REQUIRED_DIRS = [
    "configs",
    "dataset_docs",
    "evaluation",
    "inference",
    "sample_cross_domain",
    "training",
    "weights",
]

REQUIRED_WEIGHTS = [
    "weights/custom_cnn_asl.keras",
    "weights/efficientnetb0_asl.keras",
    "weights/resnet50v2_asl.keras",
    "weights/ViT.pth",
    "weights/YoLo11m_csf.pt",
    "weights/YoLo11m_dectection.pt",
    "weights/YoLo12x.pt",
    "weights/rtdetr-l.pt",
]

EXPECTED_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space",
]


def check(condition, message, errors):
    if condition:
        print(f"[OK] {message}")
    else:
        print(f"[MISSING] {message}")
        errors.append(message)


def check_required_paths(errors):
    for rel in REQUIRED_DIRS:
        check((ROOT / rel).is_dir(), rel, errors)
    for rel in REQUIRED_FILES:
        check((ROOT / rel).is_file(), rel, errors)
    for rel in REQUIRED_WEIGHTS:
        check((ROOT / rel).is_file(), rel, errors)


def check_labels(errors):
    label_path = ROOT / "inference" / "label.txt"
    if not label_path.exists():
        errors.append("inference/label.txt")
        return
    labels = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    check(labels == EXPECTED_CLASSES, "inference/label.txt has expected 29 labels in order", errors)


def check_configs(errors):
    config_dir = ROOT / "configs"
    yaml_files = sorted(config_dir.glob("*.yaml"))
    check(len(yaml_files) == 8, "8 YAML config files", errors)
    for path in yaml_files:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            ok = isinstance(data, dict) and "name" in data and "weights" in data and "task" in data
            check(ok, f"{path.relative_to(ROOT)} parseable with name/task/weights", errors)
        except Exception as exc:
            print(f"[ERROR] {path.relative_to(ROOT)}: {exc}")
            errors.append(str(path.relative_to(ROOT)))


def check_samples(errors):
    sample_dir = ROOT / "sample_cross_domain"
    present = sorted([p.name for p in sample_dir.iterdir() if p.is_dir()]) if sample_dir.exists() else []
    check(present == sorted(EXPECTED_CLASSES), "sample_cross_domain contains all 29 class folders", errors)
    for class_name in EXPECTED_CLASSES:
        class_dir = sample_dir / class_name
        if class_dir.exists():
            images = [p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
            check(len(images) >= 1, f"sample_cross_domain/{class_name} has at least one image", errors)


def main():
    errors = []
    check_required_paths(errors)
    check_labels(errors)
    check_configs(errors)
    check_samples(errors)

    if errors:
        print("\nRepository validation failed.")
        print("Please fix:")
        for err in errors:
            print(f"- {err}")
        sys.exit(1)

    print("\nRepository validation passed.")


if __name__ == "__main__":
    main()
