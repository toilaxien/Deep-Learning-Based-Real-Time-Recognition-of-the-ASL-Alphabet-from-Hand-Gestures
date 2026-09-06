import csv
import hashlib
import os
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space",
]

REQUIRED_DIRS = [
    "configs",
    "dataset_docs",
    "evaluation",
    "evaluation/technical_reexamination",
    "inference",
    "result",
    "sample_cross_domain",
    "tools",
    "training",
    "weights",
]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "configs/README.md",
    "dataset_docs/README.md",
    "dataset_docs/class_distribution.csv",
    "dataset_docs/cross_domain_metadata_template.csv",
    "dataset_docs/annotation_guidelines.md",
    "dataset_docs/inter_annotator_agreement.md",
    "evaluation/README.md",
    "evaluation/video_manifest_template.csv",
    "inference/README.md",
    "inference/webcam_app.py",
    "inference/label.txt",
    "result/README.md",
]

TARGET_CONFIGS = [
    "configs/custom_cnn.yaml",
    "configs/efficientnetb0.yaml",
    "configs/resnet50v2.yaml",
    "configs/vit.yaml",
    "configs/yolo11m_cls.yaml",
    "configs/yolo11m_detection.yaml",
    "configs/yolo12x.yaml",
    "configs/rtdetr_l.yaml",
]

SUPPORTING_CONFIGS = [
    "configs/yolov8s_stage1_hand_detector.yaml",
]

TARGET_WEIGHTS = [
    "weights/custom_cnn_asl.keras",
    "weights/efficientnetb0_asl.keras",
    "weights/resnet50v2_asl.keras",
    "weights/ViT.pth",
    "weights/YoLo11m_csf.pt",
    "weights/YoLo11m_dectection.pt",
    "weights/YoLo12x.pt",
    "weights/rtdetr-l.pt",
]

SUPPORTING_WEIGHTS = [
    "weights/yolov8s_stage1_hand_detector.pt",
]

TECHNICAL_REEXAMINATION_NOTEBOOKS = [
    "evaluation/technical_reexamination/01_dataset_audit/ijies-asl-benchmark-data-split-verification.ipynb",
    "evaluation/technical_reexamination/02_classification/IJIES_Technical_Reexamination_5Model_Classification_Benchmark_CrossDomain_Evaluation.ipynb",
    "evaluation/technical_reexamination/03_detection/ijies-benchmark-crossdomain-sequential-reevaluation.ipynb",
    "evaluation/technical_reexamination/04_realtime/IJIES_Technical_Reexamination_RealTime_Video_Evaluation.ipynb",
]

CANONICAL_EVIDENCE_DIRS = [
    "result/01_Benchmark_CrossDomain/Classification/IJIES_EDITOR_CLASSIFICATION_BENCHMARK_CROSSDOMAIN",
    "result/01_Benchmark_CrossDomain/Detection/IJIES_EDITOR_DETECTOR_BENCHMARK_CROSSDOMAIN",
    "result/02_RealTime_Video/IJIES_EDITOR_REALTIME_EVIDENCE",
    "result/03_Audits_and_Checksums/manifests",
    "result/03_Audits_and_Checksums/sha256",
    "result/03_Audits_and_Checksums/duplicate_checks",
    "result/03_Audits_and_Checksums/overlap_checks",
]

REQUIRED_EVIDENCE_FILES = [
    "result/01_Benchmark_CrossDomain/Classification/IJIES_EDITOR_CLASSIFICATION_BENCHMARK_CROSSDOMAIN/datasets/benchmark/manifest.csv",
    "result/01_Benchmark_CrossDomain/Classification/IJIES_EDITOR_CLASSIFICATION_BENCHMARK_CROSSDOMAIN/datasets/cross_domain/manifest.csv",
    "result/01_Benchmark_CrossDomain/Classification/IJIES_EDITOR_CLASSIFICATION_BENCHMARK_CROSSDOMAIN/datasets/benchmark_crossdomain_exact_overlap.csv",
    "result/01_Benchmark_CrossDomain/Detection/IJIES_EDITOR_DETECTOR_BENCHMARK_CROSSDOMAIN/02_dataset_audit/benchmark/manifest.csv",
    "result/01_Benchmark_CrossDomain/Detection/IJIES_EDITOR_DETECTOR_BENCHMARK_CROSSDOMAIN/02_dataset_audit/cross_domain/manifest.csv",
    "result/01_Benchmark_CrossDomain/Detection/IJIES_EDITOR_DETECTOR_BENCHMARK_CROSSDOMAIN/02_dataset_audit/benchmark_crossdomain_exact_overlap.csv",
    "result/02_RealTime_Video/IJIES_EDITOR_REALTIME_EVIDENCE/01_Summaries/CANONICAL_Table5_values.csv",
    "result/02_RealTime_Video/IJIES_EDITOR_REALTIME_EVIDENCE/02_Stage1_YOLOv8/stage1_summary.json",
    "result/03_Audits_and_Checksums/sha256/canonical_weight_sha256.csv",
    "result/03_Audits_and_Checksums/sha256/PACKAGE_CHECKSUMS_SHA256.csv",
    "result/03_Audits_and_Checksums/overlap_checks/SUBMISSION_STATUS_REVIEW_REQUIRED.md",
]

OBSOLETE_PATHS = [
    "evaluation/eval_classification.py",
    "evaluation/eval_detection.py",
    "evaluation/eval_realtime_video.ipynb",
    "inference/model",
    "result/03_Reproducibility_Code",
    "result/04_Audits_and_Checksums",
    "result/01_Benchmark_CrossDomain/Detection/IJIES_EDITOR_DETECTOR_BENCHMARK_CROSSDOMAIN/bbox_validation",
    "result/01_Benchmark_CrossDomain/Detection/IJIES_EDITOR_DETECTOR_BENCHMARK_CROSSDOMAIN/code",
    "result/01_Benchmark_CrossDomain/Detection/IJIES_EDITOR_DETECTOR_BENCHMARK_CROSSDOMAIN/datasets",
    "result/01_Benchmark_CrossDomain/Detection/IJIES_EDITOR_DETECTOR_BENCHMARK_CROSSDOMAIN/imagelevel_detection",
    "result/01_Benchmark_CrossDomain/Detection/IJIES_EDITOR_DETECTOR_BENCHMARK_CROSSDOMAIN/training_faithful_gt",
    "result/01_Benchmark_CrossDomain/Detection/IJIES_EDITOR_DETECTOR_BENCHMARK_CROSSDOMAIN/detection_imagelevel_NEW_crossdomain_summary.csv",
    "result/01_Benchmark_CrossDomain/Detection/IJIES_EDITOR_DETECTOR_BENCHMARK_CROSSDOMAIN/detection_final_crossdomain_summary.csv",
    "result/01_Benchmark_CrossDomain/Detection/IJIES_EDITOR_DETECTOR_BENCHMARK_CROSSDOMAIN/model_inventory_detector.csv",
    "result/01_Benchmark_CrossDomain/Detection/IJIES_EDITOR_DETECTOR_BENCHMARK_CROSSDOMAIN/environment_and_protocol.json",
    "result/01_Benchmark_CrossDomain/Detection/IJIES_EDITOR_DETECTOR_BENCHMARK_CROSSDOMAIN/Table7_detection_BENCHMARK_REUSED_NEW_CROSS_degradation.csv",
]


def ok(condition, message, errors):
    if condition:
        print(f"[OK] {message}")
    else:
        print(f"[FAIL] {message}")
        errors.append(message)


def rel(path):
    return path.as_posix()


def long_path(path):
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def exists(path):
    return path.exists() or os.path.exists(long_path(path))


def is_file(path):
    return path.is_file() or os.path.isfile(long_path(path))


def is_dir(path):
    return path.is_dir() or os.path.isdir(long_path(path))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def check_required_paths(errors):
    for item in REQUIRED_DIRS:
        ok(is_dir(ROOT / item), item, errors)
    for item in REQUIRED_FILES + TARGET_CONFIGS + SUPPORTING_CONFIGS:
        ok(is_file(ROOT / item), item, errors)
    for item in TARGET_WEIGHTS + SUPPORTING_WEIGHTS + TECHNICAL_REEXAMINATION_NOTEBOOKS:
        ok(is_file(ROOT / item), item, errors)
    for item in CANONICAL_EVIDENCE_DIRS:
        ok(is_dir(ROOT / item), item, errors)
    for item in REQUIRED_EVIDENCE_FILES:
        ok(is_file(ROOT / item), item, errors)


def check_no_obsolete_paths(errors):
    for item in OBSOLETE_PATHS:
        ok(not exists(ROOT / item), f"obsolete path absent: {item}", errors)


def check_labels(errors):
    label_path = ROOT / "inference" / "label.txt"
    labels = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok(labels == EXPECTED_CLASSES, "inference/label.txt has expected 29 labels in order", errors)


def check_configs(errors):
    yaml_files = sorted((ROOT / "configs").glob("*.yaml"))
    ok(len(yaml_files) == 9, "9 YAML config files: 8 target + 1 supporting Stage-1", errors)
    for path in yaml_files:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            base_ok = isinstance(data, dict) and "name" in data and "task" in data and "weights" in data
            ok(base_ok, f"{rel(path.relative_to(ROOT))} parseable with name/task/weights", errors)
        except Exception as exc:
            print(f"[ERROR] {rel(path.relative_to(ROOT))}: {exc}")
            errors.append(rel(path.relative_to(ROOT)))

    stage1 = yaml.safe_load((ROOT / SUPPORTING_CONFIGS[0]).read_text(encoding="utf-8"))
    ok(stage1.get("benchmark_model") is False, "YOLOv8s config marks benchmark_model=false", errors)
    ok(stage1.get("confidence_threshold") == 0.25, "YOLOv8s confidence threshold is 0.25", errors)
    ok(stage1.get("input_size") == [640, 640], "YOLOv8s input size is 640", errors)


def check_samples(errors):
    sample_dir = ROOT / "sample_cross_domain"
    present = sorted([p.name for p in sample_dir.iterdir() if p.is_dir()]) if sample_dir.exists() else []
    ok(present == sorted(EXPECTED_CLASSES), "sample_cross_domain contains all 29 class folders", errors)
    for class_name in EXPECTED_CLASSES:
        class_dir = sample_dir / class_name
        if class_dir.exists():
            images = [p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
            ok(len(images) >= 1, f"sample_cross_domain/{class_name} has at least one image", errors)


def check_class_distribution(errors):
    csv_path = ROOT / "dataset_docs" / "class_distribution.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    class_rows = [row for row in rows if row["class"] != "TOTAL"]
    total_rows = [row for row in rows if row["class"] == "TOTAL"]
    ok(len(class_rows) == 29, "class_distribution.csv has 29 class rows", errors)
    ok(all(int(row["private_cross_domain_images"]) == 30 for row in class_rows), "each class has 30 private cross-domain images", errors)
    ok(total_rows and int(total_rows[0]["private_cross_domain_images"]) == 870, "class_distribution.csv total is 870", errors)


def check_checkpoint_inventory(errors):
    weight_paths = [ROOT / item for item in TARGET_WEIGHTS + SUPPORTING_WEIGHTS]
    hashes = {}
    for path in weight_paths:
        digest = sha256(path)
        hashes.setdefault(digest, []).append(rel(path.relative_to(ROOT)))

    duplicates = {digest: paths for digest, paths in hashes.items() if len(paths) > 1}
    ok(not duplicates, f"no duplicate SHA256 among canonical weights: {duplicates}", errors)

    inventory = ROOT / "result" / "03_Audits_and_Checksums" / "sha256" / "canonical_weight_sha256.csv"
    with inventory.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    inventory_files = {row["file"] for row in rows}
    for item in TARGET_WEIGHTS + SUPPORTING_WEIGHTS:
        ok(item in inventory_files, f"SHA256 inventory includes {item}", errors)


def main():
    errors = []
    check_required_paths(errors)
    check_no_obsolete_paths(errors)
    check_labels(errors)
    check_configs(errors)
    check_samples(errors)
    check_class_distribution(errors)
    check_checkpoint_inventory(errors)

    if errors:
        print("\nRepository validation failed.")
        print("Please fix:")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)

    print("\nRepository validation passed.")


if __name__ == "__main__":
    main()
