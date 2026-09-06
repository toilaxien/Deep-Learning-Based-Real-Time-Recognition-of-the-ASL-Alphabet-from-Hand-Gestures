# IJIES EDITOR — DETECTOR BENCHMARK + CROSS-DOMAIN

This folder contains DETECTOR evidence only.

Models:
1. YOLOv11m
2. YOLOv12x
3. RT-DETR-L

Structure:
- `01_final_summary/`
- `02_dataset_audit/benchmark/`
- `02_dataset_audit/cross_domain/`
- `03_training_faithful_gt/benchmark/`
- `03_training_faithful_gt/cross_domain/`
- `04_imagelevel_predictions/benchmark/`
- `04_imagelevel_predictions/cross_domain/`
- `05_bbox_validation_outputs/benchmark/`
- `05_bbox_validation_outputs/cross_domain/`

Benchmark:
- copied from the previously completed ORIGINAL detector evidence;
- benchmark is NOT rerun in this notebook.

Cross-domain:
- freshly evaluated from `asl-crossdomain-dataset/ASL_CROSSDOMAIN_DATASET_TEST`;
- 870 images;
- 29 classes x 30 images/class;
- old cross-domain outputs are NOT included.

Training-faithful detector GT:
- each image has one full-image target:
  `class_id 0.5 0.5 1.0 1.0`
- `nothing` = class 27
- `delete` -> `del` = class 26

No ZIP is created by this notebook.
