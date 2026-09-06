
import os
import sys
import json
import time
from pathlib import Path

MODEL_NAME = sys.argv[1]
MODEL_PATH = Path(sys.argv[2])
MANIFEST_CSV = Path(sys.argv[3])
CLASS_MAPPING_CSV = Path(sys.argv[4])
OUT_DIR = Path(sys.argv[5])
IMG_SIZE = int(sys.argv[6])
BATCH_SIZE = int(sys.argv[7])

import random
import numpy as np
import pandas as pd
from PIL import Image

random.seed(42)
np.random.seed(42)

manifest = pd.read_csv(MANIFEST_CSV)
class_map = pd.read_csv(CLASS_MAPPING_CSV)

CLASS_LABELS = (
    class_map
    .sort_values("class_index")["class_name"]
    .astype(str)
    .tolist()
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

partial_csv = OUT_DIR / "predictions.partial.csv"
final_csv = OUT_DIR / "predictions.csv"
meta_json = OUT_DIR / "worker_metadata.json"

def canonical_label(name):
    s = str(name).strip()

    if len(s) == 1:
        return s.upper()

    low = s.lower()

    # "del" and "delete" are the same semantic class.
    if low == "del":
        return "delete"

    if low in {"delete", "nothing", "space"}:
        return low

    return s

# ------------------------------------------------------------
# Resume
# ------------------------------------------------------------
rows = []
done = set()

if partial_csv.exists():
    old = pd.read_csv(partial_csv)

    required = {
        "relative_path",
        "true_label",
        "pred_label",
        "correct",
    }

    if required.issubset(old.columns):
        rows = old.to_dict("records")
        done = set(old["relative_path"].astype(str))
        print("Resume images:", len(done))
    else:
        print("Ignoring incompatible partial CSV.")

remaining = manifest[
    ~manifest["relative_path"].astype(str).isin(done)
].copy()

# ------------------------------------------------------------
# Model load
# ------------------------------------------------------------
framework = None
model = None
load_t0 = time.perf_counter()

if MODEL_NAME in {"Custom_CNN", "EfficientNetB0", "ResNet50V2"}:
    import tensorflow as tf
    from tensorflow import keras

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    class Cast(tf.keras.layers.Layer):
        def __init__(self, dtype="float32", **kwargs):
            super().__init__(**kwargs)
            self._dtype = dtype

        def call(self, inputs):
            return tf.cast(inputs, self._dtype)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"dtype": self._dtype})
            return cfg

    class EffNetPreprocessCompat(tf.keras.layers.Layer):
        """
        Keras-3-safe equivalent of historical:
            lambda z: preprocess_input(z * 255.0)
        No trainable weights.
        """
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.supports_masking = True

        def call(self, inputs, training=None, mask=None, **kwargs):
            x = inputs * tf.cast(255.0, inputs.dtype)
            return tf.keras.applications.efficientnet.preprocess_input(x)

        def compute_mask(self, inputs, mask=None):
            return mask

        def get_config(self):
            return super().get_config()

    custom_objects = {
        "Cast": Cast,
    }

    if MODEL_NAME == "EfficientNetB0":
        custom_objects["EffNetPreprocessCompat"] = EffNetPreprocessCompat

    model = keras.models.load_model(
        str(MODEL_PATH),
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )

    print("Loaded:", MODEL_NAME)
    print("Input :", model.input_shape)
    print("Output:", model.output_shape)

    if model.output_shape[-1] != 29:
        raise RuntimeError(
            f"{MODEL_NAME}: expected 29 outputs, got {model.output_shape}"
        )

    if MODEL_NAME == "EfficientNetB0":
        dummy = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
        smoke = model(dummy, training=False).numpy()

        if smoke.shape != (1, 29):
            raise RuntimeError(
                f"EfficientNet smoke output shape: {smoke.shape}"
            )

        if not np.isfinite(smoke).all():
            raise RuntimeError(
                "EfficientNet smoke output contains NaN/Inf."
            )

        print("EfficientNet dummy forward: PASS")

    framework = "keras"

elif MODEL_NAME == "ViT":
    import torch
    import timm
    from torchvision import transforms

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=29,
    )

    state_dict = torch.load(
        str(MODEL_PATH),
        map_location="cpu",
    )

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {
            str(k).replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    framework = "vit"

elif MODEL_NAME == "YOLOv11m_cls":
    import torch
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))

    if len(model.names) != 29:
        raise RuntimeError(
            f"YOLOv11m-cls expected 29 classes, got {len(model.names)}"
        )

    test_names = {
        x.lower(): i
        for i, x in enumerate(CLASS_LABELS)
    }

    for _, name in model.names.items():
        label = canonical_label(name)

        if label.lower() not in test_names:
            raise RuntimeError(
                f"YOLO class '{name}' not present in dataset mapping."
            )

    framework = "yolo_cls"

else:
    raise ValueError(MODEL_NAME)

load_seconds = time.perf_counter() - load_t0

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_keras_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.float32) / 255.0

def checkpoint(batch_rows):
    global rows
    rows.extend(batch_rows)
    pd.DataFrame(rows).to_csv(partial_csv, index=False)

records = remaining.to_dict("records")

# ------------------------------------------------------------
# Keras inference
# ------------------------------------------------------------
if framework == "keras":
    import tensorflow as tf

    input_name = model.inputs[0].name.split(":")[0]

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start+BATCH_SIZE]

        x = np.stack([
            load_keras_image(r["absolute_path"])
            for r in batch
        ])

        t0 = time.perf_counter()

        if MODEL_NAME == "EfficientNetB0":
            probs = model(
                {input_name: tf.convert_to_tensor(x)},
                training=False,
            ).numpy()
        else:
            probs = model(
                tf.convert_to_tensor(x),
                training=False,
            ).numpy()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        pred_idx = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)

        batch_rows = []

        for r, pi, cf in zip(batch, pred_idx, conf):
            pi = int(pi)
            pred_label = CLASS_LABELS[pi]

            batch_rows.append({
                "relative_path": r["relative_path"],
                "absolute_path": r["absolute_path"],
                "true_label": r["label"],
                "true_index": int(r["true_index"]),
                "pred_index": pi,
                "pred_label": pred_label,
                "confidence": float(cf),
                "correct": pred_label == r["label"],
                "batch_elapsed_ms": float(elapsed_ms),
                "batch_mean_inference_ms_per_image": float(
                    elapsed_ms / max(1, len(batch))
                ),
            })

        checkpoint(batch_rows)

        print(
            MODEL_NAME,
            min(start + len(batch), len(records)),
            "/",
            len(records),
        )

# ------------------------------------------------------------
# ViT inference
# ------------------------------------------------------------
elif framework == "vit":
    import torch

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start+BATCH_SIZE]

        tensors = []

        for r in batch:
            img = Image.open(r["absolute_path"]).convert("RGB")
            tensors.append(transform(img))

        x = torch.stack(tensors, dim=0).to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        with torch.inference_mode():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        conf, pred_idx = probs.max(dim=1)

        conf = conf.detach().cpu().numpy()
        pred_idx = pred_idx.detach().cpu().numpy()

        batch_rows = []

        for r, pi, cf in zip(batch, pred_idx, conf):
            pi = int(pi)
            pred_label = CLASS_LABELS[pi]

            batch_rows.append({
                "relative_path": r["relative_path"],
                "absolute_path": r["absolute_path"],
                "true_label": r["label"],
                "true_index": int(r["true_index"]),
                "pred_index": pi,
                "pred_label": pred_label,
                "confidence": float(cf),
                "correct": pred_label == r["label"],
                "batch_elapsed_ms": float(elapsed_ms),
                "batch_mean_inference_ms_per_image": float(
                    elapsed_ms / max(1, len(batch))
                ),
            })

        checkpoint(batch_rows)

        print(
            MODEL_NAME,
            min(start + len(batch), len(records)),
            "/",
            len(records),
        )

# ------------------------------------------------------------
# YOLO classification inference
# ------------------------------------------------------------
else:
    import torch

    class_to_idx = {
        x.lower(): i
        for i, x in enumerate(CLASS_LABELS)
    }

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start+BATCH_SIZE]
        paths = [r["absolute_path"] for r in batch]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        results = model.predict(
            source=paths,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            verbose=False,
            device=0,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if len(results) != len(batch):
            raise RuntimeError(
                f"YOLO result count mismatch: {len(results)} vs {len(batch)}"
            )

        batch_rows = []

        for r, result in zip(batch, results):
            if result.probs is None:
                raise RuntimeError(
                    "YOLO classification result has no probs."
                )

            yolo_idx = int(result.probs.top1)
            confidence = float(result.probs.top1conf.item())

            pred_label = canonical_label(
                model.names[yolo_idx]
            )

            mapped_idx = class_to_idx.get(pred_label.lower())

            if mapped_idx is None:
                raise RuntimeError(
                    f"Cannot map YOLO class: {pred_label}"
                )

            pred_label = CLASS_LABELS[mapped_idx]

            batch_rows.append({
                "relative_path": r["relative_path"],
                "absolute_path": r["absolute_path"],
                "true_label": r["label"],
                "true_index": int(r["true_index"]),
                "pred_index": int(mapped_idx),
                "pred_label": pred_label,
                "confidence": confidence,
                "correct": pred_label == r["label"],
                "batch_elapsed_ms": float(elapsed_ms),
                "batch_mean_inference_ms_per_image": float(
                    elapsed_ms / max(1, len(batch))
                ),
            })

        checkpoint(batch_rows)

        print(
            MODEL_NAME,
            min(start + len(batch), len(records)),
            "/",
            len(records),
        )

# ------------------------------------------------------------
# Finalize
# ------------------------------------------------------------
df = pd.DataFrame(rows)

if len(df) != len(manifest):
    raise RuntimeError(
        f"Expected {len(manifest)} predictions, got {len(df)}"
    )

if df["relative_path"].nunique() != len(manifest):
    raise RuntimeError(
        "Duplicate or missing prediction rows."
    )

df.to_csv(final_csv, index=False)

if partial_csv.exists():
    partial_csv.unlink()

metadata = {
    "model": MODEL_NAME,
    "framework": framework,
    "load_seconds": load_seconds,
    "n_predictions": int(len(df)),
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "effnet_runtime_compatibility": (
        "EffNetPreprocessCompat replacing only serialized no-weight "
        "effnet_preprocess Lambda; same preprocess_input(x*255.0); weights unchanged."
        if MODEL_NAME == "EfficientNetB0"
        else None
    ),
}

meta_json.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("__PASS__")
print(json.dumps(metadata))
