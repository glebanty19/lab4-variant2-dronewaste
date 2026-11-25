#!/usr/bin/env python3
"""Train and evaluate Faster R-CNN on local COCO RGB train/valid/test splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_DATA = Path("data/construction_waste/Instance Segmentation Dataset")
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dataset_info(root: Path) -> tuple[list[dict[str, Any]], dict[int, int], dict[int, str]]:
    train = load_json(root / "train_rgb" / "_annotations.coco.json")
    cats = [item for item in train["categories"] if item["id"] != 0]
    cat_id_to_label = {item["id"]: index + 1 for index, item in enumerate(cats)}
    label_to_name = {index + 1: item["name"] for index, item in enumerate(cats)}
    return cats, cat_id_to_label, label_to_name


def source_group(record: dict[str, Any]) -> str:
    return record["file_name"].split("_rgb_", 1)[0]


def load_split_coco(root: Path, split: str) -> tuple[dict[str, Any], int]:
    folder = SPLITS[split]
    data = load_json(root / f"{folder}_rgb" / "_annotations.coco.json")
    excluded = 0
    if split == "train":
        heldout_groups = set()
        for heldout_split in ("val", "test"):
            heldout_folder = SPLITS[heldout_split]
            heldout_data = load_json(
                root / f"{heldout_folder}_rgb" / "_annotations.coco.json"
            )
            heldout_groups.update(source_group(record) for record in heldout_data["images"])
        original_count = len(data["images"])
        data["images"] = [
            record
            for record in data["images"]
            if source_group(record) not in heldout_groups
        ]
        keep_ids = {record["id"] for record in data["images"]}
        data["annotations"] = [
            annotation
            for annotation in data["annotations"]
            if annotation["image_id"] in keep_ids
        ]
        excluded = original_count - len(data["images"])
    return data, excluded


def validate_dataset(root: Path) -> dict[str, Any]:
    categories, category_map, _ = dataset_info(root)
    summary: dict[str, Any] = {"categories": categories, "splits": {}}
    for split, folder in SPLITS.items():
        data, excluded = load_split_coco(root, split)
        images = {item["id"]: item for item in data["images"]}
        annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in data["annotations"]:
            if item["image_id"] not in images:
                raise ValueError(f"{split}: annotation references missing image id {item['image_id']}")
            if item["category_id"] not in category_map:
                raise ValueError(f"{split}: unexpected category id {item['category_id']}")
            x, y, width, height = item["bbox"]
            if width <= 0 or height <= 0:
                raise ValueError(f"{split}: invalid bbox in annotation {item['id']}")
            record = images[item["image_id"]]
            image_path = root / f"{folder}_rgb" / record["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Image listed in COCO file is missing: {image_path}")
            if x < -1 or y < -1 or x + width > record["width"] + 1 or y + height > record["height"] + 1:
                raise ValueError(f"{split}: bbox exceeds image bounds in annotation {item['id']}")
            annotations[item["image_id"]].append(item)
        summary["splits"][split] = {
            "images": len(images),
            "annotations": len(data["annotations"]),
            "excluded_scene_images": excluded,
            "source_groups": sorted({source_group(record) for record in images.values()}),
            "class_counts": {
                category["name"]: sum(
                    1 for item in data["annotations"] if item["category_id"] == category["id"]
                )
                for category in categories
            },
            "empty_images": sum(not annotations.get(image_id) for image_id in images),
        }
    return summary


def save_annotation_preview(root: Path, output: Path) -> None:
    from PIL import ImageDraw

    categories, category_map, label_names = dataset_info(root)
    colors = ("lime", "cyan", "yellow", "magenta", "orange", "deepskyblue")
    samples = []
    for split in ("train", "val", "test"):
        folder = SPLITS[split]
        data, _ = load_split_coco(root, split)
        by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in data["annotations"]:
            by_image[item["image_id"]].append(item)
        candidate = next(
            (image for image in data["images"] if by_image.get(image["id"])),
            data["images"][0],
        )
        image_path = root / f"{folder}_rgb" / candidate["file_name"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image.thumbnail((640, 360))
        scale_x = image.width / candidate["width"]
        scale_y = image.height / candidate["height"]
        draw = ImageDraw.Draw(image)
        for ann in by_image.get(candidate["id"], []):
            x, y, width, height = ann["bbox"]
            x0, y0 = x * scale_x, y * scale_y
            x1, y1 = (x + width) * scale_x, (y + height) * scale_y
            color = colors[category_map[ann["category_id"]] - 1]
            draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
            draw.text(
                (x0, max(0, y0 - 11)),
                label_names[category_map[ann["category_id"]]],
                fill=color,
            )
        samples.append((split, image))
    canvas = Image.new("RGB", (640, 3 * 385), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (split, image) in enumerate(samples):
        y = index * 385
        canvas.paste(image, (0, y + 22))
        draw.text((8, y + 4), f"{split}: COCO ground truth", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def load_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ImportError as exc:
        raise RuntimeError("Install compatible PyTorch and torchvision packages first.") from exc
    return torch, torchvision, FastRCNNPredictor


def make_model(
    torchvision: Any,
    predictor_class: Any,
    class_count: int,
    pretrained: bool,
    min_size: int = 480,
    max_size: int = 640,
):
    if pretrained:
        import torch
        torch.hub.set_dir(str(Path(__file__).resolve().parent / ".torch_hub"))
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None
        )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = predictor_class(in_features, class_count + 1)
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size
    model.roi_heads.detections_per_img = 150
    model.rpn._pre_nms_top_n["training"] = 1000
    model.rpn._post_nms_top_n["training"] = 500
    return model


class CocoDataset:
    def __init__(
        self,
        root: Path,
        split: str,
        training: bool = False,
        degrade_factor: int = 1,
        scale_aug: bool = False,
    ):
        _, self.category_map, _ = dataset_info(root)
        self.root = root
        self.split = split
        self.folder = SPLITS[split]
        self.training = training
        self.degrade_factor = degrade_factor
        self.scale_aug = scale_aug
        self.coco, _ = load_split_coco(root, split)
        by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in self.coco["annotations"]:
            by_image[item["image_id"]].append(item)
        self.by_image = dict(by_image)

    def __len__(self):
        return len(self.coco["images"])

    def __getitem__(self, index):
        import torch

        record = self.coco["images"][index]
        path = self.root / f"{self.folder}_rgb" / record["file_name"]
        with Image.open(path) as source:
            pil_image = source.convert("RGB")
        factor = self.degrade_factor
        if self.scale_aug:
            factor = random.choice((1, 2, 3, 4))
        if factor > 1:
            width, height = pil_image.size
            pil_image = pil_image.resize(
                (max(1, width // factor), max(1, height // factor)), Image.Resampling.BOX
            ).resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(pil_image, dtype=np.float32) / 255.0
        image = torch.from_numpy(array).permute(2, 0, 1)
        items = self.by_image.get(record["id"], [])
        boxes = torch.tensor(
            [
                [
                    a["bbox"][0],
                    a["bbox"][1],
                    a["bbox"][0] + a["bbox"][2],
                    a["bbox"][1] + a["bbox"][3],
                ]
                for a in items
            ],
            dtype=torch.float32,
        ).reshape(-1, 4)
        labels = torch.tensor(
            [self.category_map[a["category_id"]] for a in items],
            dtype=torch.int64,
        )
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([record["id"]], dtype=torch.int64),
            "area": torch.tensor([a["area"] for a in items], dtype=torch.float32),
            "iscrowd": torch.tensor(
                [a.get("iscrowd", 0) for a in items], dtype=torch.int64
            ),
        }
        if self.training and random.random() < 0.5:
            image = image.flip(-1)
            if len(boxes):
                old_x0 = target["boxes"][:, 0].clone()
                old_x1 = target["boxes"][:, 2].clone()
                target["boxes"][:, 0] = record["width"] - old_x1
                target["boxes"][:, 2] = record["width"] - old_x0
        return image, target


def make_dataset(root: Path, split: str, torch: Any, training: bool = False,
                 degrade_factor: int = 1, scale_aug: bool = False):
    folder = SPLITS[split]
    return CocoDataset(root, split, training=training, degrade_factor=degrade_factor,
                       scale_aug=scale_aug)


def collate(batch):
    return tuple(zip(*batch))


def iou(a: list[float], b: list[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    union += max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def average_precision(
    predictions: list[dict[str, Any]],
    truths: dict[str, list[dict[str, Any]]],
    label: int,
    iou_threshold: float,
) -> tuple[float, list[float], list[float], int]:
    gt = {
        image_id: [item for item in values if item["label"] == label]
        for image_id, values in truths.items()
    }
    support = sum(map(len, gt.values()))
    if support == 0:
        return math.nan, [], [], 0
    matched = {image_id: set() for image_id in gt}
    ranked = []
    for prediction in predictions:
        for box, class_id, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"]
        ):
            if class_id == label:
                ranked.append((score, prediction["image_id"], box))
    ranked.sort(key=lambda item: item[0], reverse=True)
    tp, fp, recalls, precisions = 0, 0, [], []
    for _, image_id, box in ranked:
        candidates = [
            (iou(box, truth["box"]), index)
            for index, truth in enumerate(gt.get(image_id, []))
            if index not in matched.get(image_id, set())
        ]
        overlap, index = max(candidates, default=(0.0, -1))
        if overlap >= iou_threshold:
            tp += 1
            matched[image_id].add(index)
        else:
            fp += 1
        recalls.append(tp / support)
        precisions.append(tp / (tp + fp))
    envelope = np.maximum.accumulate(np.asarray(precisions[::-1]))[::-1] if precisions else []
    ap = (
        sum(
            max(
                (
                    float(envelope[index])
                    for index, recall in enumerate(recalls)
                    if recall >= target
                ),
                default=0.0,
            )
            for target in np.linspace(0, 1, 101)
        )
        / 101.0
    )
    return float(ap), recalls, precisions, support


def evaluate_model(
    model: Any,
    root: Path,
    split: str,
    torch: Any,
    category_names: dict[int, str],
    *,
    degrade_x2: bool = False,
    degrade_factor: int = 1,
    tile_size: int | None = None,
    overlap: float = 0.2,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    folder = SPLITS[split]
    data = load_json(root / f"{folder}_rgb" / "_annotations.coco.json")
    annotation_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data["annotations"]:
        x, y, width, height = ann["bbox"]
        annotation_map[ann["image_id"]].append(
            {
                "box": [x, y, x + width, y + height],
                "label": next(
                    label for label, name in category_names.items()
                    if name == next(
                        c["name"] for c in data["categories"] if c["id"] == ann["category_id"]
                    )
                ),
            }
        )
    device = next(model.parameters()).device
    model.eval()
    all_predictions = []
    all_truths: dict[str, list[dict[str, Any]]] = {}
    from torchvision.ops import nms

    for record in data["images"]:
        image_id = str(record["id"])
        image_path = root / f"{folder}_rgb" / record["file_name"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        factor = 2 if degrade_x2 else degrade_factor
        if factor > 1:
            width, height = image.size
            image = image.resize((max(1, width // factor), max(1, height // factor)), Image.Resampling.BOX).resize(
                (width, height), Image.Resampling.BILINEAR
            )
        width, height = image.size
        tile_boxes = []
        if tile_size:
            step = max(1, int(tile_size * (1 - overlap)))
            xs = list(range(0, max(1, width - tile_size + 1), step))
            ys = list(range(0, max(1, height - tile_size + 1), step))
            if not xs or xs[-1] != max(0, width - tile_size):
                xs.append(max(0, width - tile_size))
            if not ys or ys[-1] != max(0, height - tile_size):
                ys.append(max(0, height - tile_size))
            tile_boxes = [
                (x, y, min(width, x + tile_size), min(height, y + tile_size))
                for y in ys for x in xs
            ]
        else:
            tile_boxes = [(0, 0, width, height)]
        boxes, labels, scores = [], [], []
        for x0, y0, x1, y1 in tile_boxes:
            tile = image.crop((x0, y0, x1, y1))
            tensor = torch.from_numpy(
                np.asarray(tile, dtype=np.float32).copy() / 255.0
            ).permute(2, 0, 1).to(device)
            with torch.no_grad():
                pred = model([tensor])[0]
            for box, label, score in zip(
                pred["boxes"].detach().cpu().tolist(),
                pred["labels"].detach().cpu().tolist(),
                pred["scores"].detach().cpu().tolist(),
            ):
                boxes.append([box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0])
                labels.append(int(label))
                scores.append(float(score))
        if boxes:
            box_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
            score_tensor = torch.tensor(scores, dtype=torch.float32, device=device)
            label_tensor = torch.tensor(labels, dtype=torch.int64, device=device)
            keep = []
            for class_id in sorted(set(labels)):
                indexes = torch.where(label_tensor == class_id)[0]
                keep.extend(indexes[nms(box_tensor[indexes], score_tensor[indexes], 0.5)].cpu().tolist())
            boxes, labels, scores = (
                [boxes[i] for i in keep],
                [labels[i] for i in keep],
                [scores[i] for i in keep],
            )
        all_predictions.append(
            {"image_id": image_id, "boxes": boxes, "labels": labels, "scores": scores}
        )
        all_truths[image_id] = annotation_map.get(record["id"], [])

    result: dict[str, Any] = {
        "split": split,
        "degrade_x2": degrade_x2,
        "degrade_factor": 2 if degrade_x2 else degrade_factor,
        "tile_size": tile_size,
        "class_names": category_names,
        "classes": {},
    }
    for class_id, class_name in category_names.items():
        ap_values = []
        recalls, precisions = [], []
        support = 0
        for threshold in np.arange(0.5, 0.951, 0.05):
            ap, _, _, support = average_precision(
                all_predictions, all_truths, class_id, float(threshold)
            )
            if math.isfinite(ap):
                ap_values.append(ap)
        ap50, recalls, precisions, support = average_precision(
            all_predictions, all_truths, class_id, 0.5
        )
        result["classes"][class_name] = {
            "id": class_id,
            "support": support,
            "ap50": ap50 if math.isfinite(ap50) else None,
            "ap50_95": float(np.mean(ap_values)) if ap_values else None,
            "recall_curve": recalls,
            "precision_curve": precisions,
        }
    result["map50"] = float(
        np.mean([item["ap50"] for item in result["classes"].values() if item["ap50"] is not None])
    )
    result["map50_95"] = float(
        np.mean([item["ap50_95"] for item in result["classes"].values() if item["ap50_95"] is not None])
    )
    return result, all_predictions, all_truths


def best_threshold(
    predictions: list[dict[str, Any]],
    truths: dict[str, list[dict[str, Any]]],
    labels: list[int],
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.0
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in np.arange(0.05, 0.951, 0.025):
        tp = fp = fn = 0
        for prediction in predictions:
            gt = truths[prediction["image_id"]]
            found: set[int] = set()
            candidate_predictions = sorted(
                [
                    (score, box, label)
                    for box, label, score in zip(
                        prediction["boxes"], prediction["labels"], prediction["scores"]
                    )
                    if score >= threshold
                ],
                reverse=True,
            )
            for _, box, label in candidate_predictions:
                options = [
                    (iou(box, item["box"]), i)
                    for i, item in enumerate(gt)
                    if item["label"] == label and i not in found
                ]
                overlap, index = max(options, default=(0.0, -1))
                if overlap >= 0.5:
                    tp += 1
                    found.add(index)
                else:
                    fp += 1
            fn += len(gt) - len(found)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"]:
            best_threshold = float(threshold)
            best = {"precision": precision, "recall": recall, "f1": f1}
    return best_threshold, best


def save_evaluation(
    output: Path,
    prefix: str,
    result: dict[str, Any],
    predictions: list[dict[str, Any]],
    truths: dict[str, list[dict[str, Any]]],
    threshold: float | None = None,
    image_root: Path | None = None,
    split: str = "test",
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    clean = json.loads(json.dumps(result))
    for item in clean["classes"].values():
        item.pop("recall_curve", None)
        item.pop("precision_curve", None)
    if threshold is not None:
        clean["threshold"] = threshold
        clean["operating_point"] = calculate_operating_point(
            predictions, truths, threshold, clean["class_names"]
        )
    (output / f"{prefix}_metrics.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    # Retain raw detections so area metrics and error audits can be reproduced
    # without another expensive GPU inference pass.
    import gzip
    with gzip.open(output / f"{prefix}_detections.json.gz", "wt", encoding="utf-8") as stream:
        json.dump({"predictions": predictions, "truths": truths}, stream)
    with (output / f"{prefix}_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class", "support", "AP@0.5", "AP@[0.5:0.95]"])
        for name, item in clean["classes"].items():
            writer.writerow([name, item["support"], item["ap50"], item["ap50_95"]])
        writer.writerow(["mAP", "", clean["map50"], clean["map50_95"]])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6))
    for name, item in result["classes"].items():
        if item["recall_curve"]:
            axis.plot(item["recall_curve"], item["precision_curve"], label=f"{name} AP={item['ap50']:.3f}")
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / f"{prefix}_pr.png", dpi=150)
    plt.close(figure)
    if threshold is not None:
        errors = {"miss": 0, "class": 0, "background": 0, "duplicate": 0}
        missed = []
        for prediction in predictions:
            matched: set[int] = set()
            truths_here = truths[prediction["image_id"]]
            ordered = sorted(
                [
                    (score, box, label)
                    for box, label, score in zip(
                        prediction["boxes"], prediction["labels"], prediction["scores"]
                    )
                    if score >= threshold
                ],
                reverse=True,
            )
            for _, box, label in ordered:
                overlaps = [(iou(box, truth["box"]), i) for i, truth in enumerate(truths_here)]
                same = [(ov, i) for ov, i in overlaps if truths_here[i]["label"] == label]
                best_same = max(same, default=(0.0, -1))
                if best_same[0] >= 0.5 and best_same[1] not in matched:
                    matched.add(best_same[1])
                elif best_same[0] >= 0.5:
                    errors["duplicate"] += 1
                elif max(overlaps, default=(0.0, -1))[0] >= 0.5:
                    errors["class"] += 1
                else:
                    errors["background"] += 1
            errors["miss"] += len(truths_here) - len(matched)
        (output / f"{prefix}_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar(errors.keys(), errors.values())
        axis.set(ylabel="Count", title="Detection error categories")
        figure.tight_layout()
        figure.savefig(output / f"{prefix}_errors.png", dpi=150)
        plt.close(figure)
        if image_root is not None:
            save_examples(
                output / f"{prefix}_examples.png",
                predictions,
                truths,
                threshold,
                image_root,
                split,
                clean["class_names"],
            )


def calculate_operating_point(
    predictions: list[dict[str, Any]],
    truths: dict[str, list[dict[str, Any]]],
    threshold: float,
    class_names: dict[int, str],
) -> dict[str, Any]:
    tp = fp = fn = 0
    for prediction in predictions:
        matched: set[int] = set()
        ordered = sorted(
            [
                (score, box, label)
                for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"])
                if score >= threshold
            ],
            reverse=True,
        )
        for _, box, label in ordered:
            choices = [
                (iou(box, truth["box"]), index)
                for index, truth in enumerate(truths[prediction["image_id"]])
                if truth["label"] == label and index not in matched
            ]
            overlap, index = max(choices, default=(0.0, -1))
            if overlap >= 0.5:
                tp += 1
                matched.add(index)
            else:
                fp += 1
        fn += len(truths[prediction["image_id"]]) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class_names": list(class_names.values()),
    }


def save_examples(
    path: Path,
    predictions,
    truths,
    threshold: float,
    image_root: Path,
    split: str,
    class_names: dict[str, str],
    limit: int = 8,
):
    if not predictions:
        return
    columns = 2
    rows = math.ceil(min(limit, len(predictions)) / columns)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    coco = load_json(image_root / f"{SPLITS[split]}_rgb" / "_annotations.coco.json")
    file_by_id = {str(item["id"]): item["file_name"] for item in coco["images"]}
    name_by_id = {int(key): value for key, value in class_names.items()}
    figure, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows), squeeze=False)
    for axis, prediction in zip(axes.flat, predictions[:limit]):
        image_id = prediction["image_id"]
        with Image.open(image_root / f"{SPLITS[split]}_rgb" / file_by_id[image_id]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(f"Image ID {image_id}")
        for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
            if score >= threshold:
                axis.add_patch(
                    Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        fill=False,
                        edgecolor="red",
                        linewidth=1.5,
                    )
                )
                axis.text(
                    box[0],
                    max(0, box[1] - 3),
                    f"{name_by_id.get(label, label)} {score:.2f}",
                    fontsize=7,
                    color="white",
                    bbox={"facecolor": "red", "alpha": 0.75, "pad": 1},
                )
        for truth in truths[image_id]:
            box = truth["box"]
            axis.add_patch(
                Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    fill=False,
                    edgecolor="lime",
                    linewidth=1,
                )
            )
        axis.axis("off")
    for axis in axes.flat[len(predictions[:limit]):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def train(args: argparse.Namespace) -> None:
    torch, torchvision, predictor_class = load_runtime()
    root, output = args.data.resolve(), args.output.resolve()
    summary = validate_dataset(root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_annotation_preview(root, output / "annotations_preview.png")
    categories, category_map, category_names_raw = dataset_info(root)
    category_names = {label: name for label, name in category_names_raw.items()}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}; classes: {category_names}")
    train_set = make_dataset(root, "train", torch, training=True,
                             degrade_factor=args.degrade_train,
                             scale_aug=args.scale_aug)
    validation_set = make_dataset(root, "val", torch)
    loader_options = {
        "batch_size": 1,
        "shuffle": args.positive_weight == 1,
        "num_workers": args.workers,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
    }
    if args.positive_weight <= 0:
        raise ValueError("--positive-weight must be positive")
    if args.positive_weight != 1:
        weights = [args.positive_weight if train_set.by_image.get(record["id"]) else 1.0
                   for record in train_set.coco["images"]]
        loader_options["sampler"] = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(train_set), replacement=True
        )
    if args.workers > 0:
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = 2
    loader = torch.utils.data.DataLoader(train_set, **loader_options)
    resumed = None
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
    model = make_model(
        torchvision,
        predictor_class,
        len(categories),
        not args.no_pretrained and resumed is None,
        args.min_size,
        args.max_size,
    )
    if resumed is not None:
        model.load_state_dict(resumed["model"])
        print(f"Warm-started from epoch {resumed.get('epoch')} checkpoint: {args.resume}")
    model = model.to(device)
    if args.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.roi_heads.box_predictor.parameters():
            param.requires_grad = True
    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, args.epochs // 3), gamma=0.1
    )
    history = []
    best_score, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_index, (images, targets) in enumerate(loader, 1):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items()}
                for target in targets
            ]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}: {loss}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if batch_index % 100 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "batch": batch_index,
                            "batches": len(loader),
                            "loss": float(np.mean(losses[-100:])),
                        }
                    ),
                    flush=True,
                )
        scheduler.step()
        val_metrics, _, _ = evaluate_model(
            model, root, "val", torch, category_names,
            degrade_factor=args.degrade_train,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_map50": val_metrics["map50"],
            "val_map50_95": val_metrics["map50_95"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "classes": category_names,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "degrade_train": args.degrade_train,
            "scale_aug": args.scale_aug,
            "seed": args.seed,
            "positive_weight": args.positive_weight,
            "validation": val_metrics,
        }
        torch.save(checkpoint, output / "last.pt")
        if row["val_map50_95"] > best_score:
            best_score, best_epoch = row["val_map50_95"], epoch
            torch.save(checkpoint, output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break
    plot_training(history, output / "training_curves.png")


def plot_training(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [item["epoch"] for item in history]
    axes[0].plot(epochs, [item["train_loss"] for item in history], marker="o")
    axes[0].set(xlabel="Epoch", ylabel="Train loss", title="Training loss")
    axes[1].plot(epochs, [item["val_map50_95"] for item in history], marker="o")
    axes[1].set(xlabel="Epoch", ylabel="Validation mAP@[.5:.95]", title="Validation mAP")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_command(args: argparse.Namespace) -> None:
    torch, torchvision, predictor_class = load_runtime()
    root = args.data.resolve()
    categories, _, raw_names = dataset_info(root)
    names = {label: name for label, name in raw_names.items()}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = make_model(
        torchvision,
        predictor_class,
        len(categories),
        pretrained=False,
        min_size=checkpoint.get("min_size", 480),
        max_size=checkpoint.get("max_size", 640),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    result, predictions, truths = evaluate_model(
        model,
        root,
        args.split,
        torch,
        names,
        degrade_x2=args.degrade_x2,
        degrade_factor=args.degrade_factor,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )
    threshold = args.threshold
    if args.split == "val" and threshold is None:
        threshold, point = best_threshold(predictions, truths, list(names))
        result["best_f1_threshold"] = {"confidence": threshold, **point}
        print(f"Validation threshold={threshold:.3f}; {point}", flush=True)
    save_evaluation(
        args.output.resolve(),
        args.prefix,
        result,
        predictions,
        truths,
        threshold,
        image_root=root,
        split=args.split,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "classes"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--patience", type=int, default=3)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--lr", type=float, default=0.005)
    train_parser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument("--positive-weight", type=float, default=1.0)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument("--min-size", type=int, default=480)
    train_parser.add_argument("--max-size", type=int, default=640)
    train_parser.add_argument("--resume", type=Path, default=None)
    train_parser.add_argument("--no-pretrained", action="store_true")
    train_parser.add_argument("--freeze-backbone", action="store_true")
    train_parser.add_argument("--degrade-train", type=int, choices=(1, 2, 4), default=1)
    train_parser.add_argument("--scale-aug", action="store_true")
    train_parser.set_defaults(func=train)
    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    eval_parser.add_argument("--prefix", default="evaluation")
    eval_parser.add_argument("--threshold", type=float, default=None)
    eval_parser.add_argument("--degrade-x2", action="store_true")
    eval_parser.add_argument("--degrade-factor", type=int, choices=(1, 2, 4), default=1)
    eval_parser.add_argument("--tile-size", type=int, default=None)
    eval_parser.add_argument("--overlap", type=float, default=0.2)
    eval_parser.add_argument("--device", default=None)
    eval_parser.set_defaults(func=evaluate_command)
    args = parser.parse_args()
    try:
        args.func(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
