#!/usr/bin/env python3
"""Create a site-disjoint two-class COCO benchmark for LR4 variant 2.

Positive objects are DroneWaste categories Rubble and Construction and
demolition materials. The output uses hard links to avoid copying image bytes.
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

SOURCE=Path(__file__).resolve().parents[1]/"datasets/dronewaste"
OUTPUT=Path(__file__).resolve().parent/"data/dronewaste_piles"
POSITIVE_IDS={1,2}
VALIDATION_SITE="site5"
TEST_SITE="site17"
SEED=42


def main():
    source=json.loads((SOURCE/"dronewaste_v1.0.json").read_text(encoding="utf8"))
    if not (SOURCE/"images").is_dir():
        raise FileNotFoundError("Extract official images.tar.gz first; see download_dronewaste_parallel.py")
    randomizer=random.Random(SEED)
    by_image=defaultdict(list)
    for ann in source["annotations"]:
        if ann["category_id"] in POSITIVE_IDS:
            by_image[ann["image_id"]].append(ann)
    groups={"train":[],"val":[],"test":[]}
    for im in source["images"]:
        split="val" if im["site"]==VALIDATION_SITE else "test" if im["site"]==TEST_SITE else "train"
        groups[split].append(im)
    positives=[im for im in groups["train"] if by_image[im["id"]]]
    negatives=[im for im in groups["train"] if not by_image[im["id"]]]
    groups["train"]=sorted(positives+randomizer.sample(negatives,min(len(negatives),2*len(positives))),key=lambda im:im["id"])
    summary={"source":"https://zenodo.org/records/17045559","seed":SEED,"positive_category_ids":sorted(POSITIVE_IDS),
             "positive_category_names":[c["name"] for c in source["categories"] if c["id"] in POSITIVE_IDS],
             "validation_site":VALIDATION_SITE,"test_site":TEST_SITE,"train_negative_sampling":"2 negatives per positive image",
             "splits":{}}
    for split,folder in (("train","train_rgb"),("val","valid_rgb"),("test","test_rgb")):
        dest=OUTPUT/folder;dest.mkdir(parents=True,exist_ok=True)
        images=groups[split];ids={im["id"] for im in images}
        annotations=[]
        for ann in source["annotations"]:
            if ann["image_id"] in ids and ann["category_id"] in POSITIVE_IDS:
                new=dict(ann)
                annotations.append(new)
        coco={"info":source.get("info",{}),
              "categories":[{"id":0,"name":"background"},{"id":1,"name":"Rubble"},
                            {"id":2,"name":"Construction and demolition materials"}],
              "images":images,"annotations":annotations}
        (dest/"_annotations.coco.json").write_text(json.dumps(coco,ensure_ascii=False),encoding="utf8")
        for im in images:
            original=SOURCE/"images"/im["file_name"]
            linked=dest/im["file_name"]
            if not original.is_file():
                raise FileNotFoundError(original)
            if not linked.exists():
                os.link(original,linked)
        summary["splits"][split]={"images":len(images),"positive_images":sum(bool(by_image[im["id"]]) for im in images),
                                   "positive_boxes":len(annotations),"sites":dict(Counter(im["site"] for im in images))}
    (OUTPUT/"preparation.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
