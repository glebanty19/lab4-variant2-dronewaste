#!/usr/bin/env python3
"""Analyze LR4 DroneWaste detections and error examples."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from cdw_detector import iou

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs/variant2_audit"


def read_detections(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def size_label(box):
    area = max(0, box[2]-box[0])*max(0,box[3]-box[1])
    return "small" if area < 32**2 else "medium" if area < 96**2 else "large"


def area_ap(predictions, truths, label, threshold, size):
    """101-point AP with out-of-area GT ignored, following COCO area logic."""
    in_range = {iid:[j for j,t in enumerate(items) if t["label"] == label and size_label(t["box"]) == size]
                for iid,items in truths.items()}
    support = sum(map(len,in_range.values()))
    if not support:
        return None, 0
    matched = defaultdict(set)
    ranked = sorted(((score,p["image_id"],box) for p in predictions
                     for box,lab,score in zip(p["boxes"],p["labels"],p["scores"]) if lab == label),reverse=True)
    tp=fp=0; recalls=[]; precisions=[]
    for score,iid,box in ranked:
        gt=truths[iid]
        choices=[(iou(box,gt[j]["box"]),j) for j in in_range[iid] if j not in matched[iid]]
        overlap,index=max(choices,default=(0.,-1))
        if overlap >= threshold:
            tp+=1;matched[iid].add(index)
        else:
            outside=[t for t in gt if t["label"]==label and size_label(t["box"])!=size]
            if size_label(box)!=size or any(iou(box,t["box"])>=threshold for t in outside):
                continue
            fp+=1
        recalls.append(tp/support);precisions.append(tp/(tp+fp))
    if not recalls:
        return 0.,support
    envelope=np.maximum.accumulate(np.asarray(precisions[::-1]))[::-1]
    ap=float(np.mean([max((float(envelope[i]) for i,r in enumerate(recalls) if r>=t),default=0.)
                      for t in np.linspace(0,1,101)]))
    return ap,support


def lr4() -> None:
    run="dronewaste_piles_fine"
    choice=OUT/"selected_lr4_model.json"
    if choice.is_file():
        run=json.loads(choice.read_text(encoding="utf8"))["run"]
    prefix="lr4_exact"
    image_root=ROOT/"data/dronewaste_piles/test_rgb"
    coco=json.loads((image_root/"_annotations.coco.json").read_text(encoding="utf-8-sig"))
    names={c["id"]:c["name"] for c in coco["categories"] if c["id"]!=0}
    val=json.loads((ROOT/f"runs/{run}/val/val_metrics.json").read_text(encoding="utf8"))
    threshold=val["best_f1_threshold"]["confidence"]
    data=read_detections(ROOT/f"runs/{run}/test/test_detections.json.gz")
    preds,truths=data["predictions"],data["truths"]
    areas={}
    for size in ("small","medium","large"):
        areas[size]={}
        for label,name in names.items():
            values=[]; support=0
            for iou_threshold in np.arange(.5,.951,.05):
                ap,support=area_ap(preds,truths,label,float(iou_threshold),size)
                if ap is not None: values.append(ap)
            areas[size][name]={"support":support,"ap50_95":float(np.mean(values)) if values else None,
                               "ap50":values[0] if values else None}
        valid=[v["ap50_95"] for v in areas[size].values() if v["ap50_95"] is not None]
        areas[size]["mAP50_95"]=float(np.mean(valid)) if valid else None
    by_scale={"x1":{size:areas[size]["mAP50_95"] for size in ("small","medium","large")}}
    for mode in ("test_x2","test_x4","test_tiles"):
        path=ROOT/f"runs/{run}/{mode}/{mode}_detections.json.gz"
        if not path.is_file():
            continue
        scaled=read_detections(path)
        by_scale[mode.replace("test_","")]={}
        for size in ("small","medium","large"):
            class_aps=[]
            for label in names:
                values=[area_ap(scaled["predictions"],scaled["truths"],label,float(iou_threshold),size)[0]
                        for iou_threshold in np.arange(.5,.951,.05)]
                valid=[v for v in values if v is not None]
                if valid:class_aps.append(float(np.mean(valid)))
            by_scale[mode.replace("test_","")][size]=float(np.mean(class_aps)) if class_aps else None
    counts=Counter(); misses=Counter()
    examples=defaultdict(list)
    for p in preds:
        gt=truths[p["image_id"]];matched=set()
        ranked=sorted(zip(p["scores"],p["boxes"],p["labels"]),reverse=True)
        for score,box,lab in ranked:
            if score<threshold: break
            same=[(iou(box,t["box"]),j) for j,t in enumerate(gt) if t["label"]==lab]
            all_overlaps=[(iou(box,t["box"]),j) for j,t in enumerate(gt)]
            unmatched=[(overlap,j) for overlap,j in same if j not in matched]
            best_unmatched=max(unmatched,default=(0.,-1))
            closest=max(all_overlaps,default=(0.,-1))[1]
            if best_unmatched[0]>=.5:
                matched.add(best_unmatched[1]);counts["tp"]+=1
                kind="tp"
            elif max(same,default=(0.,-1))[0]>=.5:
                counts["duplicate"]+=1
                kind="duplicate"
            elif max((overlap for overlap,j in all_overlaps if gt[j]["label"]!=lab),default=0.)>=.5:
                counts["class"]+=1
                kind="class"
            elif max(same,default=(0.,-1))[0]>=.1:
                counts["localization"]+=1
                kind="localization"
            elif max(all_overlaps,default=(0.,-1))[0]>=.1:
                counts["class_and_localization"]+=1
                kind="class_and_localization"
            else:
                counts["background"]+=1
                kind="background"
            if len(examples[kind])<10 and all(e["image_id"]!=p["image_id"] for e in examples[kind]):
                examples[kind].append({"image_id":p["image_id"],"prediction":box,"gt":gt[closest]["box"] if closest>=0 else None,
                                       "confidence":score,"predicted_class":lab,"gt_class":gt[closest]["label"] if closest>=0 else None})
        for j,t in enumerate(gt):
            if j not in matched:
                misses[size_label(t["box"])]+=1
                if len(examples["miss"])<10 and all(e["image_id"]!=p["image_id"] for e in examples["miss"]):
                    examples["miss"].append({"image_id":p["image_id"],"prediction":None,"gt":t["box"],"gt_class":t["label"]})
    counts["miss"]=sum(misses.values())
    output={"area_ap":areas,"area_ap_by_scale":by_scale,"errors":dict(counts),"misses_by_size":dict(misses),
            "definitions":{"small":"area < 32^2 px","medium":"32^2 <= area < 96^2 px","large":"area >= 96^2 px",
                           "localization":"same class, 0.1 <= IoU < 0.5","matching":f"confidence >= {threshold:.3f}, IoU >= 0.5"}}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f"{prefix}_size_errors.json").write_text(json.dumps(output,indent=2),encoding="utf8")
    (OUT/f"{prefix}_error_examples.json").write_text(json.dumps(examples,indent=2),encoding="utf8")
    make_galleries(examples,image_root,prefix)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    size_keys=("small","medium","large")
    axes[0].bar(size_keys,[areas[k]["mAP50_95"] for k in size_keys]);axes[0].set(ylabel="mAP@[.5:.95]",title="AP by COCO object area")
    error_keys=("class","localization","class_and_localization","duplicate","background","miss")
    axes[1].bar(range(len(error_keys)),[counts[k] for k in error_keys]);axes[1].set_xticks(range(len(error_keys)),error_keys,rotation=40,ha="right")
    axes[1].set(title="Error categories",ylabel="Count")
    fig.tight_layout();fig.savefig(OUT/f"{prefix}_size_errors.png",dpi=160);plt.close(fig)
    print(json.dumps(output,indent=2))


def make_galleries(examples, root: Path, prefix: str) -> None:
    from PIL import ImageDraw
    coco=json.loads((root/"_annotations.coco.json").read_text(encoding="utf-8-sig"))
    names={str(im["id"]):im["file_name"] for im in coco["images"]}
    for kind,items in examples.items():
        sheet=Image.new("RGB",(5*320,2*265),"white")
        draw_sheet=ImageDraw.Draw(sheet)
        for index,item in enumerate(items):
            with Image.open(root/names[item["image_id"]]) as source:
                image=source.convert("RGB")
            focus=item["prediction"] or item["gt"]
            cx=(focus[0]+focus[2])/2;cy=(focus[1]+focus[3])/2
            span=max(280,focus[2]-focus[0]+100,focus[3]-focus[1]+100)
            left=max(0,min(image.width-span,cx-span/2));top=max(0,min(image.height-span*.69,cy-span*.345))
            right=min(image.width,left+span);bottom=min(image.height,top+span*.69)
            crop=image.crop((int(left),int(top),int(right),int(bottom))).resize((320,240))
            draw=ImageDraw.Draw(crop)
            for key,color in (("gt","lime"),("prediction","red")):
                box=item.get(key)
                if box:
                    xy=((box[0]-left)*320/(right-left),(box[1]-top)*240/(bottom-top),
                        (box[2]-left)*320/(right-left),(box[3]-top)*240/(bottom-top))
                    draw.rectangle(xy,outline=color,width=3)
            x=(index%5)*320;y=(index//5)*265
            sheet.paste(crop,(x,y+20))
            draw_sheet.text((x+3,y+3),f"{kind} #{index+1}, image {item['image_id']}",fill="black")
        sheet.save(OUT/f"{prefix}_examples_{kind}.png")



if __name__ == "__main__":
    lr4()
