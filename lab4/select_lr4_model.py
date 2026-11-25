#!/usr/bin/env python3
"""Select the LR4 pile detector strictly by held-out validation mAP."""
from __future__ import annotations

import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CANDIDATES={
    "dronewaste_piles_fine":"Повне донавчання, звичайне перемішування",
    "dronewaste_piles_weighted_fine":"Повне донавчання, вага позитивних кадрів ×4",
}


def main() -> None:
    rows=[]
    for name,description in CANDIDATES.items():
        path=ROOT/"runs"/name/"val/val_metrics.json"
        if path.is_file():
            row=json.loads(path.read_text(encoding="utf8"))
            rows.append({"run":name,"description":description,"val_map50_95":row["map50_95"],
                         "val_map50":row["map50"]})
    if not rows:
        raise FileNotFoundError("No LR4 fine-tuned candidate has validation metrics")
    best=max(rows,key=lambda row:row["val_map50_95"])
    output={"selection":"Maximum validation mAP@[0.5:0.95] on site5; test not used",
            "run":best["run"],"checkpoint":f"runs/{best['run']}/best.pt",
            "validation_map50_95":best["val_map50_95"],"candidates":rows}
    path=ROOT/"runs/variant2_audit/selected_lr4_model.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps(output,indent=2,ensure_ascii=True),flush=True)


if __name__=="__main__":main()
