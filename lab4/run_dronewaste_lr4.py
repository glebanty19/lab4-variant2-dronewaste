#!/usr/bin/env python3
"""Reproduce LR4 variant 2 on the aerial DroneWaste pile classes."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATA="data/dronewaste_piles"
BASE="dronewaste_piles_head"
FINE="dronewaste_piles_fine"


def run(*arguments: str) -> None:
    print("RUN", " ".join(arguments),flush=True)
    subprocess.run([sys.executable,*arguments],cwd=ROOT,check=True)


def evaluate(run_name: str,mode: str,extra: tuple[str,...]=(),force: bool=False) -> None:
    out=ROOT/"runs"/run_name/mode
    path=out/f"{mode}_metrics.json"
    if path.exists() and not force:
        return
    val_path=ROOT/"runs"/run_name/"val"/"val_metrics.json"
    threshold=None
    if mode!="val":
        threshold=json.loads(val_path.read_text(encoding="utf8"))["best_f1_threshold"]["confidence"]
    command=["cdw_detector.py","evaluate","--data",DATA,"--checkpoint",f"runs/{run_name}/best.pt",
             "--output",f"runs/{run_name}/{mode}","--split","val" if mode=="val" else "test", "--prefix",mode]
    if threshold is not None:command.extend(("--threshold",str(threshold)))
    run(*command,*extra)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-train",action="store_true",help="Repeat model training even if checkpoints exist")
    args=parser.parse_args()
    run("prepare_dronewaste_lr4.py")
    run("audit_lr4.py")
    for run_name,epochs,freeze in ((BASE,4,True),(FINE,6,False)):
        checkpoint=ROOT/"runs"/run_name/"best.pt"
        retrain=args.force_train or not checkpoint.exists()
        if retrain:
            command=["cdw_detector.py","train","--data",DATA,"--output",f"runs/{run_name}",
                     "--epochs",str(epochs),"--patience","2","--seed","42","--workers","2",
                     "--min-size","640","--max-size","640"]
            if freeze:command.append("--freeze-backbone")
            run(*command)
        evaluate(run_name,"val",force=retrain)
        evaluate(run_name,"test",force=retrain)
    for candidate in ("dronewaste_piles_weighted_fine",):
        if (ROOT/"runs"/candidate/"best.pt").is_file():
            evaluate(candidate,"val")
    run("select_lr4_model.py")
    selected=json.loads((ROOT/"runs/variant2_audit/selected_lr4_model.json").read_text(encoding="utf8"))["run"]
    evaluate(selected,"test",force=args.force_train and selected==FINE)
    evaluate(selected,"test_x2",("--degrade-x2",),force=args.force_train and selected==FINE)
    evaluate(selected,"test_x4",("--degrade-factor","4"),force=args.force_train and selected==FINE)
    evaluate(selected,"test_tiles",("--tile-size","320","--overlap","0.2"),force=args.force_train and selected==FINE)
    run("analyze_lr4.py")
    run("build_dronewaste_lr4_report.py")
    exact=ROOT/"Лаб4_Бондар_Аерокосмічні_купи.docx"
    previous=ROOT/"Лаб4_Бондар.docx"
    backup=ROOT/"Лаб4_ReCoDeWaste_додатковий.docx"
    if previous.exists() and not backup.exists():shutil.copyfile(previous,backup)
    shutil.copyfile(exact,previous)
    shutil.copyfile(exact,ROOT/"Lab4_Variant2_Bondar_FINAL.docx")
    print("LR4 variant 2 complete:",exact,flush=True)


if __name__=="__main__":main()
