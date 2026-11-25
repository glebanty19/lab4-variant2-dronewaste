#!/usr/bin/env python3
"""Resumable parallel download of the official 3.9 GB DroneWaste archive."""
from __future__ import annotations

import hashlib
import json
import math
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT=Path(__file__).resolve().parents[1]/"datasets/dronewaste"
URL="https://zenodo.org/records/17045559/files/images.tar.gz?download=1"
ANNOTATION_URL="https://zenodo.org/records/17045559/files/dronewaste_v1.0.json?download=1"
SIZE=3882465332
MD5="0e2c2b2424737b94e9cfa28ff90e6544"
ANNOTATION_MD5="87469b3641aade7c4580b8f0ed5c6300"
CHUNK=32*1024*1024
WORKERS=8


def fetch(index: int, archive: Path) -> int:
    start=index*CHUNK;end=min(SIZE-1,start+CHUNK-1)
    for attempt in range(10):
        try:
            with requests.get(URL,headers={"Range":f"bytes={start}-{end}"},stream=True,timeout=(20,120)) as response:
                response.raise_for_status()
                if response.status_code!=206 or response.headers.get("Content-Range","").split("/")[0]!=f"bytes {start}-{end}":
                    raise RuntimeError(f"Incorrect range response for part {index}")
                received=0
                with archive.open("r+b") as stream:
                    stream.seek(start)
                    for block in response.iter_content(2*1024*1024):
                        if block:
                            stream.write(block);received+=len(block)
                if received!=end-start+1:
                    raise RuntimeError(f"Short part {index}: {received}")
            return index
        except Exception:
            if attempt==9:raise
            time.sleep(min(60,2**attempt))
    raise AssertionError("unreachable")


def file_md5(path: Path) -> str:
    digest=hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_annotation() -> None:
    target=ROOT/"dronewaste_v1.0.json"
    if target.is_file() and file_md5(target)==ANNOTATION_MD5:
        return
    temporary=ROOT/"dronewaste_v1.0.json.part"
    with requests.get(ANNOTATION_URL,stream=True,timeout=(20,120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for block in response.iter_content(1024*1024):
                if block:stream.write(block)
    if file_md5(temporary)!=ANNOTATION_MD5:
        raise RuntimeError("DroneWaste annotation checksum mismatch")
    temporary.replace(target)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    archive=ROOT/"images.tar.gz";state=ROOT/"images_download_parts.json"
    parts=math.ceil(SIZE/CHUNK)
    if state.exists():
        done=set(json.loads(state.read_text(encoding="utf8")))
    else:
        old_size=archive.stat().st_size if archive.exists() else 0
        done=set(range(old_size//CHUNK))
        state.write_text(json.dumps(sorted(done)),encoding="utf8")
    with archive.open("r+b" if archive.exists() else "w+b") as stream:
        stream.truncate(SIZE)
    missing=[i for i in range(parts) if i not in done]
    print(f"Downloading {len(missing)} parts with {WORKERS} workers",flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures=[pool.submit(fetch,i,archive) for i in missing]
        for future in as_completed(futures):
            index=future.result();done.add(index)
            state.write_text(json.dumps(sorted(done)),encoding="utf8")
            if len(done)%8==0 or len(done)==parts:
                print(f"Completed {len(done)}/{parts} parts ({len(done)*CHUNK/1e9:.2f} GB)",flush=True)
    if file_md5(archive)!=MD5:
        raise RuntimeError("DroneWaste image archive MD5 mismatch")
    print("Checksum OK; extracting",flush=True)
    with tarfile.open(archive,"r:gz") as source:
        source.extractall(ROOT,filter="data")
    ensure_annotation()
    (ROOT/"download_verified.json").write_text(json.dumps({"url":URL,"bytes":SIZE,"md5":MD5,
        "annotation_url":ANNOTATION_URL,"annotation_md5":ANNOTATION_MD5,
        "image_files":len(list((ROOT/"images").glob("*.png")))},indent=2),encoding="utf8")
    print("DroneWaste ready",flush=True)


if __name__=="__main__":main()
