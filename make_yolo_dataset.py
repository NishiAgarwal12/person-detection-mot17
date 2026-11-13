#!/usr/bin/env python3
"""Convert the train/val splits made by prepare_mot17.py to the YOLO format used by Ultralytics.
Only persons with visibility >= --min-vis are labelled (same rule as in the SSDLite experiments).
Usage: python make_yolo_dataset.py --data data        (writes data/yolo_mot/ and data/yolo_mot/data.yaml)"""
import argparse, json, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--min-vis", type=float, default=0.25)
a = ap.parse_args()
Y = os.path.abspath(os.path.join(a.data, "yolo_mot"))
for name in ["train", "val"]:
    d = json.load(open(os.path.join(a.data, "splits_mot640", f"{name}.json")))
    os.makedirs(f"{Y}/{name}/images", exist_ok=True); os.makedirs(f"{Y}/{name}/labels", exist_ok=True)
    by = {}
    for an in d["annotations"]:
        if an["visibility"] >= a.min_vis:
            by.setdefault(an["image_id"], []).append(an)
    for im in d["images"]:
        stem = im["file_name"].replace("/", "_").rsplit(".", 1)[0]
        shutil.copy(os.path.join(a.data, "mot640", im["file_name"]), f"{Y}/{name}/images/{stem}.jpg")
        with open(f"{Y}/{name}/labels/{stem}.txt", "w") as f:
            for an in by.get(im["id"], []):
                x, y, w, h = an["bbox"]
                f.write(f"0 {(x+w/2)/im['width']:.6f} {(y+h/2)/im['height']:.6f} {w/im['width']:.6f} {h/im['height']:.6f}\n")
open(f"{Y}/data.yaml", "w").write(f"path: {Y}\ntrain: train/images\nval: val/images\nnames:\n  0: person\n")
print("YOLO dataset written to", Y)
