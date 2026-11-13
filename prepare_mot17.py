#!/usr/bin/env python3
"""
Prepare MOT17 for person_detection_experiments.py.

Input : the folder where MOT17.zip was unzipped (it must contain .../train/MOT17-XX-FRCNN/).
Output: <out>/mot640/...                 resized frames (longer side <= 640 px)
        <out>/splits_mot640/{train,val,test}.json         COCO-format person boxes, all persons
        <out>/splits_mot640/{train,val,test}_vis25.json   persons with visibility < 0.25 marked
                                                          as ignore regions (iscrowd = 1)

Splits are made by video (never by frame):
    train: MOT17-04, -05, -10, -13   val: MOT17-02, -09   test: MOT17-11
Every second frame is kept. Only the FRCNN copy of each video is used because the three
public copies (DPM, FRCNN, SDP) contain identical images and ground truth.
MOT17 must be obtained from motchallenge.net and is subject to its own terms of use.
"""
import argparse, configparser, glob, json, os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from PIL import Image

SPLIT = {"train": ["04", "05", "10", "13"], "val": ["02", "09"], "test": ["11"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mot-root", required=True, help="folder where MOT17.zip was unzipped")
    ap.add_argument("--out", default=".", help="output folder")
    ap.add_argument("--step", type=int, default=2, help="keep every n-th frame")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--min-vis", type=float, default=0.25)
    a = ap.parse_args()

    seqs = sorted(glob.glob(os.path.join(a.mot_root, "**", "train", "MOT17-*-FRCNN"), recursive=True))
    if not seqs:
        raise SystemExit("no MOT17-*-FRCNN folders found under " + a.mot_root)
    root = os.path.dirname(seqs[0])
    dst = os.path.join(a.out, "mot640")
    spl = os.path.join(a.out, "splits_mot640")
    os.makedirs(spl, exist_ok=True)

    def resize_one(job):
        src, out = job
        os.makedirs(os.path.dirname(out), exist_ok=True)
        img = Image.open(src).convert("RGB")
        w, h = img.size
        sc = min(1.0, a.max_side / max(w, h))
        if sc < 1:
            img = img.resize((round(w * sc), round(h * sc)), Image.LANCZOS)
        img.save(out, quality=95)
        return sc, img.size

    img_id = ann_id = 0
    for name, ids in SPLIT.items():
        images, anns, jobs = [], [], []
        for s in seqs:
            sq = os.path.basename(s)
            if sq.split("-")[1] not in ids:
                continue
            cp = configparser.ConfigParser(); cp.read(f"{s}/seqinfo.ini")
            W, H = int(cp["Sequence"]["imWidth"]), int(cp["Sequence"]["imHeight"])
            gt = pd.read_csv(f"{s}/gt/gt.txt", header=None,
                             names=["frame", "id", "x", "y", "w", "h", "flag", "cls", "vis"])
            gt = gt[(gt.flag == 1) & (gt.cls == 1)]            # considered pedestrians only
            for fr, g in gt.groupby("frame"):
                if fr % a.step:
                    continue
                fn = f"{sq}/img1/{fr:06d}.jpg"
                if not os.path.exists(f"{root}/{fn}"):
                    continue
                img_id += 1
                images.append({"id": img_id, "file_name": fn})
                jobs.append((f"{root}/{fn}", f"{dst}/{fn}"))
                for r in g.itertuples():
                    x1, y1 = max(r.x, 0), max(r.y, 0)
                    x2, y2 = min(r.x + r.w, W), min(r.y + r.h, H)
                    if x2 - x1 < 2 or y2 - y1 < 2:
                        continue
                    ann_id += 1
                    anns.append({"id": ann_id, "image_id": img_id, "category_id": 1, "iscrowd": 0,
                                 "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                 "visibility": float(r.vis)})
        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(resize_one, jobs))
        scale = {im["id"]: res[i] for i, im in enumerate(images)}
        for im in images:
            im["width"], im["height"] = scale[im["id"]][1]
        keep = []
        for an in anns:
            sc = scale[an["image_id"]][0]
            an["bbox"] = [v * sc for v in an["bbox"]]
            if an["bbox"][2] >= 2 and an["bbox"][3] >= 2:
                an["area"] = an["bbox"][2] * an["bbox"][3]
                keep.append(an)
        data = {"images": images, "annotations": keep, "categories": [{"id": 1, "name": "person"}]}
        json.dump(data, open(f"{spl}/{name}.json", "w"))
        n = 0
        for an in keep:
            if an["visibility"] < a.min_vis:
                an["iscrowd"] = 1; n += 1
        json.dump(data, open(f"{spl}/{name}_vis25.json", "w"))
        print(f"{name}: {len(images)} images, {len(keep)} boxes, {n} ignored (visibility < {a.min_vis})")


if __name__ == "__main__":
    main()
