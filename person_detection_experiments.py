#!/usr/bin/env python3
"""
Reproducible person-detection experiments: MobileNet-SSDLite (torchvision)
vs. recent lightweight YOLO baselines, evaluated with one common protocol.

Sub-commands
  env     log hardware / software environment (-> env.json, requirements_frozen.txt)
  split   split ONE COCO-format annotation file into train/val/test (fixed seed)
  train   fine-tune SSDLite320-MobileNetV3-Large (ImageNet-pretrained backbone)
  eval    evaluate a detector on a test set (COCO AP, P/R/F1, latency, params, FLOPs)

Detector names for `eval --model`
  ssdlite_pretrained   torchvision COCO-pretrained SSDLite (zero-shot baseline)
  ckpt:<path>          checkpoint produced by `train`
  yolo:<weights>       Ultralytics model, e.g. yolo:yolov8n.pt (needs `pip install ultralytics`)

Every number that goes into the paper must come from metrics.json files that
this script writes. Do not type numbers into the manuscript by hand.
"""
import argparse, json, os, platform, random, subprocess, sys, time
from pathlib import Path
import numpy as np

TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)


# ----------------------------------------------------------------- utilities
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def xywh2xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def iou_matrix(a, b):
    """a: (N,4) b: (M,4) xyxy -> (N,M) IoU."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def log_env(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"python": sys.version, "platform": platform.platform(),
            "machine": platform.machine(), "cpu": platform.processor(),
            "cpu_count": os.cpu_count()}
    try:  # readable CPU name on Linux / Colab
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        import torch, torchvision
        info.update(torch=torch.__version__, torchvision=torchvision.__version__,
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        pass
    json.dump(info, open(out_dir / "env.json", "w"), indent=2)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True).stdout
    (out_dir / "requirements_frozen.txt").write_text(freeze)
    return info


# ------------------------------------------------------------------- dataset
def make_split(ann_json, out_dir, seed=42, ratios=(0.7, 0.15, 0.15)):
    """Image-level split (no image appears in two subsets)."""
    d = json.load(open(ann_json))
    ids = sorted(im["id"] for im in d["images"])
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    a, b = int(ratios[0] * n), int((ratios[0] + ratios[1]) * n)
    parts = {"train": set(ids[:a]), "val": set(ids[a:b]), "test": set(ids[b:])}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, keep in parts.items():
        sub = {k: v for k, v in d.items() if k not in ("images", "annotations")}
        sub["images"] = [im for im in d["images"] if im["id"] in keep]
        sub["annotations"] = [an for an in d["annotations"] if an["image_id"] in keep]
        json.dump(sub, open(out_dir / f"{name}.json", "w"))
        print(f"{name}: {len(sub['images'])} images, {len(sub['annotations'])} boxes")
    json.dump({"seed": seed, "ratios": ratios,
               "image_ids": {k: sorted(v) for k, v in parts.items()}},
              open(out_dir / "split_ids.json", "w"))


class PersonDataset:
    """COCO-format dataset restricted to one category (person)."""

    def __init__(self, img_dir, ann_json, person_id=1, train=False, max_images=None):
        from pycocotools.coco import COCO
        self.coco = COCO(ann_json)
        self.img_dir, self.person_id, self.train = Path(img_dir), person_id, train
        self.ids = sorted(self.coco.getImgIds())[:max_images]

    def __len__(self):
        return len(self.ids)

    def anns(self, img_id):
        return self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, catIds=[self.person_id]))

    def path(self, img_id):
        return self.img_dir / self.coco.loadImgs(img_id)[0]["file_name"]

    def __getitem__(self, i):
        import torch
        from PIL import Image
        import torchvision.transforms.functional as F
        img_id = self.ids[i]
        img = Image.open(self.path(img_id)).convert("RGB")
        boxes = [xywh2xyxy(a["bbox"]) for a in self.anns(img_id)
                 if not a.get("iscrowd", 0) and a["bbox"][2] > 1 and a["bbox"][3] > 1]
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        x = F.pil_to_tensor(img).float() / 255.0
        if self.train:
            if random.random() < 0.5:                       # horizontal flip
                x = x.flip(-1)
                w = x.shape[-1]
                boxes = torch.stack([w - boxes[:, 2], boxes[:, 1],
                                     w - boxes[:, 0], boxes[:, 3]], 1) if len(boxes) else boxes
            x = F.adjust_brightness(x, random.uniform(0.7, 1.3))
            x = F.adjust_contrast(x, random.uniform(0.7, 1.3))
        tgt = {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64),
               "image_id": img_id}
        return x, tgt


def collate(batch):
    return tuple(zip(*batch))


# --------------------------------------------------------------------- models
def build_ssdlite(num_classes=2, pretrained_coco=False, init="imagenet"):
    import torch
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    from torchvision.models import MobileNet_V3_Large_Weights
    if pretrained_coco:
        return ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    if init == "coco":   # start from the COCO detector, replace the 91-class output layers
        m = ssdlite320_mobilenet_v3_large(weights="DEFAULT")
        head = m.head.classification_head
        for seq in head.module_list:
            old = seq[-1]
            n_anchor = old.out_channels // 91
            new = torch.nn.Conv2d(old.in_channels, n_anchor * num_classes, 1)
            if num_classes == 2:      # keep the COCO "background" and "person" output channels
                with torch.no_grad():
                    w = old.weight.reshape(n_anchor, 91, old.in_channels, 1, 1)[:, [0, 1]]
                    new.weight.copy_(w.reshape(n_anchor * 2, old.in_channels, 1, 1))
                    if old.bias is not None:
                        new.bias.copy_(old.bias.reshape(n_anchor, 91)[:, [0, 1]].reshape(-1))
            seq[-1] = new
        if hasattr(head, "num_columns"):
            head.num_columns = num_classes
        return m
    return ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
        num_classes=num_classes)


def set_size(model, s):
    """Change the square network input size (the weights are fully convolutional)."""
    if s and s != 320:
        model.transform.fixed_size = (s, s)
        model.transform.min_size = (s,)
        model.transform.max_size = s
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_flops(model, device, size=(1, 3, 320, 320)):
    """FLOPs of one forward pass of the network at 320x320 (2 x MACs)."""
    try:
        import torch
        from torch.utils.flop_counter import FlopCounterMode
        model.eval().to(device)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model([torch.rand(size[1:], device=device)])
        return fc.get_total_flops()
    except Exception as e:  # older torch etc.
        print("FLOP counting skipped:", e)
        return None


def count_default_boxes(model, device):
    """N_dbox = sum_k f_k^2 * n_k for the 320x320 input (feature-map side f_k, boxes per cell n_k)."""
    try:
        import torch
        model.eval().to(device)
        with torch.no_grad():
            feats = list(model.backbone(torch.rand(1, 3, *getattr(model.transform, "fixed_size", (320, 320)), device=device)).values())
        n_k = model.anchor_generator.num_anchors_per_location()
        sizes = [tuple(f.shape[-2:]) for f in feats]
        return {"feature_map_sizes": sizes, "boxes_per_location": list(n_k),
                "n_default_boxes": int(sum(h * w * n for (h, w), n in zip(sizes, n_k)))}
    except Exception as e:
        print("default-box counting skipped:", e)
        return None


# ------------------------------------------------------------------ inference
def predict_tv(model, ds, device, batch=16, workers=2):
    import torch
    model.eval().to(device)
    model.score_thresh, model.detections_per_img = 0.001, 100
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                         num_workers=workers, collate_fn=collate)
    out = []
    with torch.no_grad():
        for imgs, tgts in loader:
            res = model([i.to(device) for i in imgs])
            for r, t in zip(res, tgts):
                keep = r["labels"] == 1          # label 1 = person (COCO and fine-tuned)
                out.append({"image_id": t["image_id"],
                            "boxes": r["boxes"][keep].cpu().numpy(),
                            "scores": r["scores"][keep].cpu().numpy()})
    return out


def predict_yolo(weights, ds, device, imgsz=640):  # imgsz: native 640; also try 320 for a like-for-like input size
    from ultralytics import YOLO
    m = YOLO(weights)
    out = []
    for img_id in ds.ids:
        r = m.predict(str(ds.path(img_id)), conf=0.001, iou=0.7, imgsz=imgsz,
                      classes=[0], device=device, verbose=False, max_det=100)[0]
        out.append({"image_id": img_id, "boxes": r.boxes.xyxy.cpu().numpy(),
                    "scores": r.boxes.conf.cpu().numpy()})
    return m, out


def latency_stats(fn, warmup=20, runs=200, sync=None):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        if sync: sync()
        t0 = time.perf_counter()
        fn()
        if sync: sync()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return {"mean_ms": float(ts.mean()), "std_ms": float(ts.std()),
            "p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
            "fps": float(1000 / ts.mean()), "runs": runs, "warmup": warmup}


# ------------------------------------------------------------------ metrics
def coco_eval(ds, dets, quiet=False):
    from pycocotools.cocoeval import COCOeval
    res = [{"image_id": int(d["image_id"]), "category_id": ds.person_id,
            "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
            "score": float(s)} for d in dets for b, s in zip(d["boxes"], d["scores"])]
    names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    if not res:
        return dict.fromkeys(names, 0.0), None
    E = COCOeval(ds.coco, ds.coco.loadRes(res), "bbox")
    E.params.imgIds, E.params.catIds = ds.ids, [ds.person_id]
    E.evaluate(); E.accumulate()
    if quiet:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            E.summarize()
    else:
        E.summarize()
    return dict(zip(names, [float(x) for x in E.stats])), E


def prf1(ds, dets, tau, iou_thr=0.5):
    """Greedy one-to-one matching at confidence tau; crowd regions are ignored."""
    TP = FP = FN = 0
    for d in dets:
        anns = ds.anns(int(d["image_id"]))
        gt = np.array([xywh2xyxy(a["bbox"]) for a in anns if not a.get("iscrowd", 0)]).reshape(-1, 4)
        crowd = np.array([xywh2xyxy(a["bbox"]) for a in anns if a.get("iscrowd", 0)]).reshape(-1, 4)
        keep = d["scores"] >= tau
        boxes, sc = d["boxes"][keep], d["scores"][keep]
        boxes = boxes[np.argsort(-sc)]
        used = np.zeros(len(gt), bool)
        ious = iou_matrix(boxes, gt)
        cious = iou_matrix(boxes, crowd)
        for i in range(len(boxes)):
            cand = np.where(~used)[0]
            j = cand[np.argmax(ious[i, cand])] if len(cand) else -1
            if j >= 0 and ious[i, j] >= iou_thr:
                used[j] = True; TP += 1
            elif len(crowd) and cious[i].max() >= iou_thr:
                continue                       # inside a crowd region: ignore
            else:
                FP += 1
        FN += int((~used).sum())
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"tau": float(tau), "TP": TP, "FP": FP, "FN": FN, "precision": P, "recall": R, "f1": F}


def select_tau(ds_val, dets_val):
    """Choose the confidence threshold on VALIDATION data (max F1)."""
    best = max((prf1(ds_val, dets_val, t) for t in TAUS), key=lambda r: r["f1"])
    return best["tau"]


def save_plots(out, E, ds, dets, tau):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    if E is not None:                                  # PR curve at IoU 0.50
        pr = E.eval["precision"][0, :, 0, 0, 2]
        plt.figure(figsize=(4.5, 4)); plt.plot(np.linspace(0, 1, 101), pr)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve (IoU=0.50)")
        plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(out / "pr_curve.png", dpi=300); plt.close()
    rows = [prf1(ds, dets, t) for t in TAUS]
    plt.figure(figsize=(5, 4))
    for k in ("precision", "recall", "f1"):
        plt.plot(TAUS, [r[k] for r in rows], label=k)
    plt.axvline(tau, ls="--", c="gray"); plt.xlabel("Confidence threshold")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(out / "threshold_sweep.png", dpi=300); plt.close()
    json.dump(rows, open(out / "threshold_sweep.json", "w"), indent=1)
    # qualitative grid: blue = ground truth, green = detection >= tau
    from PIL import Image
    n = min(6, len(dets))
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    for ax, d in zip(axs.ravel(), dets[:n]):
        ax.imshow(Image.open(ds.path(d["image_id"])).convert("RGB")); ax.axis("off")
        for a in ds.anns(d["image_id"]):
            x1, y1, x2, y2 = xywh2xyxy(a["bbox"])
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="deepskyblue", lw=1.5))
        for b, s in zip(d["boxes"], d["scores"]):
            if s >= tau:
                ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="lime", lw=1.5, ls="--"))
    plt.tight_layout(); plt.savefig(out / "qualitative.png", dpi=200); plt.close()


# ------------------------------------------------------------------ commands
def cmd_train(a):
    import torch
    if not a.val_json:
        sys.exit('train needs --val-json (model selection must use validation data)')
    set_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log_env(out); json.dump(vars(a), open(out / "config.json", "w"), indent=2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = PersonDataset(a.img_dir, a.train_json, a.person_id, train=True)
    va = PersonDataset(a.img_dir, a.val_json, a.person_id)
    g = torch.Generator().manual_seed(a.seed)
    loader = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                                         collate_fn=collate, generator=g, drop_last=True)
    model = set_size(build_ssdlite(2, init=a.init), a.size).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    use_amp = dev == "cuda" and a.amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best, hist = -1.0, []
    for ep in range(a.epochs):
        model.train(); tot = 0.0; t0 = time.time()
        for imgs, tgts in loader:
            imgs = [i.to(dev) for i in imgs]
            tgts = [{k: (v.to(dev) if hasattr(v, "to") else v) for k, v in t.items()} for t in tgts]
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = sum(model(imgs, tgts).values())
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        sched.step()
        stats, _ = coco_eval(va, predict_tv(model, va, dev, workers=a.workers), quiet=True)
        hist.append({"epoch": ep + 1, "train_loss": tot / len(loader), "val_AP": stats["AP"],
                     "val_AP50": stats["AP50"], "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0})
        print(hist[-1]); json.dump(hist, open(out / "history.json", "w"), indent=1)
        if stats["AP"] > best:
            best = stats["AP"]; torch.save(model.state_dict(), out / "best.pt")
    print("best val AP:", best)


def cmd_eval(a):
    import torch
    set_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); log_env(out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    te = PersonDataset(a.img_dir, a.test_json, a.person_id)
    va = PersonDataset(a.img_dir, a.val_json, a.person_id) if a.val_json else None
    res = {"model": a.model, "device": dev, "n_test_images": len(te), "seed": a.seed}
    if a.model.startswith("yolo:"):
        w = a.model.split(":", 1)[1]
        m, dets = predict_yolo(w, te, dev, a.yolo_imgsz)
        dets_val = predict_yolo(w, va, dev, a.yolo_imgsz)[1] if va else None
        res["yolo_imgsz"] = a.yolo_imgsz
        import numpy as _np
        img = (_np.random.rand(480, 640, 3) * 255).astype("uint8")
        sync = torch.cuda.synchronize if dev == "cuda" else None
        res["latency"] = latency_stats(lambda: m.predict(img, imgsz=a.yolo_imgsz, conf=0.25, device=dev, verbose=False),
                                       sync=sync)
        res["params"] = int(sum(p.numel() for p in m.model.parameters()))
    else:
        init, size = "imagenet", a.size
        if a.model.startswith("ckpt:"):
            cfgp = Path(a.model.split(":", 1)[1]).parent / "config.json"
            if cfgp.exists():
                cfg = json.load(open(cfgp))
                init, size = cfg.get("init", "imagenet"), cfg.get("size", 320)
        model = build_ssdlite(pretrained_coco=True) if a.model == "ssdlite_pretrained" else build_ssdlite(2, init=init)
        model = set_size(model, size)
        res["input_size"] = size
        if a.model.startswith("ckpt:"):
            model.load_state_dict(torch.load(a.model.split(":", 1)[1], map_location="cpu"))
        dets = predict_tv(model, te, dev, workers=a.workers)
        dets_val = predict_tv(model, va, dev, workers=a.workers) if va else None
        model.score_thresh = select_tau(va, dets_val) if va else a.tau  # time at deployment threshold
        x = [torch.rand(3, 480, 640, device=dev)]
        sync = torch.cuda.synchronize if dev == "cuda" else None
        with torch.no_grad():
            res["latency"] = latency_stats(lambda: model(x), sync=sync)
        res["params"], res["flops_320"] = count_params(model), count_flops(model, dev)
        res["default_boxes"] = count_default_boxes(model, dev)
    tau = select_tau(va, dets_val) if va else a.tau          # tau chosen on val, applied to test
    res["tau_selected_on"] = "validation" if va else "fixed"
    res["coco"], E = coco_eval(te, dets)
    res["operating_point_iou50"] = prf1(te, dets, tau)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    save_plots(out, E, te, dets, tau)
    print(json.dumps(res, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("env"); e.add_argument("--out", default="results/env")
    s = sp.add_parser("split"); s.add_argument("--ann-json", required=True)
    s.add_argument("--out", default="splits"); s.add_argument("--seed", type=int, default=42)
    for name in ("train", "eval"):
        q = sp.add_parser(name)
        q.add_argument("--img-dir", required=True); q.add_argument("--val-json")
        q.add_argument("--person-id", type=int, default=1); q.add_argument("--seed", type=int, default=42)
        q.add_argument("--workers", type=int, default=2); q.add_argument("--out", required=True)
        if name == "train":
            q.add_argument("--train-json", required=True)
            q.add_argument("--epochs", type=int, default=30); q.add_argument("--batch", type=int, default=16)
            q.add_argument("--lr", type=float, default=0.01); q.add_argument("--wd", type=float, default=4e-5)
            q.add_argument("--amp", action="store_true")
            q.add_argument("--init", default="imagenet", choices=["imagenet", "coco"])
            q.add_argument("--size", type=int, default=320)
        else:
            q.add_argument("--model", required=True); q.add_argument("--test-json", required=True)
            q.add_argument("--tau", type=float, default=0.5)
            q.add_argument("--yolo-imgsz", type=int, default=640)
            q.add_argument("--size", type=int, default=320)
    a = p.parse_args()
    {"env": lambda: log_env(a.out), "split": lambda: make_split(a.ann_json, a.out, a.seed),
     "train": lambda: cmd_train(a), "eval": lambda: cmd_eval(a)}[a.cmd]()


if __name__ == "__main__":
    main()
