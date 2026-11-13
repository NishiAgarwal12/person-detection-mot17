#!/usr/bin/env python3
"""GPU latency of the trained models (run on a GPU machine after run_all_eval.py).
Needs results/ssdlite512_s*/best.pt, results/yolo/*/weights/best.pt, the YOLO starter weights and
results/TEST_cpu/*/metrics.json (for each model's operating threshold). Random 640x480 frames, batch size 1."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, ".")
import person_detection_experiments as m
from ultralytics import YOLO

assert torch.cuda.is_available(), "no GPU found"
sync = torch.cuda.synchronize
print("GPU:", torch.cuda.get_device_name(0))
tau_of = lambda tag: json.load(open(f"results/TEST_cpu/{tag}/metrics.json"))["operating_point_iou50"]["tau"]

def time_ssd(model, tau):
    model.eval().to("cuda"); model.score_thresh = tau
    x = [torch.rand(3, 480, 640, device="cuda")]
    with torch.no_grad():
        return m.latency_stats(lambda: model(x), sync=sync)["mean_ms"]
def time_yolo(path):
    y = YOLO(path); img = (np.random.rand(480, 640, 3) * 255).astype("uint8")
    return m.latency_stats(lambda: y.predict(img, imgsz=640, conf=0.25, device=0, verbose=False), sync=sync)["mean_ms"]

res = {}
add = lambda g, v: res.setdefault(g, []).append(v)
for s in (42, 43, 44):
    mdl = m.set_size(m.build_ssdlite(2, init="coco"), 512)
    mdl.load_state_dict(torch.load(f"results/ssdlite512_s{s}/best.pt", map_location="cpu"))
    add("SSDLite 512, fine-tuned", time_ssd(mdl, tau_of(f"ssdlite512_s{s}")))
add("SSDLite 320, COCO only", time_ssd(m.build_ssdlite(pretrained_coco=True), tau_of("ssdlite320_coco_pretrained")))
for w in ["yolov8n", "yolov10n", "yolo26n"]:
    for p in (f"results/yolo/{w}/weights/best.pt", f"results/yolo/{w}_s43/weights/best.pt", f"results/yolo/{w}_s44/weights/best.pt"):
        add(f"{w}, fine-tuned", time_yolo(p))
    add(f"{w}, COCO only", time_yolo(f"{w}.pt"))
os.makedirs("results/TEST_gpu", exist_ok=True)
json.dump(res, open("results/TEST_gpu/latency.json", "w"), indent=1)
print("model | GPU ms | FPS")
for g, v in res.items():
    v = np.array(v); sd = f" ± {v.std(ddof=1):.1f}" if len(v) > 1 else ""
    print(g, "|", f"{v.mean():.1f}{sd}", "|", f"{1000 / v.mean():.0f}")
