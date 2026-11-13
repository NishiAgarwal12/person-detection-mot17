#!/usr/bin/env python3
"""Final test evaluation of all models (run once, after training) and summary table.
Results are written to results/TEST_cpu/<model>/metrics.json (finished models are skipped, so it can be resumed).
Run it on the hardware whose latency you want to report (CPU runtime for the CPU column)."""
import json, os, subprocess, sys
import numpy as np

IMG, OUT = "data/mot640", "results/TEST_cpu"
jobs = {f"ssdlite512_s{s}": f"ckpt:results/ssdlite512_s{s}/best.pt" for s in (42, 43, 44)}
jobs["ssdlite320_coco_pretrained"] = "ssdlite_pretrained"
for w in ["yolov8n", "yolov10n", "yolo26n"]:
    jobs[f"{w}_ft_s42"] = f"yolo:results/yolo/{w}/weights/best.pt"
    for s in (43, 44):
        jobs[f"{w}_ft_s{s}"] = f"yolo:results/yolo/{w}_s{s}/weights/best.pt"
    jobs[f"{w}_pretrained"] = f"yolo:{w}.pt"

for tag, model in jobs.items():
    if os.path.exists(f"{OUT}/{tag}/metrics.json"):
        print(tag, "already done"); continue
    subprocess.run([sys.executable, "person_detection_experiments.py", "eval", "--img-dir", IMG,
                    "--val-json", "data/splits_mot640/val_vis25.json", "--test-json", "data/splits_mot640/test_vis25.json",
                    "--model", model, "--yolo-imgsz", "640", "--out", f"{OUT}/{tag}"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(tag, "done" if os.path.exists(f"{OUT}/{tag}/metrics.json") else "FAILED")

def agg(tags):
    R = [json.load(open(f"{OUT}/{t}/metrics.json")) for t in tags if os.path.exists(f"{OUT}/{t}/metrics.json")]
    if not R: return None
    def ms(f, k=1):
        v = np.array([f(r) * k for r in R]); return f"{v.mean():.1f}" + (f" ± {v.std(ddof=1):.1f}" if len(v) > 1 else "")
    c = lambda key: (lambda r: r["coco"][key]); o = lambda key: (lambda r: r["operating_point_iou50"][key])
    return [len(R), ms(c("AP"), 100), ms(c("AP50"), 100), ms(c("AP75"), 100), ms(c("APs"), 100), ms(c("APm"), 100),
            ms(c("APl"), 100), ms(o("precision"), 100), ms(o("recall"), 100), ms(o("f1"), 100),
            ms(lambda r: r["latency"]["mean_ms"]), f"{R[0]['params']/1e6:.2f}"]
groups = {"SSDLite 512 (fine-tuned)": [f"ssdlite512_s{s}" for s in (42, 43, 44)], "SSDLite 320 (COCO only)": ["ssdlite320_coco_pretrained"]}
for w in ["yolov8n", "yolov10n", "yolo26n"]:
    groups[f"{w} (fine-tuned)"] = [f"{w}_ft_s{s}" for s in (42, 43, 44)]
    groups[f"{w} (COCO only)"] = [f"{w}_pretrained"]
print("\nmodel | n | AP | AP50 | AP75 | APs | APm | APl | P | R | F1 | latency ms | params(M)")
for g, tags in groups.items():
    a = agg(tags); print(g, "|", " | ".join(map(str, a)) if a else "missing")
