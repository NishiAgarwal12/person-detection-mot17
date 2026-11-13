# Lightweight person detection for surveillance video: MobileNet-SSDLite vs. recent YOLO detectors

Code, split definitions and evaluation scripts for the paper
*"Lightweight person detection for surveillance video: a reproducible benchmark of MobileNet-SSDLite and recent YOLO detectors"*
(authors: Nishi Agarwal, Mohd. Gulman Siddiqui; citation and DOI: to be added after publication).

The study fine-tunes MobileNet-SSDLite (torchvision, three input sizes, three seeds at the chosen size) and three compact
YOLO detectors (YOLOv8n, YOLOv10n, YOLO26n; three seeds each) on four MOT17 videos, selects checkpoints and thresholds on
two other videos, and evaluates every model once on a held-out video (MOT17-11).

## Data
MOT17 is **not** included. Download it from <https://motchallenge.net> and follow its terms of use (research use; check the
licence text on the site). `prepare_mot17.py` builds the splits and the resized frames from your own copy.

* Split by video: train = MOT17-04, -05, -10, -13; val = MOT17-02, -09; test = MOT17-11.
* Only the FRCNN copy of each video is used (the three public copies have identical images and ground truth).
* Every second frame is kept; frames are resized so that the longer side is 640 px.
* Persons with annotated visibility < 0.25 are treated as ignore regions (`*_vis25.json`, `iscrowd = 1`).

## Reproduce (Python 3.13, a GPU is needed for training only)
```bash
pip install -r requirements.txt

# 1. data
python prepare_mot17.py --mot-root /path/to/unzipped/MOT17 --out data
python make_yolo_dataset.py --data data

# 2. SSDLite-512, three seeds (about 10 min each on a T4)
for s in 42 43 44; do
python person_detection_experiments.py train --img-dir data/mot640 \
  --train-json data/splits_mot640/train_vis25.json --val-json data/splits_mot640/val_vis25.json \
  --epochs 8 --batch 16 --lr 0.003 --amp --init coco --size 512 --seed $s --out results/ssdlite512_s$s
done

# 3. YOLO models, three seeds each (seed 42 is stored without suffix, seeds 43/44 with _s43 / _s44)
for w in yolov8n yolov10n yolo26n; do
  yolo detect train data=$PWD/data/yolo_mot/data.yaml model=$w.pt epochs=10 imgsz=640 batch=16 seed=42 device=0 \
       project=$PWD/results/yolo name=$w exist_ok=True plots=False
  for s in 43 44; do
    yolo detect train data=$PWD/data/yolo_mot/data.yaml model=$w.pt epochs=10 imgsz=640 batch=16 seed=$s device=0 \
         project=$PWD/results/yolo name=${w}_s$s exist_ok=True plots=False
  done
done

# 4. final test evaluation (once) and summary table; run on the machine whose CPU latency you want to report
python run_all_eval.py

# 5. GPU latency (on a GPU machine)
python gpu_timing.py
```
`person_detection_experiments.py` also has `env`, `split`, `train` and `eval` sub-commands (`--help`). Every run writes its
settings (`config.json`), environment (`env.json`) and metrics (`metrics.json`). All numbers in the paper come from these files.

Note: `prepare_mot17.py`, `make_yolo_dataset.py`, `run_all_eval.py` and `gpu_timing.py` were converted from the notebook cells
used for the experiments; `prepare_mot17.py` and `make_yolo_dataset.py` were additionally tested on a small synthetic dataset.
Please report problems as GitHub issues.

## Results (test video MOT17-11, visibility rule applied; mean ± s.d. over three seeds for fine-tuned models)
| Model | AP | AP50 | F1 | CPU ms (2-core cloud CPU) | GPU ms (Tesla T4) | Params (M) |
|---|---|---|---|---|---|---|
| SSDLite 512, fine-tuned | 39.3 ± 0.1 | 68.1 ± 0.3 | 70.6 ± 0.4 | 127.0 ± 1.9 | 14.9 ± 2.3 | 2.21 |
| SSDLite 320, COCO weights only | 34.9 | 63.7 | 68.9 | 75.2 | 33.0 | 3.44 |
| YOLOv8n, fine-tuned | 54.8 ± 1.4 | 78.6 ± 0.6 | 81.0 ± 0.7 | 163.8 ± 4.0 | 8.0 ± 1.4 | 3.14 |
| YOLOv10n, fine-tuned | 56.2 ± 0.4 | 77.5 ± 0.4 | 80.7 ± 0.5 | 145.6 ± 0.6 | 9.2 ± 1.5 | 2.74 |
| YOLO26n, fine-tuned | 57.1 ± 2.9 | 80.1 ± 1.6 | 82.4 ± 1.2 | 131.7 ± 0.6 | 11.5 ± 2.0 | 2.53 |

See the paper for all rows (including COCO-weights-only YOLO models), the size breakdown and the limitations. The test set is a single
video; latency was measured on random 640 × 480 frames at batch size 1 on shared cloud hardware.

## Licence
Code: MIT (see the LICENSE file). Ultralytics is licensed separately (AGPL-3.0); check its terms before redistributing combined code. MOT17 and the pretrained weights have their own licences.
