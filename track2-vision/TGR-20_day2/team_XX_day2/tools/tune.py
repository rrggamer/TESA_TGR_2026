"""จูน conf / NMS iou / min_keep บน val ให้ P1+P2 สูงสุด (รันโมเดลครั้งเดียว แล้ว sweep เฉพาะ post-process)

python tune.py --weights weights --ncnn best_ncnn_model_640 --images data/piles/val/images --gt data/piles/val/masks
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # infer.py / score.py อยู่โฟลเดอร์แม่

import argparse, sys, glob, itertools, json, os
import numpy as np
from infer import SegModel, read_image, find_roi, detect, build_instances, finalize, DEFAULTS
from score import read_mask, score_image


def cache_outputs(model, files, roi_on=True):
    out = []
    for f in files:
        img, W0, H0 = read_image(f); fx, fy = W0 / img.shape[1], H0 / img.shape[0]
        roi = find_roi(img) if roi_on else None
        if roi is None: roi = (0, 0, img.shape[1], img.shape[0])
        o0, proto, r, px, py = model.forward(img[roi[1]:roi[3], roi[0]:roi[2]])
        out.append((f, o0, proto, r, px, py, roi, fx, fy, W0, H0))
    return out


def evaluate(model, cache, gts, conf, iou, min_keep, mask_thr=0.5):
    p1, p2 = [], []
    for (f, o0, proto, r, px, py, roi, fx, fy, W0, H0), gt in zip(cache, gts):
        b, s, c, k = detect(o0, model.nc, conf, iou, 400)
        cv, kept = build_instances(b, s, k, proto, r, px, py, roi, fx, fy, W0, H0, mask_thr, min_keep)
        cv, _ = finalize(cv, kept, c, s, model.names)
        sc = score_image(gt, cv); p1.append(sc["p1"]); p2.append(sc["p2"])
    return 7.5 * np.mean(p1), 7.5 * np.mean(p2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights"); ap.add_argument("--ncnn", default="best_ncnn_model")
    ap.add_argument("--images", default="data/piles/val/images"); ap.add_argument("--gt", default="data/piles/val/masks")
    ap.add_argument("--no-roi", action="store_true")
    ap.add_argument("--confs", type=float, nargs="+", default=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
    ap.add_argument("--ious", type=float, nargs="+", default=[0.5, 0.6, 0.7])
    ap.add_argument("--keeps", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--save", help="บันทึกค่าที่ดีที่สุดลง infer_config.json ใน --weights")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.images, "*.jpg")))
    gts = [read_mask(os.path.join(a.gt, os.path.splitext(os.path.basename(f))[0] + ".png")) for f in files]
    model = SegModel(os.path.join(a.weights, a.ncnn))
    cache = cache_outputs(model, files, not a.no_roi)
    res = []
    for conf, iou, keep in itertools.product(a.confs, a.ious, a.keeps):
        P1, P2 = evaluate(model, cache, gts, conf, iou, keep)
        res.append((P1 + P2, P1, P2, conf, iou, keep))
    res.sort(reverse=True)
    print(f"{'P1+P2':>6} {'P1':>6} {'P2':>6}  conf  iou  min_keep   ({len(files)} images, {a.ncnn}, roi={not a.no_roi})")
    for t, p1, p2, c, i, k in res[:8]: print(f"{t:6.3f} {p1:6.3f} {p2:6.3f}  {c:.2f}  {i:.2f}  {k:.2f}")
    best = res[0]
    if a.save:
        cfg = dict(DEFAULTS); p = os.path.join(a.weights, "infer_config.json")
        if os.path.isfile(p): cfg.update(json.load(open(p)))
        cfg.update({"ncnn": a.save, "conf": best[3], "iou": best[4], "min_keep": best[5], "roi": not a.no_roi})
        json.dump(cfg, open(p, "w"), indent=1); print("saved", p, cfg)


if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace"); sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    main()
