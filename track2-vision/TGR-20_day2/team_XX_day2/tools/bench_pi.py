"""benchmark บน Raspberry Pi จริง: median latency end-to-end ต่อภาพ + อุณหภูมิชิป + RAM

python bench_pi.py --input test_images/ --weights weights/ --rounds 3
python bench_pi.py --input test_images/ --weights weights/ --ncnn best_ncnn_model_416 --gt gt_masks/   # + คะแนน

ใช้ pipeline เดียวกับ infer.py ทุกขั้น (อ่านภาพ → ... → เขียน seg_mask/vis) และเขียนผลลง --out
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # infer.py อยู่โฟลเดอร์แม่

import argparse, json, os, platform, subprocess, sys, time
import numpy as np
import infer


def temp_c():
    for cmd in (["vcgencmd", "measure_temp"],):
        try:
            s = subprocess.run(cmd, capture_output=True, text=True, timeout=2).stdout   # "temp=52.1'C"
            return float(s.split("=")[1].split("'")[0])
        except Exception:
            pass
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except Exception:
        return float("nan")


def throttled():
    try: return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout.strip()
    except Exception: return "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--weights", default="weights")
    ap.add_argument("--out", default="bench_out"); ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--ncnn"); ap.add_argument("--threads", type=int); ap.add_argument("--conf", type=float)
    ap.add_argument("--no-roi", action="store_true"); ap.add_argument("--cls-model"); ap.add_argument("--no-cls-model", action="store_true")
    ap.add_argument("--decode-min-side", dest="decode_min_side", type=int)
    ap.add_argument("--gt", help="โฟลเดอร์ GT mask (ถ้ามี) → คำนวณ P1/P2")
    ap.add_argument("--cooldown", type=float, default=0, help="พักกี่วินาทีระหว่างภาพ")
    a = ap.parse_args()
    a.iou = a.min_keep = a.mask_thr = None
    cfg = infer.load_cfg(a.weights, a)
    files = infer.list_images(a.input)
    for d in ("seg_mask", "vis"): os.makedirs(os.path.join(a.out, d), exist_ok=True)
    print(f"{platform.machine()} {platform.platform()}  python {platform.python_version()}  cfg={cfg}")
    print(f"temp start {temp_c():.1f}C  throttled={throttled()}")
    t0 = time.perf_counter()
    model = infer.load_model(a.weights, cfg)
    print(f"load model {time.perf_counter() - t0:.2f}s")
    model.forward(np.full((model.S, model.S, 3), 114, np.uint8))             # warm-up
    lat, temps, preds = [], [], {}
    for r in range(a.rounds):
        for f in files:
            t = time.perf_counter()
            rec, _ = infer.process(model, f, cfg, a.out)
            dt = time.perf_counter() - t; lat.append(dt); temps.append(temp_c()); preds[rec["image_name"]] = rec
            print(f"round {r} {rec['image_name']}: {len(rec['objects'])} beans {dt * 1000:.0f} ms  {temps[-1]:.1f}C")
            if a.cooldown: time.sleep(a.cooldown)
    json.dump(list(preds.values()), open(os.path.join(a.out, "predictions.json"), "w"))
    try:
        import resource                                                      # Linux/Pi เท่านั้น
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        rss = float("nan")
    print(f"\nMEDIAN {np.median(lat) * 1000:.0f} ms  mean {np.mean(lat) * 1000:.0f}  p90 {np.percentile(lat, 90) * 1000:.0f}  "
          f"max {np.max(lat) * 1000:.0f} ms   ({len(lat)} runs)")
    print(f"temp max {np.nanmax(temps):.1f}C  end {temps[-1]:.1f}C  throttled={throttled()}  peak RSS {rss:.0f} MB")
    if a.gt:
        from score import score_dirs, latency_points
        s = score_dirs(a.gt, os.path.join(a.out, "seg_mask"))
        L = latency_points(float(np.median(lat)), s["P1+P2"])
        print(f"P1={s['P1']:.3f} P2={s['P2']:.3f} P1+P2={s['P1+P2']:.3f}  latency pts={L:.2f}  TOTAL={s['P1+P2'] + L:.2f}/25")


if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace"); sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    main()
