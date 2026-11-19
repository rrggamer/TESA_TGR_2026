"""ขั้นที่ 4: local scorer เลียนแบบเกณฑ์กรรมการ (ไม่คิดคลาส)

ต่อภาพ: IoU matrix ระหว่างเม็ดจริง (G) กับเม็ดที่ทาย (P) → Hungarian หาคู่ 1:1 ที่ผลรวม IoU สูงสุด
  P1_img = ΣIoU(คู่) / max(G, P)        (เม็ดจริงที่ไม่มีคู่ + เม็ดที่ทายเกิน ได้ IoU = 0)
  P2_img = #คู่ / max(G, P)             (คู่ = คู่จาก Hungarian ที่ IoU > --pair-thr; ค่าเริ่มต้น 0 = ต้องทับกันจริง)
  P1 = 7.5 × mean(P1_img),  P2 = 7.5 × mean(P2_img)
ภาพที่ทั้ง G=0 และ P=0 นับเป็น 1.0

python score.py --gt data/piles/val/masks --pred runs_infer/val/seg_mask
"""
import argparse, glob, os, sys
import cv2, numpy as np
from scipy.optimize import linear_sum_assignment


F1_THRS = (0.5, 0.75, 0.9, 0.95)


def read_mask(path):
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None: raise FileNotFoundError(path)
    if m.ndim == 3: m = m[..., 0]
    return m if m.dtype in (np.uint8, np.uint16) else m.astype(np.int32)     # เก็บแบบประหยัด RAM


def relabel(m):
    """id อาจไม่ต่อเนื่อง → map เป็น 0..K (ตัด id ที่ไม่มีพิกเซล)"""
    m = m.astype(np.int64, copy=False).ravel()
    present = np.bincount(m) > 0; present[0] = True
    lut = np.cumsum(present) - 1                                   # 0 → 0, id ที่มีอยู่ → 1..K
    return lut[m], int(present.sum()) - 1


def iou_matrix(gt, pred):
    g, G = relabel(gt); p, P = relabel(pred)
    inter = np.bincount(g * (P + 1) + p, minlength=(G + 1) * (P + 1)).reshape(G + 1, P + 1)[1:, 1:]
    ag = np.bincount(g, minlength=G + 1)[1:]; ap = np.bincount(p, minlength=P + 1)[1:]
    union = ag[:, None] + ap[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def score_image(gt, pred, pair_thr=0.0):
    """→ dict(p1, p2, G, P, pairs, miou_matched)"""
    if gt.shape != pred.shape: raise ValueError(f"ขนาด mask ไม่เท่ากัน gt={gt.shape} pred={pred.shape}")
    M = iou_matrix(gt, pred); G, P = M.shape; D = max(G, P)
    if D == 0: return {"p1": 1.0, "p2": 1.0, "G": 0, "P": 0, "pairs": 0, "miou": 1.0, **{f"f1@{t}": 1.0 for t in F1_THRS}, **{f"tp{t}": 0 for t in F1_THRS}}
    if G == 0 or P == 0: return {"p1": 0.0, "p2": 0.0, "G": G, "P": P, "pairs": 0, "miou": 0.0, **{f"f1@{t}": 0.0 for t in F1_THRS}, **{f"tp{t}": 0 for t in F1_THRS}}
    r, c = linear_sum_assignment(-M); v = M[r, c]
    ok = v > pair_thr
    out = {"p1": float(v[ok].sum() / D), "p2": float(ok.sum() / D), "G": G, "P": P, "pairs": int(ok.sum()),
           "miou": float(v[ok].mean()) if ok.any() else 0.0}
    for t in F1_THRS:                                              # สเปก DAY2_INFER_SPEC: F1 ที่ mask IoU ≥ 0.95
        tp = int((v >= t).sum()); out[f"tp{t}"] = tp; out[f"f1@{t}"] = 2 * tp / (G + P)
    return out


def score_dirs(gt_dir, pred_dir, pair_thr=0.0, verbose=False):
    gts = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    if not gts: raise FileNotFoundError(f"ไม่พบ GT ใน {gt_dir}")
    rows = []
    for g in gts:
        name = os.path.basename(g); pp = os.path.join(pred_dir, name)
        gt = read_mask(g)
        pred = read_mask(pp) if os.path.isfile(pp) else np.zeros_like(gt)   # ไม่มีไฟล์ = ทายไม่ได้เลย
        s = score_image(gt, pred, pair_thr); s["name"] = name; s["p2_50"] = score_image(gt, pred, 0.5)["p2"]
        rows.append(s)
        if verbose:
            print(f"{name:28s} G={s['G']:3d} P={s['P']:3d} pairs={s['pairs']:3d} IoU(matched)={s['miou']:.3f} "
                  f"P1img={s['p1']:.3f} P2img={s['p2']:.3f} F1@0.95={s['f1@0.95']:.3f}")
    P1 = 7.5 * np.mean([r["p1"] for r in rows]); P2 = 7.5 * np.mean([r["p2"] for r in rows])
    f1 = {}                                                         # F1 แบบ micro (รวม TP/G/P ทุกภาพ) และ macro (เฉลี่ยต่อภาพ)
    for t in F1_THRS:
        tp = sum(r[f"tp{t}"] for r in rows); gp = sum(r["G"] + r["P"] for r in rows)
        f1[t] = (2 * tp / gp if gp else 1.0, float(np.mean([r[f"f1@{t}"] for r in rows])))
    return {"P1": P1, "P2": P2, "P1+P2": P1 + P2, "P2@0.5": 7.5 * np.mean([r["p2_50"] for r in rows]),
            "mIoU": np.mean([r["miou"] for r in rows]), "n": len(rows), "rows": rows, "F1": f1}


def latency_points(median_s, p1p2):
    pts = 10 if median_s <= 1.0 else 7 if median_s <= 2.0 else 4 if median_s <= 3.0 else 0
    return pts * min(1.0, p1p2 / 7.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True); ap.add_argument("--pred", required=True)
    ap.add_argument("--pair-thr", type=float, default=0.0)
    ap.add_argument("--latency", type=float, help="median latency (s) → คำนวณคะแนนรวมด้วย")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    r = score_dirs(a.gt, a.pred, a.pair_thr, a.verbose)
    print(f"images={r['n']}  P1={r['P1']:.3f}/7.5  P2={r['P2']:.3f}/7.5  P1+P2={r['P1+P2']:.3f}/15  "
          f"(P2@IoU0.5={r['P2@0.5']:.3f}, matched mIoU={r['mIoU']:.3f})")
    print("F1 (micro / macro ต่อภาพ): " + "  ".join(f"@{t}: {mi:.3f}/{ma:.3f}" for t, (mi, ma) in r["F1"].items()))
    if a.latency is not None:
        L = latency_points(a.latency, r["P1+P2"])
        print(f"latency {a.latency:.2f}s → {L:.2f}/10   TOTAL={r['P1+P2'] + L:.2f}/25")


if __name__ == "__main__":
    sys.exit(main())
