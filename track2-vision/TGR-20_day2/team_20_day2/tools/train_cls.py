"""train ตัวจำแนกคลาสต่อเม็ด (สี+รูปทรง 12 ค่า → StandardScaler + LogReg) สำหรับโมเดล segmentation 1 คลาส

feature คำนวณจาก mask ที่ "โมเดลทายจริง" (ผ่าน pipeline เดียวกับ infer.py) ไม่ใช่ GT → เห็นเม็ดที่ถูกทับ/ขอบไม่เป๊ะ
เหมือนตอน test; label = class ของ GT ที่จับคู่ได้ (Hungarian, IoU ≥ 0.5)
train: กองสังเคราะห์ฝั่ง train + ภาพจริงของ source ฝั่ง train | val: กองฝั่ง val + ภาพจริงของ source ฝั่ง val

python train_cls.py --weights exp/crop1_o2o
→ <weights>/cls_head.json (ใช้ numpy ล้วนตอน infer)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # infer.py / score.py อยู่โฟลเดอร์แม่

import argparse, sys, csv, glob, json, os
import cv2, numpy as np
from scipy.optimize import linear_sum_assignment
import infer
from score import iou_matrix, read_mask
from common import CLASSES
from make_piles import val_sources


def predict_objs(model, path, cfg):
    img, W0, H0 = infer.read_image(path); fx, fy = W0 / img.shape[1], H0 / img.shape[0]
    roi = infer.find_roi(img) or (0, 0, img.shape[1], img.shape[0])
    o0, proto, r, px, py = model.forward(img[roi[1]:roi[3], roi[0]:roi[2]])
    b, s, c, k = infer.detect(o0, model.nc, cfg["conf"], cfg["iou"], cfg["max_det"])
    cv, kept = infer.build_instances(b, s, k, proto, r, px, py, roi, fx, fy, W0, H0, cfg["mask_thr"], cfg["min_keep"])
    cv, objs = infer.finalize(cv, kept, c, s, model.names)
    return img, cv, objs, roi


def collect(model, cfg, items):
    """items: list ของ (image path, gt mask path, list class ต่อ gt id, group) → X, y, group"""
    X, y, g = [], [], []
    for path, gtp, gcls, grp in items:
        img, cv, objs, roi = predict_objs(model, path, cfg)
        if not objs: continue
        gt = read_mask(gtp)
        F = infer.bean_features(img, cv, objs, infer.paper_gain(img), roi)
        M = iou_matrix(gt, cv)                                      # [G, P] เรียงตาม id 1..N ของ cv (ต่อเนื่องอยู่แล้ว)
        gid = np.unique(gt); gid = gid[gid > 0]
        r_, c_ = linear_sum_assignment(-M)
        for gi, pi in zip(r_, c_):
            if M[gi, pi] >= 0.5:
                X.append(F[pi]); y.append(CLASSES.index(gcls[int(gid[gi]) - 1])); g.append(grp)
    return np.array(X, np.float32), np.array(y), np.array(g)


def items_for(split, piles, autolabel):
    vs = val_sources(); out = []
    meta = json.load(open(os.path.join(piles, split, "meta.json")))
    for name, m in sorted(meta.items()):
        out.append((os.path.join(piles, split, "images", name + ".jpg"), os.path.join(piles, split, "masks", name + ".png"),
                    m["classes"], name if len(m["classes_pool"]) == 1 else "mixed"))
    for r in csv.DictReader(open(os.path.join(autolabel, "report.csv"))):
        if (r["image"] in vs) == (split == "val"):                  # ภาพจริงความละเอียดเต็ม (ผ่าน decode ย่อเหมือน test)
            out.append((r["src"], os.path.join(autolabel, "masks", r["image"] + ".png"), [r["class"]] * int(r["n"]),
                        "real_" + r["image"]))
    return out


def image_acc(P, y, g):
    """ความแม่นรายภาพ (เฉพาะภาพคลาสเดียว): เฉลี่ยความน่าจะเป็นทุกเม็ด → argmax"""
    ok, n = 0, 0
    for grp in np.unique(g):
        if grp == "mixed": continue
        m = g == grp; ok += int(P[m].mean(0).argmax() == np.bincount(y[m]).argmax()); n += 1
    return ok / max(n, 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="exp/crop1_o2o"); ap.add_argument("--piles", default="data/piles")
    ap.add_argument("--autolabel", default="data/autolabel")
    ap.add_argument("--Cs", type=float, nargs="+", default=[0.03, 0.1, 0.3, 1, 3, 10])
    ap.add_argument("--final", action="store_true", help="หลังเลือก C แล้ว train ใหม่ด้วย train+val ทั้งหมด")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix

    class _A: no_roi = False
    cfg = infer.load_cfg(a.weights, _A()); model = infer.load_model(a.weights, cfg)
    cache = os.path.join(a.weights, "_cls_feats.npz")
    if os.path.isfile(cache):
        d = np.load(cache, allow_pickle=True); Xtr, ytr, gtr, Xva, yva, gva = [d[k] for k in ("Xtr", "ytr", "gtr", "Xva", "yva", "gva")]
    else:
        Xtr, ytr, gtr = collect(model, cfg, items_for("train", a.piles, a.autolabel))
        Xva, yva, gva = collect(model, cfg, items_for("val", a.piles, a.autolabel))
        np.savez(cache, Xtr=Xtr, ytr=ytr, gtr=gtr, Xva=Xva, yva=yva, gva=gva)
    print(f"beans: train {len(ytr)}  val {len(yva)}")

    best = None
    print(f"{'C':>6} {'bean acc':>9} {'img acc(syn)':>13} {'img acc(real)':>14}")
    for C in a.Cs:
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=5000, random_state=a.seed).fit(sc.transform(Xtr), ytr)
        P = clf.predict_proba(sc.transform(Xva)); ba = (P.argmax(1) == yva).mean()
        real = np.char.startswith(gva.astype(str), "real_")
        ia_s, ns = image_acc(P[~real], yva[~real], gva[~real]); ia_r, nr = image_acc(P[real], yva[real], gva[real])
        print(f"{C:6.2f} {ba:9.3f} {ia_s:9.3f} ({ns}) {ia_r:9.3f} ({nr})")
        key = (ia_s + ia_r, ba)
        if best is None or key > best[0]: best = (key, C, sc, clf, P)
    _, C, sc, clf, P = best
    print(f"\nเลือก C={C}\nconfusion (val, ต่อเม็ด) แถว=จริง คอลัมน์=ทาย  {CLASSES}")
    print(confusion_matrix(yva, P.argmax(1), labels=range(len(CLASSES))))
    if a.final:
        X = np.vstack([Xtr, Xva]); y = np.r_[ytr, yva]
        sc = StandardScaler().fit(X); clf = LogisticRegression(C=C, max_iter=5000, random_state=a.seed).fit(sc.transform(X), y)
        print("final: train ด้วย train+val")
    out = {"classes": CLASSES, "features": infer.FEAT_NAMES, "C": C, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
           "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist()}
    json.dump(out, open(os.path.join(a.weights, "cls_head.json"), "w"), indent=1)
    print("saved", os.path.join(a.weights, "cls_head.json"))


if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace"); sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    main()
