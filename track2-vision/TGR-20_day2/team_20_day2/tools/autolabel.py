"""ขั้นที่ 2: auto-label instance mask ของเมล็ดในภาพ train (เมล็ดวางแยกกัน) ด้วย FastSAM-s + ตัวกรองแบบ classical

ขั้นตอนต่อภาพ
  1) FastSAM-s ทั้งภาพ (ย่อเหลือด้านยาว --work px, imgsz=1024, retina mask) → mask ผู้สมัคร
  2) กรอง: ขนาด (0.3–3 เท่าของ median เมล็ดจาก classical), aspect ≤ 3.5, ไม่แตะขอบภาพ,
     สี = ≥60% ของพิกเซลต้องอยู่ใน foreground ของ Lab-b/dark mask (ตัดพื้น เงา ขอบโต๊ะ ป้าย)
     mask ใหญ่เกิน 3 เท่า หรือสี่เหลี่ยมจัด (rectangularity>0.85 และใหญ่) = ป้าย/ขอบกระดาษ → ทิ้ง
  3) ลบ mask ซ้อน (ชิ้นส่วนของเมล็ดเดียวกัน) แล้วจับคู่ 1:1 กับ connected component ของ classical (Hungarian บน IoU)
     คู่ IoU ≥ --min-iou → ใช้ mask FastSAM (ขอบแม่นกว่า), classical ที่ไม่มีคู่ → ใช้ mask classical (fallback, แจ้งเตือน)
  4) class ของทุกเมล็ด = ชื่อโฟลเดอร์

ผลลัพธ์ (--out)
  labels/<stem>.txt   YOLO-seg: "cls x1 y1 x2 y2 ..." (normalize 0–1)
  masks/<stem>.png    instance mask ขนาดเต็มภาพ (0=พื้น, 1..N) — ใช้เป็น GT ของ score.py และตัด bean bank
  vis/<stem>.jpg      overlay ไว้ตรวจด้วยตา
  report.csv          จำนวนเมล็ด / จำนวน fallback / IoU กับ classical ต่อภาพ

python autolabel.py --src D:/TGR20/Track02/day_1 --out data/autolabel
"""
import argparse, csv, glob, os, random
import cv2, numpy as np
from scipy.optimize import linear_sum_assignment
from common import CLASSES, load, white_balance, find_beans, bean_fg


def largest_cc_filled(m):
    """เก็บเฉพาะชิ้นใหญ่สุด + อุดรู"""
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    if n <= 1: return m.astype(bool)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    one = (lbl == k).astype(np.uint8)
    cnts, _ = cv2.findContours(one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(one); cv2.drawContours(out, cnts, -1, 1, -1)
    return out.astype(bool)


def refine(m, fg):
    """ตัดส่วนที่ล้นออกนอก foreground สี (เงา/มุมกล่อง) แล้วเกลาขอบให้เรียบ"""
    r = np.logical_and(m, fg).astype(np.uint8)
    r = cv2.morphologyEx(r, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    r = cv2.GaussianBlur(r.astype(np.float32), (0, 0), 2.0) > 0.5
    return largest_cc_filled(r) if r.sum() > 0.5 * m.sum() else largest_cc_filled(m)


def iou(a, b):
    i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()
    return i / u if u else 0.0


def process(model, path, cls, a):
    img = load(path); H, W = img.shape[:2]
    wb = white_balance(img)
    s = a.work / max(H, W); h, w = round(H * s), round(W * s)
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    # ---- classical: foreground + component ต่อเมล็ด (ย้ายมาที่ความละเอียด small) ----
    _, lbl, keep, _ = find_beans(wb)
    lbl = cv2.resize(lbl.astype(np.int32).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    cls_masks = [lbl == k for k in keep]
    fg = np.isin(lbl, keep) if keep else np.zeros((h, w), bool)
    fg_d = cv2.dilate(fg.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    fine, _ = bean_fg(wb, work=max(w, h))                        # foreground ละเอียดที่ความละเอียดเดียวกับ small
    fine = cv2.resize(fine, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    fine_d = cv2.dilate(fine.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    A0 = np.median([m.sum() for m in cls_masks]) if cls_masks else 1.0

    # ---- FastSAM ----
    r = model(small, imgsz=1024, conf=a.conf, iou=0.7, retina_masks=True, device=a.device, verbose=False)[0]
    cand = []
    stats = {"fsam_raw": 0, "rej_size": 0, "rej_big": 0, "rej_aspect": 0, "rej_border": 0, "rej_color": 0}
    if r.masks is not None:
        ms = r.masks.data.cpu().numpy() > 0.5; sc = r.boxes.conf.cpu().numpy()
        stats["fsam_raw"] = len(ms)
        for m, c in zip(ms, sc):
            m = largest_cc_filled(m); ar = m.sum()
            if ar > 3 * A0:
                stats["rej_big"] += 1; continue                    # ป้าย / ขอบกระดาษ / พื้นทั้งแผ่น
            if ar < 0.3 * A0:
                stats["rej_size"] += 1; continue                   # เศษ / ฝุ่น / ชิ้นส่วนของเมล็ด
            ys, xs = np.nonzero(m)
            if ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1:
                stats["rej_border"] += 1; continue
            cnt = np.stack([xs, ys], 1).astype(np.float32)
            (_, (rw, rh), _) = cv2.minAreaRect(cnt)
            if max(rw, rh) / max(min(rw, rh), 1) > 3.5:
                stats["rej_aspect"] += 1; continue
            if fg_d[m].mean() < 0.6:
                stats["rej_color"] += 1; continue                  # ไม่ใช่สีเมล็ด (เงา / พื้น / ป้ายขาว)
            cand.append((float(c), m))
    # ลบซ้อน: เรียงตาม conf, ทิ้งตัวที่ทับตัวที่เก็บแล้ว > 50% ของพื้นที่ตัวเล็ก
    cand.sort(key=lambda t: -t[0]); kept = []
    for c, m in cand:
        if all(np.logical_and(m, k).sum() / min(m.sum(), k.sum()) < 0.5 for _, k in kept):
            kept.append((c, m))
    stats["fsam_kept"] = len(kept)

    # ---- จับคู่กับ classical ----
    final, ious = [], []
    nf, nc = len(kept), len(cls_masks)
    if nf and nc:
        M = np.array([[iou(k, cm) for cm in cls_masks] for _, k in kept])
        ri, ci = linear_sum_assignment(-M)
        pair = {j: i for i, j in zip(ri, ci) if M[i, j] >= a.min_iou}
    else:
        M, pair = None, {}
    n_fb = 0
    for j, cm in enumerate(cls_masks):
        if j in pair:
            final.append(refine(kept[pair[j]][1], fine_d)); ious.append(M[pair[j], j])
        else:
            final.append(largest_cc_filled(cm)); n_fb += 1          # fallback: FastSAM หาเมล็ดนี้ไม่เจอ
    stats["fsam_extra"] = nf - len(pair)                          # FastSAM เจอแต่ classical ไม่เจอ → ทิ้ง (ตรวจใน vis)
    stats["fallback"] = n_fb
    stats["n"] = len(final)
    stats["iou_med"] = float(np.median(ious)) if ious else 0.0
    stats["iou_min"] = float(np.min(ious)) if ious else 0.0

    # ---- เขียนผล ----
    stem = os.path.splitext(os.path.basename(path))[0]; ci_ = CLASSES.index(cls)
    lines = []
    inst = np.zeros((H, W), np.uint8)
    for i, m in enumerate(final, 1):
        full = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
        inst[full] = i
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 0.7, True).reshape(-1, 2).astype(np.float64)   # ลดจุดเฉพาะ label train (≤0.7px ที่ small)
        c[:, 0] /= w; c[:, 1] /= h
        lines.append(f"{ci_} " + " ".join(f"{v:.6f}" for v in c.clip(0, 1).ravel()))
    with open(os.path.join(a.out, "labels", stem + ".txt"), "w") as fp: fp.write("\n".join(lines) + "\n")
    cv2.imwrite(os.path.join(a.out, "masks", stem + ".png"), inst)

    vis = small.copy(); rng = np.random.default_rng(0)
    for i, m in enumerate(final):
        col = rng.integers(60, 255, 3).tolist()
        vis[m] = (0.5 * vis[m] + 0.5 * np.array(col)).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 1)
    for j, cm in enumerate(cls_masks):                            # fallback = กรอบแดง
        if j not in pair:
            ys, xs = np.nonzero(cm); cv2.rectangle(vis, (xs.min() - 4, ys.min() - 4), (xs.max() + 4, ys.max() + 4), (0, 0, 255), 2)
    cv2.putText(vis, f"{stem} n={len(final)} fb={n_fb}", (10, 40), 0, 1.2, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(a.out, "vis", stem + ".jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return stem, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/TGR20/Track02/day_1")
    ap.add_argument("--out", default="data/autolabel")
    ap.add_argument("--model", default="FastSAM-s.pt")
    ap.add_argument("--work", type=int, default=2048, help="ย่อด้านยาวก่อนเข้า FastSAM / ความละเอียดของ label")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-iou", type=float, default=0.5, help="IoU ขั้นต่ำกับ classical เพื่อยอมรับ mask FastSAM")
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    for d in ("labels", "masks", "vis"): os.makedirs(os.path.join(a.out, d), exist_ok=True)
    from ultralytics import FastSAM
    model = FastSAM(a.model)
    rows = []
    for cls in CLASSES:
        for path in sorted(glob.glob(os.path.join(a.src, cls, "*.jpg"))):
            stem, st = process(model, path, cls, a)
            rows.append({"image": stem, "class": cls, "src": os.path.abspath(path), **st})
            print(f"{stem:22s} n={st['n']:2d} fsam_raw={st['fsam_raw']:3d} kept={st['fsam_kept']:2d} "
                  f"fallback={st['fallback']} extra={st['fsam_extra']} IoU(med/min)={st['iou_med']:.3f}/{st['iou_min']:.3f} "
                  f"rej[size={st['rej_size']} big={st['rej_big']} border={st['rej_border']} aspect={st['rej_aspect']} color={st['rej_color']}]")
    with open(os.path.join(a.out, "report.csv"), "w", newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print("total beans:", sum(r["n"] for r in rows), " fallback:", sum(r["fallback"] for r in rows))


if __name__ == "__main__":
    main()
