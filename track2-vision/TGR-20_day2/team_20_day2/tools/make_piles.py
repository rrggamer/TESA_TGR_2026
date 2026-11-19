"""ขั้นที่ 3: สังเคราะห์ภาพกองเมล็ด (ชิด + ซ้อนกัน) จาก bean bank ที่ได้จาก autolabel

1) bean bank: ตัดเมล็ดทุกเม็ดจากภาพจริงความละเอียดเต็ม เป็น RGB + alpha (ขอบ feather 1 px)
2) พื้นหลัง: ภาพจริงทั้งภาพที่ลบเมล็ด+เงาออกด้วย inpaint (เก็บแสงตกไม่สม่ำเสมอ/ขอบโต๊ะจริงไว้) ย่อเหลือครึ่ง
3) วางเมล็ดทีละเม็ด (เม็ดหลังอยู่บน) ในรัศมี R ที่คำนวณจาก packing → ชิด/ซ้อนกันจริง
   - สุ่ม rotate 0–360, flip, scale ±15% ต่อเม็ด (+ global ±8%), color jitter เล็กน้อย, เงาใต้เมล็ด
   - id map ระดับพิกเซล → เม็ดใหม่ทับเม็ดเก่าอัตโนมัติ (mask_old &= ~mask_new)
   - ไม่ยอมวางตำแหน่งที่ทำให้เม็ดเดิมเหลือส่วนที่มองเห็น < --min-vis หรือแตกเป็นหลายชิ้น
     (= ไม่มีเศษเมล็ดที่ไม่มี label ค้างอยู่ในภาพ) หาที่ใหม่ไม่ได้ → ข้ามเม็ดนั้น
4) แบ่ง train/val ตาม source image (ภาพต้นฉบับ) ทั้ง bean bank และพื้นหลัง → ไม่มีเมล็ดเดียวกันรั่วข้ามฝั่ง

ผลลัพธ์ (--out, ค่าเริ่มต้น data/piles)
  {train,val}/images/*.jpg   ภาพกอง 1536x2048
  {train,val}/masks/*.png    instance id (0=พื้น, 1..N) = GT ของ score.py
  {train,val}/meta.json      class ของแต่ละ id, จำนวนเม็ด, source
  real_val/                  ภาพจริงของ source ฝั่ง val (ย่อครึ่ง) + mask → ตรวจว่าไม่พังบนภาพจริง
  ../datasets/{full,crop}/   YOLO-seg dataset + beans.yaml (full = ทั้งภาพ, crop = ตัดด้วย find_roi แบบเดียวกับ infer)

python make_piles.py --n-train 400 --n-val 80
"""
import argparse, csv, json, os, shutil, time
import cv2, numpy as np
from common import CLASSES, load, find_roi

LAYOUTS = ["circle", "grid", "square"]
SCALE = 0.5                                   # ภาพสังเคราะห์ = ครึ่งหนึ่งของภาพจริง (4096 → 2048)


# ───────────────────────── bean bank / background ─────────────────────────
def val_sources():
    """source ฝั่ง val: คลาสละ 1 ภาพ หมุนเวียน layout (black→circle, broken→grid, good→square, ...)"""
    return {f"{LAYOUTS[i % 3]}_{c}" for i, c in enumerate(CLASSES)}


def build_bank_and_bg(src_root, al_dir):
    bank, bgs, reals = [], {}, {}
    with open(os.path.join(al_dir, "report.csv")) as fp:
        rows = list(csv.DictReader(fp))
    for r in rows:
        stem, cls = r["image"], r["class"]
        img = load(r["src"]); inst = cv2.imread(os.path.join(al_dir, "masks", stem + ".png"), cv2.IMREAD_UNCHANGED)
        H, W = inst.shape
        for k in range(1, int(inst.max()) + 1):
            ys, xs = np.nonzero(inst == k); p = 8
            y0, y1, x0, x1 = max(ys.min() - p, 0), min(ys.max() + p + 1, H), max(xs.min() - p, 0), min(xs.max() + p + 1, W)
            m = (inst[y0:y1, x0:x1] == k).astype(np.uint8)
            # กัน halo ขาวรอบเมล็ด: ขอบนุ่มอยู่ "ด้านใน" mask และเติมสีขอบด้วยสีเมล็ดเอง (ไม่ใช่สีกระดาษ)
            core = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            rgb = cv2.inpaint(img[y0:y1, x0:x1], (1 - core).astype(np.uint8), 5, cv2.INPAINT_TELEA)
            a = cv2.GaussianBlur(cv2.erode(m, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), 1.0)
            bank.append({"cls": cls, "src": stem, "rgb": rgb.astype(np.float32), "a": a,
                         "area": float((inst == k).sum())})
        # พื้นหลัง: ลบเมล็ด + เงา (dilate กว้าง) ด้วย inpaint ที่ 1/8 แล้วเติม noise ให้เหมือนกระดาษ
        h, w = round(H * SCALE), round(W * SCALE)
        half = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        hole = cv2.resize((inst > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        hole = cv2.dilate(hole, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71)))
        q = 4; sm = cv2.resize(half, (w // q, h // q), interpolation=cv2.INTER_AREA)
        hs = cv2.resize(hole, (w // q, h // q), interpolation=cv2.INTER_NEAREST)
        fill = cv2.resize(cv2.inpaint(sm, hs, 9, cv2.INPAINT_TELEA), (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        hp = half.astype(np.float32) - cv2.GaussianBlur(half.astype(np.float32), (0, 0), 3)
        sigma = float(hp[hole == 0].std())
        fill += np.random.default_rng(len(bgs)).normal(0, sigma, fill.shape)
        soft = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 6)[..., None]
        bgs[stem] = np.clip(half * (1 - soft) + fill * soft, 0, 255).astype(np.uint8)
        reals[stem] = (half, cv2.resize(inst, (w, h), interpolation=cv2.INTER_NEAREST), cls)
        print(f"  bank {stem}: {int(inst.max())} beans")
    return bank, bgs, reals


# ───────────────────────── compose ─────────────────────────
def transform(bean, rng, gscale):
    s = SCALE * gscale * rng.uniform(0.85, 1.15)
    sx, sy = s, s
    if rng.random() < 0.3: sy *= rng.uniform(0.78, 1.0)                  # เมล็ดเอียงในกอง → ดูสั้นลงด้านหนึ่ง
    rgb = cv2.resize(bean["rgb"], None, fx=sx, fy=sy, interpolation=cv2.INTER_AREA)
    a = cv2.resize(bean["a"], (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5: rgb, a = rgb[:, ::-1], a[:, ::-1]
    h, w = a.shape; M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(0, 360), 1)
    c, si = abs(M[0, 0]), abs(M[0, 1]); W2, H2 = int(h * si + w * c) + 2, int(h * c + w * si) + 2
    M[0, 2] += W2 / 2 - w / 2; M[1, 2] += H2 / 2 - h / 2
    rgb = cv2.warpAffine(np.ascontiguousarray(rgb), M, (W2, H2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a = cv2.warpAffine(np.ascontiguousarray(a), M, (W2, H2), flags=cv2.INTER_LINEAR, borderValue=0)
    gain = rng.uniform(0.9, 1.1) * rng.uniform(0.97, 1.03, 3)            # color jitter เล็กน้อย (สีเป็นข้อมูลของคลาส)
    mean = rgb.reshape(-1, 3).mean(0)
    rgb = (rgb - mean) * rng.uniform(0.92, 1.08) + mean                   # contrast
    return np.clip(rgb * gain, 0, 255), a


class Canvas:
    def __init__(self, bg, min_vis):
        self.img = bg.astype(np.float32); H, W = bg.shape[:2]
        self.ids = np.zeros((H, W), np.int32); self.min_vis = min_vis
        self.area, self.vis, self.box, self.cls = [0], [0], [None], [None]   # index 0 = พื้น

    def try_place(self, rgb, a, cx, cy, cls, shadow):
        H, W = self.ids.shape; h, w = a.shape
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        if x0 < 2 or y0 < 2 or x0 + w > W - 2 or y0 + h > H - 2: return False
        m = a > 0.5; ar = int(m.sum())
        if ar < 50: return False
        reg = self.ids[y0:y0 + h, x0:x0 + w]
        cnt = np.bincount(reg[m], minlength=len(self.area))
        for k in np.nonzero(cnt[1:])[0] + 1:                              # เม็ดเดิมที่ถูกทับ
            left = self.vis[k] - cnt[k]
            if left < self.min_vis * self.area[k]: return False
            bx0, by0, bx1, by1 = self.box[k]
            sub = self.ids[by0:by1, bx0:bx1] == k
            ox0, oy0 = max(x0, bx0), max(y0, by0); ox1, oy1 = min(x0 + w, bx1), min(y0 + h, by1)
            sub = sub.copy(); sub[oy0 - by0:oy1 - by0, ox0 - bx0:ox1 - bx0] &= ~m[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0]
            n, _, st, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8))
            if n > 2 and st[1:, cv2.CC_STAT_AREA].max() < 0.97 * left: return False   # แตกเป็นหลายชิ้น
        # เงา: alpha เลื่อน + blur ทำให้พื้น/เม็ดข้างใต้มืดลง
        dx, dy, sig, k_ = shadow; P = int(3 * sig + max(abs(dx), abs(dy))) + 1
        sy0, sx0 = max(y0 - P, 0), max(x0 - P, 0); sy1, sx1 = min(y0 + h + P, H), min(x0 + w + P, W)
        sh = np.zeros((sy1 - sy0, sx1 - sx0), np.float32)
        ty, tx = y0 - sy0 + dy, x0 - sx0 + dx
        ty0, tx0 = max(ty, 0), max(tx, 0); ty1, tx1 = min(ty + h, sh.shape[0]), min(tx + w, sh.shape[1])
        sh[ty0:ty1, tx0:tx1] = a[ty0 - ty:ty1 - ty, tx0 - tx:tx1 - tx]
        sh = cv2.GaussianBlur(sh, (0, 0), sig)
        self.img[sy0:sy1, sx0:sx1] *= (1 - k_ * sh)[..., None]
        roi = self.img[y0:y0 + h, x0:x0 + w]
        self.img[y0:y0 + h, x0:x0 + w] = roi * (1 - a[..., None]) + rgb * a[..., None]
        for k in np.nonzero(cnt[1:])[0] + 1: self.vis[k] -= int(cnt[k])
        nid = len(self.area); reg[m] = nid
        self.area.append(ar); self.vis.append(ar); self.box.append((x0, y0, x0 + w, y0 + h)); self.cls.append(cls)
        return True

    def finish(self, rng):
        img = self.img + rng.normal(0, 1.2, self.img.shape)
        if rng.random() < 0.3: img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 0.9))
        keep = [k for k in range(1, len(self.area)) if self.vis[k] >= self.min_vis * self.area[k]]
        lut = np.zeros(len(self.area), np.int32)
        for i, k in enumerate(keep, 1): lut[k] = i
        return np.clip(img, 0, 255).astype(np.uint8), lut[self.ids], [self.cls[k] for k in keep]


def make_pile(bank, bgs, rng, a):
    bg_stem = list(bgs)[rng.integers(len(bgs))]; bg = bgs[bg_stem]
    if rng.random() < 0.5: bg = bg[:, ::-1]
    if rng.random() < 0.5: bg = bg[::-1]
    bg = np.clip(bg.astype(np.float32) * rng.uniform(0.93, 1.05), 0, 255)
    H, W = bg.shape[:2]
    cv_ = Canvas(bg, a.min_vis)
    by_cls = {}
    for b in bank: by_cls.setdefault(b["cls"], []).append(b)
    if rng.random() < a.mixed:                                             # หลายคลาส (เตรียม Day 3)
        pool_cls = list(rng.choice(sorted(by_cls), rng.integers(2, 5), replace=False))
    else:
        pool_cls = [sorted(by_cls)[rng.integers(len(by_cls))]]
    pool = [b for c in pool_cls for b in by_cls[c]]
    n_target = int(rng.integers(a.min_beans, a.max_beans + 1))
    gscale = rng.uniform(0.92, 1.08)
    mean_area = np.mean([b["area"] for b in pool]) * (SCALE * gscale) ** 2
    packing = rng.uniform(0.75, 1.9)                                       # >1 = ซ้อนกัน
    R = np.sqrt(n_target * mean_area / (np.pi * packing))
    ell = rng.uniform(0.7, 1.4); th = rng.uniform(0, np.pi)
    mg = R * max(ell, 1 / ell) + 120
    cx0, cy0 = rng.uniform(mg, W - mg) if W > 2 * mg else W / 2, rng.uniform(mg, H - mg) if H > 2 * mg else H / 2
    shadow = (int(rng.integers(-4, 5)), int(rng.integers(1, 7)), rng.uniform(3, 7), rng.uniform(0.15, 0.4))
    placed = 0
    for _ in range(n_target):
        b = pool[rng.integers(len(pool))]; rgb, al = transform(b, rng, gscale)
        for _try in range(25):
            r = R * np.sqrt(rng.random()); t = rng.uniform(0, 2 * np.pi)
            u, v = r * np.cos(t) * ell, r * np.sin(t) / ell
            cx, cy = cx0 + u * np.cos(th) - v * np.sin(th), cy0 + u * np.sin(th) + v * np.cos(th)
            if cv_.try_place(rgb, al, cx, cy, b["cls"], shadow): placed += 1; break
    if rng.random() < a.p_scatter:                                         # เมล็ดกระจายรอบกอง
        for _ in range(int(rng.integers(1, 7))):
            b = pool[rng.integers(len(pool))]; rgb, al = transform(b, rng, gscale)
            for _try in range(10):
                r = R * rng.uniform(1.3, 2.6); t = rng.uniform(0, 2 * np.pi)
                if cv_.try_place(rgb, al, cx0 + r * np.cos(t), cy0 + r * np.sin(t), b["cls"], shadow): break
    img, ids, classes = cv_.finish(rng)
    return img, ids, classes, {"bg": bg_stem, "packing": round(packing, 2), "target": n_target, "classes_pool": pool_cls}


# ───────────────────────── export YOLO ─────────────────────────
def mask_to_lines(ids, classes, ox=0, oy=0, w=None, h=None):
    """instance map → บรรทัด YOLO-seg (normalize ด้วยขนาดภาพ/crop)"""
    H, W = ids.shape; w = w or W; h = h or H; lines = []
    for i, c in enumerate(classes, 1):
        m = (ids == i).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: continue
        cn = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 0.7, True).reshape(-1, 2).astype(np.float64)
        if len(cn) < 3: continue
        cn[:, 0] = (cn[:, 0] - ox) / w; cn[:, 1] = (cn[:, 1] - oy) / h
        lines.append(f"{CLASSES.index(c)} " + " ".join(f"{v:.6f}" for v in cn.clip(0, 1).ravel()))
    return lines


def export(name, img, ids, classes, root, split):
    """เขียนลง datasets/full และ datasets/crop"""
    H, W = ids.shape
    for kind in ("full", "crop"):
        d = os.path.join(root, kind)
        os.makedirs(os.path.join(d, "images", split), exist_ok=True); os.makedirs(os.path.join(d, "labels", split), exist_ok=True)
        if kind == "full":
            im, lines = img, mask_to_lines(ids, classes)
        else:
            r = find_roi(img) or (0, 0, W, H); x0, y0, x1, y1 = r
            sub = ids[y0:y1, x0:x1]
            lost = set(np.unique(ids)) - set(np.unique(sub))
            if lost - {0}: print(f"    WARN {name}: ROI ตัดเม็ด {sorted(lost - {0})} ทิ้ง")
            im = img[y0:y1, x0:x1]
            # ids ใน crop ยังอ้าง class เดิม: ทำ label จาก sub โดยตรง
            lines = mask_to_lines(sub, classes)
        cv2.imwrite(os.path.join(d, "images", split, name + ".jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(d, "labels", split, name + ".txt"), "w") as fp: fp.write("\n".join(lines) + ("\n" if lines else ""))


def write_yaml(root):
    for kind in ("full", "crop"):
        d = os.path.abspath(os.path.join(root, kind)).replace("\\", "/")
        with open(os.path.join(root, kind, "beans.yaml"), "w") as fp:
            fp.write(f"path: {d}\ntrain: images/train\nval: images/val\nnames:\n")
            for i, c in enumerate(CLASSES): fp.write(f"  {i}: {c}\n")


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="D:/TGR20/Track02/day_1")
    ap.add_argument("--autolabel", default="data/autolabel")
    ap.add_argument("--out", default="data/piles")
    ap.add_argument("--datasets", default="datasets")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-val", type=int, default=80)
    ap.add_argument("--min-beans", type=int, default=30)
    ap.add_argument("--max-beans", type=int, default=120)
    ap.add_argument("--min-vis", type=float, default=0.3, help="สัดส่วนที่มองเห็นขั้นต่ำของเม็ดที่ถูกทับ")
    ap.add_argument("--mixed", type=float, default=0.2, help="สัดส่วนภาพที่มีหลายคลาส")
    ap.add_argument("--p-scatter", type=float, default=0.3)
    ap.add_argument("--no-real", action="store_true", help="ไม่ใส่ภาพจริง (เมล็ดแยกกัน) ฝั่ง train ลงใน dataset")
    ap.add_argument("--all-sources", action="store_true", help="ใช้ทุก source เป็น train (สำหรับโมเดลส่งจริงหลังเลือก config แล้ว)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.random.seed(a.seed)
    for d in (a.out, a.datasets):
        if os.path.isdir(d): shutil.rmtree(d)
    print("building bean bank + backgrounds ...")
    bank, bgs, reals = build_bank_and_bg(a.src, a.autolabel)
    vs = set() if a.all_sources else val_sources()
    split_bank = {"train": [b for b in bank if b["src"] not in vs], "val": [b for b in bank if b["src"] in vs]}
    split_bg = {"train": {k: v for k, v in bgs.items() if k not in vs}, "val": {k: v for k, v in bgs.items() if k in vs}}
    if a.all_sources: split_bank["val"], split_bg["val"] = bank, bgs        # val ไม่อิสระในโหมดนี้ (ใช้แค่ early stop)
    print(f"val sources: {sorted(vs)}")
    print({s: (len(split_bank[s]), len(split_bg[s])) for s in ("train", "val")}, "(beans, backgrounds)")
    assert not ({b["src"] for b in split_bank["train"]} & {b["src"] for b in split_bank["val"]}) or a.all_sources

    for split, n in (("train", a.n_train), ("val", a.n_val)):
        for d in ("images", "masks"): os.makedirs(os.path.join(a.out, split, d), exist_ok=True)
        meta, t0 = {}, time.time()
        for i in range(n):
            rng = np.random.default_rng([a.seed, 0 if split == "train" else 1, i])
            img, ids, classes, info = make_pile(split_bank[split], split_bg[split], rng, a)
            name = f"pile_{split}_{i:04d}"
            cv2.imwrite(os.path.join(a.out, split, "images", name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(85, 96))])
            cv2.imwrite(os.path.join(a.out, split, "masks", name + ".png"), ids.astype(np.uint16 if len(classes) > 255 else np.uint8))
            img = cv2.imread(os.path.join(a.out, split, "images", name + ".jpg"))   # ใช้ภาพหลัง JPEG เหมือนกันทุกที่
            export(name, img, ids, classes, a.datasets, split)
            meta[name] = {"n": len(classes), "classes": classes, **info}
            if (i + 1) % 50 == 0 or i == n - 1:
                ns = [m["n"] for m in meta.values()]
                print(f"  {split} {i + 1}/{n}  beans/img mean={np.mean(ns):.1f} min={min(ns)} max={max(ns)}  {time.time() - t0:.0f}s")
        with open(os.path.join(a.out, split, "meta.json"), "w") as fp: json.dump(meta, fp, indent=1)

    # ภาพจริง: ฝั่ง train ใส่ใน dataset (ความหลากหลายของภาพถ่ายจริง), ฝั่ง val เก็บไว้ตรวจใน real_val/
    os.makedirs(os.path.join(a.out, "real_val", "images"), exist_ok=True); os.makedirs(os.path.join(a.out, "real_val", "masks"), exist_ok=True)
    for stem, (half, inst, cls) in reals.items():
        classes = [cls] * int(inst.max())
        if stem in vs:
            cv2.imwrite(os.path.join(a.out, "real_val", "images", stem + ".jpg"), half, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(a.out, "real_val", "masks", stem + ".png"), inst)
        elif not a.no_real:
            export("real_" + stem, half, inst, classes, a.datasets, "train")
    write_yaml(a.datasets)
    print("done")


if __name__ == "__main__":
    main()
