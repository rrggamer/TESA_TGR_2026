"""TGR-20 Day 2 — instance segmentation เมล็ดกาแฟ (NCNN, CPU, ไม่ใช้ torch)

python infer.py --input test_folder/ --weights weights/ --out outputs/

ต่อภาพ: อ่าน JPEG แบบย่อ (libjpeg DCT scaling) → หา ROI ของกอง → letterbox สี่เหลี่ยม → NCNN YOLO-seg
        → NMS → mask ต่อเม็ด upsample เฉพาะในกรอบของเม็ด (ความละเอียดภาพต้นฉบับ) → conf สูงได้พิกเซลก่อน
        → seg_mask/<stem>.png, vis/<stem>.jpg, predictions.json (polygon จาก mask เดียวกัน CHAIN_APPROX_NONE)
"""
import argparse, json, os, sys, time
import cv2, numpy as np

EXTS = (".jpg", ".jpeg", ".png")                                   # ตามสเปก (ไม่สนตัวพิมพ์)
from concurrent.futures import ThreadPoolExecutor
_POOL = ThreadPoolExecutor(1)                                        # เขียน seg_mask ขนานกับงานที่เหลือของภาพเดียวกัน
DEFAULTS = {"ncnn": "best_ncnn_model", "cls_ncnn": None, "conf": 0.25, "iou": 0.6, "min_keep": 0.4, "mask_thr": 0.5,
            "roi": True, "max_det": 400, "threads": 4,
            "decode_min_side": 1800}                                  # อ่าน JPEG แบบย่อให้ด้านยาวเหลือ ≥ ค่านี้ (900 → ย่อ 1/4 บนภาพ 4096)


def fail(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr); sys.exit(code)


# ───────────── ROI (เหมือน common.find_roi — คัดลอกมาให้ infer.py ทำงานได้ด้วยตัวเอง) ─────────────
def find_roi(img, work=512, margin=0.05, min_area_frac=1.5e-4):
    H, W = img.shape[:2]; s = work / max(H, W)
    sm = cv2.resize(img, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, b = lab[..., 0], lab[..., 2]
    paper = L >= np.percentile(L, 70)
    Lp, bp = np.median(L[paper]), np.median(b[paper])
    Lbg = cv2.blur(np.where(paper, L, Lp).astype(np.float32), (61, 61))
    fg = (((b - bp) > 8) | (L < 0.62 * Lbg)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(fg)
    h, w = fg.shape; boxes = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < min_area_frac * h * w: continue
        touch = x == 0 or y == 0 or x + bw == w or y + bh == h
        if touch and (max(bw, bh) / max(min(bw, bh), 1) > 4 or a / (bw * bh) < 0.3): continue
        boxes.append((x, y, x + bw, y + bh))
    if not boxes: return None
    bx = np.array(boxes); x0, y0 = bx[:, :2].min(0); x1, y1 = bx[:, 2:].max(0)
    m = margin * max(H, W)
    return (int(max(0, x0 / s - m)), int(max(0, y0 / s - m)), int(min(W, x1 / s + m)), int(min(H, y1 / s + m)))


# ───────────── image I/O ─────────────
def orig_size(path):
    """ขนาดภาพต้นฉบับหลังหมุนตาม EXIF (อ่านแค่ header)"""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        try: o = im.getexif().get(274, 1)
        except Exception: o = 1
    return (w, h) if o not in (5, 6, 7, 8) else (h, w)


def read_image(path, min_side=1800):
    """อ่านภาพแบบย่อ 1/2 หรือ 1/4 ถ้าภาพใหญ่ (เร็วกว่าอ่านเต็มหลายเท่าบน Pi) → (img, W0, H0)"""
    W0, H0 = orig_size(path); flag = cv2.IMREAD_COLOR
    if path.lower().endswith((".jpg", ".jpeg")):
        if max(W0, H0) >= 4 * min_side: flag = cv2.IMREAD_REDUCED_COLOR_4
        elif max(W0, H0) >= 2 * min_side: flag = cv2.IMREAD_REDUCED_COLOR_2
    img = cv2.imread(path, flag)
    if img is None: raise IOError(f"เปิดภาพไม่ได้: {path}")
    return img, W0, H0


# ───────────── model ─────────────
class SegModel:
    def __init__(self, ncnn_dir, threads=4):
        import ncnn, yaml
        self.ncnn = ncnn
        meta = yaml.safe_load(open(os.path.join(ncnn_dir, "metadata.yaml"), encoding="utf-8"))
        self.names = [meta["names"][k] for k in sorted(meta["names"])]
        self.S = int(meta["imgsz"][0]); self.nc = len(self.names)
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads; self.net.opt.use_vulkan_compute = False
        self.net.opt.use_fp16_storage = True; self.net.opt.use_fp16_arithmetic = False
        self.net.load_param(os.path.join(ncnn_dir, "model.ncnn.param"))
        self.net.load_model(os.path.join(ncnn_dir, "model.ncnn.bin"))

    def forward(self, crop):
        """crop BGR → (out0 [4+nc+32, A], proto [32, S/4, S/4], r, padx, pady)"""
        nc_ = self.ncnn; S = self.S; h, w = crop.shape[:2]
        r = S / max(h, w); nw, nh = max(1, round(w * r)), max(1, round(h * r))
        padx, pady = (S - nw) // 2, (S - nh) // 2
        mat = nc_.Mat.from_pixels_resize(np.ascontiguousarray(crop), nc_.Mat.PixelType.PIXEL_BGR2RGB, w, h, nw, nh)
        mat = nc_.copy_make_border(mat, pady, S - nh - pady, padx, S - nw - padx, nc_.BorderType.BORDER_CONSTANT, 114.0)
        mat.substract_mean_normalize([], [1 / 255.0] * 3)
        with self.net.create_extractor() as ex:
            ex.input("in0", mat)
            o0 = np.array(ex.extract("out0")[1]); o1 = np.array(ex.extract("out1")[1])
        return o0, o1, r, padx, pady


def detect(o0, nc, conf, iou, max_det):
    """→ boxes xyxy (letterbox px), scores, cls, coefs — เรียง score มาก→น้อย (NMS ไม่คิดคลาส)"""
    sc_all = o0[4:4 + nc]; cls = sc_all.argmax(0); sc = sc_all[cls, np.arange(sc_all.shape[1])]
    k = np.nonzero(sc > conf)[0]
    if len(k) == 0: return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros((0, 32))
    cx, cy, bw, bh = o0[0, k], o0[1, k], o0[2, k], o0[3, k]
    xywh = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1)
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), sc[k].tolist(), conf, iou, top_k=max_det)
    keep = np.array(keep, int).reshape(-1)
    keep = keep[np.argsort(-sc[k][keep])]
    k = k[keep]; xywh = xywh[keep]
    boxes = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], 1)
    return boxes, sc[k], cls[k], o0[4 + nc:4 + nc + 32, k].T


def build_instances(boxes, scores, coefs, proto, r, padx, pady, roi, fx, fy, W0, H0, mask_thr=0.5, min_keep=0.4):
    """mask ต่อเม็ด → paint ลง canvas ขนาดภาพต้นฉบับ (เฉพาะในกรอบ) → (canvas, list ของ (idx_det, bbox ภาพเต็ม))"""
    P, ph, pw = proto.shape; st = 4.0                              # proto stride = 4 (S/4)
    canvas = np.zeros((H0, W0), np.uint8 if len(boxes) <= 255 else np.uint16)
    rx0, ry0 = roi[0], roi[1]
    to_x = lambda xl: ((xl - padx) / r + rx0) * fx                 # letterbox px → ภาพต้นฉบับ
    to_y = lambda yl: ((yl - pady) / r + ry0) * fy
    proto_flat = proto.reshape(P, -1)
    kept = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        px0, py0 = max(int(np.floor(x1 / st)), 0), max(int(np.floor(y1 / st)), 0)
        px1, py1 = min(int(np.ceil(x2 / st)), pw), min(int(np.ceil(y2 / st)), ph)
        if px1 <= px0 or py1 <= py0: continue
        sub = proto[:, py0:py1, px0:px1].reshape(P, -1)
        m = (coefs[i] @ sub).reshape(py1 - py0, px1 - px0)
        # พื้นที่ของ patch proto ในภาพต้นฉบับ
        FX0, FY0 = to_x(px0 * st), to_y(py0 * st); FX1, FY1 = to_x(px1 * st), to_y(py1 * st)
        bx0, by0 = int(max(np.floor(to_x(x1)), 0)), int(max(np.floor(to_y(y1)), 0))     # กรอบเม็ด (crop mask แบบ ultralytics)
        bx1, by1 = int(min(np.ceil(to_x(x2)), W0)), int(min(np.ceil(to_y(y2)), H0))
        if bx1 - bx0 < 2 or by1 - by0 < 2: continue
        # resize patch logit → พิกัดภาพเต็มเฉพาะช่วง [bx0,bx1)×[by0,by1) ด้วย affine (ไม่ทำทั้งภาพ)
        sx = (px1 - px0) / max(FX1 - FX0, 1e-6); sy = (py1 - py0) / max(FY1 - FY0, 1e-6)
        M = np.array([[sx, 0, (bx0 + 0.5 - FX0) * sx - 0.5], [0, sy, (by0 + 0.5 - FY0) * sy - 0.5]], np.float32)
        logit = cv2.warpAffine(m.astype(np.float32), M, (bx1 - bx0, by1 - by0),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        mk = logit > np.log(mask_thr / (1 - mask_thr))              # sigmoid(x) > thr
        full = int(mk.sum())
        if full < 20: continue
        reg = canvas[by0:by1, bx0:bx1]
        free = mk & (reg == 0)                                      # conf สูงกว่า (ทาสีก่อน) ได้พิกเซลไปแล้ว
        if free.sum() < min_keep * full: continue                   # ส่วนใหญ่ซ้ำกับเม็ดที่มั่นใจกว่า → ทิ้ง (กันทายเกิน)
        nid = len(kept) + 1; reg[free] = nid
        kept.append((i, (bx0, by0, bx1, by1)))
    return canvas, kept


def finalize(canvas, kept, cls, scores, names):
    """เก็บชิ้นใหญ่สุดต่อ id, ทำ polygon จาก mask เดียวกัน, เรียง id ใหม่ 1..N → (canvas, objects)"""
    objs = []
    labels = [names[int(cls[i])] for i, _ in kept]
    for j, (i, (x0, y0, x1, y1)) in enumerate(kept, 1):
        reg = canvas[y0:y1, x0:x1]; m = (reg == j).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts: continue
        if len(cnts) > 1:                                            # หลายชิ้น → เก็บชิ้นใหญ่สุด ให้ mask กับ polygon ตรงกัน
            n, lb, stt, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            big = 1 + int(np.argmax(stt[1:, cv2.CC_STAT_AREA])); reg[(lb != big) & (m > 0)] = 0; m = (lb == big).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=len).reshape(-1, 2)
        if len(c) < 3: reg[m > 0] = 0; continue
        nid = len(objs) + 1
        if nid != j: reg[m > 0] = nid                                # เลื่อน id ให้ต่อเนื่อง (เฉพาะในกรอบ ไม่แตะทั้งภาพ)
        fill = np.zeros_like(m); cv2.drawContours(fill, [c.reshape(-1, 1, 2)], -1, 1, -1)
        reg[(fill > 0) & (reg == 0)] = nid                           # อุดรูภายใน (พิกเซลพื้น) ให้ mask ตรงกับ polygon
        objs.append({"id": nid, "class": labels[j - 1], "conf": round(float(scores[i]), 4),
                     "polygon": (c + [x0, y0]).tolist(), "_box": (x0, y0, x1, y1)})
    dt = np.uint8 if len(objs) <= 255 else np.uint16
    return canvas.astype(dt), objs


# ───────────── class ต่อเม็ด (สำหรับโมเดล 1 คลาส): สี + รูปทรง 12 ค่า (ชุดเดียวกับ Day 1) → LogReg ใน numpy ─────────────
FEAT_NAMES = ["L_mean", "a_mean", "b_mean", "L_std", "a_std", "b_std", "L_p10", "L_p90",
              "area", "aspect", "solidity", "circularity"]


def paper_gain(img):
    """white balance: gain ต่อช่องที่ทำให้กระดาษ (พิกเซลสว่าง 40% บน) เป็นสีเทา — คำนวณจากภาพย่อ 256 px"""
    s = 256 / max(img.shape[:2]); t = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    g = t.mean(1); bg = t[g > np.percentile(g, 60)]
    return bg.mean() / np.maximum(bg.mean(0), 1)


def bean_features(img, canvas, objs, gain, roi=None, sub=2):
    """feature 12 ค่าต่อเม็ด (vectorized): สีจากภาพที่อ่านมา (ย่อ) ภายใน mask, sampling ทุก sub พิกเซล ในกรอบ roi
    รูปทรงจาก polygon ภาพเต็ม"""
    n = len(objs)
    if n == 0: return np.zeros((0, len(FEAT_NAMES)), np.float32)
    H0, W0 = canvas.shape; h, w = img.shape[:2]; k = max(1, int(round(W0 / w)))
    x0, y0, x1, y1 = roi if roi is not None else (0, 0, w, h)
    ids = canvas[y0 * k:y1 * k:k * sub, x0 * k:x1 * k:k * sub]
    im = img[y0:y1:sub, x0:x1:sub][:ids.shape[0], :ids.shape[1]]; ids = ids[:im.shape[0], :im.shape[1]]
    sel = ids > 0; lid = ids[sel].astype(np.int64)
    px = np.clip(im[sel].astype(np.float32) * gain, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    cnt = np.bincount(lid, minlength=n + 1)[1:].astype(np.float64); c1 = np.maximum(cnt, 1)
    mean = np.stack([np.bincount(lid, lab[:, i], n + 1)[1:] for i in range(3)], 1) / c1[:, None]
    sq = np.stack([np.bincount(lid, lab[:, i] ** 2, n + 1)[1:] for i in range(3)], 1) / c1[:, None]
    std = np.sqrt(np.maximum(sq - mean ** 2, 0))
    key = np.sort(lid * 256 + lab[:, 0].astype(np.int64)); Ls = key % 256         # เรียงตาม (id, L) ครั้งเดียว
    st = np.r_[0, np.cumsum(cnt)[:-1]].astype(np.int64); cm1 = np.maximum(cnt - 1, 0)
    p10 = np.where(cnt > 0, Ls[np.minimum(st + (0.1 * cm1).astype(np.int64), len(Ls) - 1)] if len(Ls) else 0, 0)
    p90 = np.where(cnt > 0, Ls[np.minimum(st + (0.9 * cm1).astype(np.int64), len(Ls) - 1)] if len(Ls) else 0, 0)
    shp = []
    for o in objs:
        c = np.asarray(o["polygon"], np.int32).reshape(-1, 1, 2)
        A = cv2.contourArea(c); P = cv2.arcLength(c, True); hull = cv2.contourArea(cv2.convexHull(c))
        (_, (rw, rh), _) = cv2.minAreaRect(c)
        shp.append([A / (W0 * H0) * 1e4, max(rw, rh) / max(min(rw, rh), 1), A / max(hull, 1), 4 * np.pi * A / max(P * P, 1)])
    return np.hstack([mean, std, p10[:, None], p90[:, None], np.array(shp)]).astype(np.float32)


class ClsHead:
    """StandardScaler + LogisticRegression (multinomial) ที่ export เป็น JSON → ใช้ numpy ล้วน"""
    def __init__(self, path):
        d = json.load(open(path, encoding="utf-8"))
        self.classes = d["classes"]; self.mean = np.array(d["mean"]); self.scale = np.array(d["scale"])
        self.coef = np.array(d["coef"]); self.intercept = np.array(d["intercept"])

    def proba(self, X):
        z = ((X - self.mean) / self.scale) @ self.coef.T + self.intercept
        z -= z.max(1, keepdims=True); e = np.exp(z)
        return e / e.sum(1, keepdims=True)


def classes_from_model(cls_model, crop, roi, fx, fy, objs, conf=0.2):
    """(ก+) รันโมเดลหลายคลาสอีกตัว (imgsz เล็ก) บน crop เดียวกัน → class ของกล่องที่ IoU กับกรอบเม็ดสูงสุด
    คืน array [N, nc] (one-hot ถ่วงด้วย score; แถวศูนย์ = ไม่มีกล่องที่ทับ)"""
    o0, _, r, px, py = cls_model.forward(crop)
    b, s, c, _ = detect(o0, cls_model.nc, conf, 0.7, 400)
    V = np.zeros((len(objs), cls_model.nc))
    if not len(b): return V
    bx = np.stack([((b[:, 0] - px) / r + roi[0]) * fx, ((b[:, 1] - py) / r + roi[1]) * fy,
                   ((b[:, 2] - px) / r + roi[0]) * fx, ((b[:, 3] - py) / r + roi[1]) * fy], 1)
    ob = np.array([o["_box"] for o in objs], np.float64)
    ix = np.clip(np.minimum(ob[:, None, 2], bx[None, :, 2]) - np.maximum(ob[:, None, 0], bx[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(ob[:, None, 3], bx[None, :, 3]) - np.maximum(ob[:, None, 1], bx[None, :, 1]), 0, None)
    inter = ix * iy; ua = (ob[:, 2] - ob[:, 0]) * (ob[:, 3] - ob[:, 1])
    iou = inter / (ua[:, None] + ((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]))[None] - inter + 1e-9)
    j = iou.argmax(1); ok = iou[np.arange(len(objs)), j] > 0.3
    V[np.nonzero(ok)[0], c[j[ok]]] = s[j[ok]]
    return V


def assign_classes(model, img, canvas, objs, vote=False, crop_info=None):
    """โมเดล 1 คลาส + cls_head → class จาก feature สี/รูปทรง; โมเดลหลายคลาส → class จากโมเดล
    vote: ทุกเม็ดในภาพได้คลาสเดียวกัน (เฉลี่ยความน่าจะเป็น / majority)"""
    if not objs: return
    head = getattr(model, "cls_head", None); cm = getattr(model, "cls_model", None)
    if cm is not None and crop_info is not None:                   # (ก+) class จากโมเดลหลายคลาส
        V = classes_from_model(cm, *crop_info, objs)
        if head is not None and model.nc == 1:                     # เม็ดที่ไม่มีกล่องทับ → ใช้ LogReg สำรอง
            miss = V.sum(1) == 0
            if miss.any(): V[miss] = head.proba(bean_features(img, canvas, objs, paper_gain(img), crop_info[1]))[miss]
        idx = np.full(len(objs), V.sum(0).argmax()) if vote else V.argmax(1)
        names = cm.names
        for o, i in zip(objs, idx): o["class"] = names[int(i)]
        return
    if head is not None and model.nc == 1:
        P = head.proba(bean_features(img, canvas, objs, paper_gain(img), crop_info[1] if crop_info else None))
        idx = np.full(len(objs), P.mean(0).argmax()) if vote else P.argmax(1)
        for o, i in zip(objs, idx): o["class"] = head.classes[int(i)]
    elif vote:
        u, c = np.unique([o["class"] for o in objs], return_counts=True)
        for o in objs: o["class"] = str(u[c.argmax()])


def draw_vis(img, canvas, objs, path, size=1024):
    """overlay ขนาดด้านยาว size px (ไว้ดูด้วยตา) — ย่อก่อนวาดเพื่อประหยัดเวลา"""
    h, w = img.shape[:2]; s = min(1.0, size / max(h, w)); tw, th = round(w * s), round(h * s)
    im = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    ids = cv2.resize(canvas, (tw, th), interpolation=cv2.INTER_NEAREST)
    pal = np.random.default_rng(0).integers(40, 255, (len(objs) + 1, 3)).astype(np.uint8); pal[0] = 0
    on = ids > 0
    vis = im.copy(); vis[on] = (im[on] * 0.45 + pal[ids[on]] * 0.55).astype(np.uint8)
    edge = np.zeros_like(on)
    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]; edge[1:] |= ids[1:] != ids[:-1]
    vis[edge] = (255, 255, 255)
    cv2.putText(vis, f"{len(objs)} beans", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(path, vis, [cv2.IMWRITE_JPEG_QUALITY, 85])


def process(model, path, cfg, out, vote=False, write=True):
    img, W0, H0 = read_image(path, cfg.get("decode_min_side", 1800))
    fx, fy = W0 / img.shape[1], H0 / img.shape[0]
    roi = find_roi(img) if cfg["roi"] else None
    if roi is None or (roi[2] - roi[0]) < 32 or (roi[3] - roi[1]) < 32: roi = (0, 0, img.shape[1], img.shape[0])
    crop = img[roi[1]:roi[3], roi[0]:roi[2]]
    o0, proto, r, padx, pady = model.forward(crop)
    boxes, scores, cls, coefs = detect(o0, model.nc, cfg["conf"], cfg["iou"], cfg["max_det"])
    canvas, kept = build_instances(boxes, scores, coefs, proto, r, padx, pady, roi, fx, fy, W0, H0,
                                   cfg["mask_thr"], cfg["min_keep"])
    canvas, objs = finalize(canvas, kept, cls, scores, model.names)
    name = os.path.basename(path); stem = os.path.splitext(name)[0]
    if write:                                                        # PNG 12 MP เขียนขนานใน thread (cv2 ปล่อย GIL)
        fut = _POOL.submit(write_png, os.path.join(out, "seg_mask", stem + ".png"), canvas)
    assign_classes(model, img, canvas, objs, vote, (crop, roi, fx, fy))
    if write:
        draw_vis(img, canvas, objs, os.path.join(out, "vis", stem + ".jpg"))
        if not fut.result(): raise IOError(f"เขียน seg_mask ของ {name} ไม่ได้")   # รอให้เขียนเสร็จก่อนนับเวลาจบ
    return {"image_name": name, "objects": [{k: v for k, v in o.items() if not k.startswith("_") and k != "conf"} for o in objs]}, canvas


PNG_SIG = bytes([137, 80, 78, 71, 13, 10, 26, 10])


def write_png(path, a):
    """PNG แบบ filter=None + zlib RLE level 1 (เร็วกว่า cv2.imwrite ~3 เท่าสำหรับ mask ที่เป็นพื้น 0 เกือบทั้งภาพ)
    uint16 (เม็ด > 255) → ใช้ cv2"""
    if a.dtype != np.uint8:
        return cv2.imwrite(path, a, [cv2.IMWRITE_PNG_COMPRESSION, 1, cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_RLE])
    import zlib, struct
    h, w = a.shape; raw = np.zeros((h, w + 1), np.uint8); raw[:, 1:] = a          # ไบต์แรกของทุกแถว = filter 0
    co = zlib.compressobj(1, zlib.DEFLATED, 15, 9, zlib.Z_RLE); data = co.compress(raw) + co.flush()
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(PNG_SIG + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                + chunk(b"IDAT", data) + chunk(b"IEND", b""))
    return True


def list_images(inp):
    if os.path.isfile(inp): return [inp]
    if not os.path.isdir(inp): fail(f"ไม่พบ input: {inp}")
    fs = sorted(os.path.join(inp, f) for f in os.listdir(inp) if f.lower().endswith(EXTS))
    if not fs: fail(f"ไม่พบภาพใน {inp}")
    return fs


def load_model(wdir, cfg):
    """โหลด NCNN + cls_head.json (ถ้ามี)"""
    m = SegModel(os.path.join(wdir, cfg["ncnn"]), cfg["threads"])
    hp = os.path.join(wdir, "cls_head.json")
    m.cls_head = ClsHead(hp) if os.path.isfile(hp) else None
    m.cls_model = None
    if cfg.get("cls_ncnn"):                                          # (ก+) โมเดลหลายคลาสสำหรับ class
        cd = os.path.join(wdir, cfg["cls_ncnn"])
        if not os.path.isfile(os.path.join(cd, "model.ncnn.param")): fail(f"ไม่พบ cls model ใน {cd}")
        m.cls_model = SegModel(cd, cfg["threads"])
        m.cls_model.forward(np.full((m.cls_model.S, m.cls_model.S, 3), 114, np.uint8))   # warm-up
    if m.nc == 1 and m.cls_head is None:
        print("WARNING: โมเดล 1 คลาสแต่ไม่พบ cls_head.json → class = 'bean'", file=sys.stderr)
    return m


def load_cfg(wdir, args):
    cfg = dict(DEFAULTS); p = os.path.join(wdir, "infer_config.json")
    if os.path.isfile(p): cfg.update(json.load(open(p, encoding="utf-8")))
    for k in ("ncnn", "conf", "iou", "min_keep", "mask_thr", "threads", "decode_min_side"):
        v = getattr(args, k, None)
        if v is not None: cfg[k] = v
    if getattr(args, "no_roi", False): cfg["roi"] = False
    if getattr(args, "cls_model", None): cfg["cls_ncnn"] = args.cls_model
    if getattr(args, "no_cls_model", False): cfg["cls_ncnn"] = None
    return cfg


def format_predictions(preds, per_line=4):
    """JSON หลายบรรทัดตามตัวอย่างในสเปก: 1 object = 2 บรรทัด, polygon ตัดบรรทัดทุก per_line จุด"""
    dumps = lambda v: json.dumps(v, ensure_ascii=False, separators=(", ", ": "))
    items = []
    for e in preds:
        if not e["objects"]:
            items.append("  {" + f'"image_name": {dumps(e["image_name"])}, "objects": []' + "}")
            continue
        objs = []
        for o in e["objects"]:
            pts = [dumps(p) for p in o["polygon"]]
            rows = [", ".join(pts[i:i + per_line]) for i in range(0, len(pts), per_line)]
            objs.append("     {" + f'"id": {o["id"]}, "class": {dumps(o["class"])},\n' + " " * 6
                        + '"polygon": [' + (",\n" + " " * 18).join(rows) + "]}")
        items.append("  {" + f'"image_name": {dumps(e["image_name"])},\n' + '   "objects": [\n'
                     + ",\n".join(objs) + "\n   ]}")
    return "[\n" + ",\n".join(items) + "\n]\n"


def main():
    ap = argparse.ArgumentParser(description="TGR-20 Day 2 coffee bean instance segmentation (NCNN)")
    ap.add_argument("--input", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--vote", action="store_true", help="ใช้ majority vote ของคลาสทั้งภาพ")
    ap.add_argument("--ncnn", help="ชื่อโฟลเดอร์ NCNN ใน weights/ (ค่าเริ่มต้นจาก infer_config.json)")
    ap.add_argument("--conf", type=float); ap.add_argument("--iou", type=float)
    ap.add_argument("--min-keep", dest="min_keep", type=float); ap.add_argument("--mask-thr", dest="mask_thr", type=float)
    ap.add_argument("--cls-model", help="(ก+) โฟลเดอร์ NCNN หลายคลาสใน weights/ ใช้หา class (เช่น cls_ncnn_model_416)")
    ap.add_argument("--no-cls-model", action="store_true", help="ไม่ใช้ cls model แม้ตั้งไว้ใน infer_config.json")
    ap.add_argument("--decode-min-side", dest="decode_min_side", type=int, help="900 = อ่าน JPEG ย่อ 1/4 (เร็วกว่า)")
    ap.add_argument("--threads", type=int); ap.add_argument("--no-roi", action="store_true", help="ไม่ crop ROI (ส่งทั้งภาพ)")
    a = ap.parse_args()

    files = list_images(a.input)
    stems = [os.path.splitext(os.path.basename(f))[0].lower() for f in files]
    if len(set(stems)) != len(stems): fail("ชื่อไฟล์ input (ไม่รวมนามสกุล) ซ้ำกัน → seg_mask จะเขียนทับกัน")
    in_dir = os.path.abspath(a.input if os.path.isdir(a.input) else os.path.dirname(a.input) or ".")
    out_dir = os.path.abspath(a.out)
    if out_dir == in_dir or out_dir.startswith(in_dir + os.sep): fail("--out ต้องแยกจากโฟลเดอร์ input")
    from PIL import Image
    for f in files:                                                  # ตรวจว่าเปิดได้ทุกภาพก่อนเริ่ม (ไม่ข้ามภาพเงียบ ๆ)
        try:
            with Image.open(f) as im: im.verify()
        except Exception as e:
            fail(f"เปิดภาพ {os.path.basename(f)} ไม่ได้: {e}")
    if not os.path.isdir(a.weights): fail(f"ไม่พบโฟลเดอร์ weights: {a.weights}")
    cfg = load_cfg(a.weights, a)
    nd = os.path.join(a.weights, cfg["ncnn"])
    if not os.path.isfile(os.path.join(nd, "model.ncnn.param")): fail(f"ไม่พบ NCNN model ใน {nd}")
    for d in ("seg_mask", "vis"): os.makedirs(os.path.join(a.out, d), exist_ok=True)
    cv2.setNumThreads(cfg["threads"])

    model = load_model(a.weights, cfg)
    model.forward(np.full((model.S, model.S, 3), 114, np.uint8))       # warm-up (ไม่นับเวลา)
    print(f"model {nd} imgsz={model.S} classes={model.cls_head.classes if model.cls_head else model.names} cfg={ {k: cfg[k] for k in ('conf', 'iou', 'min_keep', 'roi')} }")

    preds, times = [], []
    for f in files:
        t0 = time.perf_counter()
        try:
            rec, _ = process(model, f, cfg, a.out, a.vote)
        except Exception as e:
            fail(f"ประมวลผล {os.path.basename(f)} ไม่ได้: {e}")
        dt = time.perf_counter() - t0; times.append(dt); preds.append(rec)
        print(f"{rec['image_name']}: {len(rec['objects'])} beans  {dt * 1000:.0f} ms")
    t0 = time.perf_counter()
    with open(os.path.join(a.out, "predictions.json"), "w", encoding="utf-8") as fp:
        fp.write(format_predictions(preds))                                      # รูปแบบบรรทัดตามสเปก (ยังเป็น JSON ที่ถูกต้อง)
    tj = time.perf_counter() - t0
    for f, rec in zip(files, preds):                                 # ตรวจผลลัพธ์ขั้นต่ำก่อนจบ
        stem = os.path.splitext(os.path.basename(f))[0]
        if not os.path.isfile(os.path.join(a.out, "seg_mask", stem + ".png")): fail(f"ไม่มี seg_mask ของ {rec['image_name']}")
        if [o["id"] for o in rec["objects"]] != list(range(1, len(rec["objects"]) + 1)): fail(f"id ไม่ต่อเนื่องใน {rec['image_name']}")
        if any(len(o["polygon"]) < 3 for o in rec["objects"]): fail(f"polygon < 3 จุดใน {rec['image_name']}")
    print(f"done {len(files)} images  median={np.median(times) * 1000:.0f} ms  mean={np.mean(times) * 1000:.0f} ms  "
          f"max={np.max(times) * 1000:.0f} ms  (json {tj * 1000:.0f} ms)")
    return 0


if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace"); sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    sys.exit(main())