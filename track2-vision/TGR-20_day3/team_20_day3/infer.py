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
            "decode_min_side": 1800,
            "refine": True, "color_alpha": 0.5, "color_clip": 6.0, "ws_margin": 4.0}   # ขัดขอบ mask (ดู build_instances)                                  # อ่าน JPEG แบบย่อให้ด้านยาวเหลือ ≥ ค่านี้ (900 → ย่อ 1/4 บนภาพ 4096)


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


_LAB_LUT = None


def lab_lut():
    """สี BGR quantize 5 บิต/ช่อง (32768 สี) → Lab float (คำนวณครั้งเดียว)"""
    global _LAB_LUT
    if _LAB_LUT is None:
        v = np.arange(32, dtype=np.uint8) * 8 + 4
        b, g, r = np.meshgrid(v, v, v, indexing="ij")
        _LAB_LUT = cv2.cvtColor(np.stack([b, g, r], -1).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    return _LAB_LUT


def color_llr(im, fg_small, ring=3):
    """prior สีต่อภาพ: Gaussian ใน Lab ของ "เม็ด" (mask โมเดลหด ring px) กับ "พื้น" (ห่างจาก mask > ring px)
    เรียนจากภาพนี้เอง → log-likelihood ratio ต่อพิกเซลของ im (ผ่าน LUT สี 32^3 → ไม่ต้องคำนวณทีละพิกเซล)"""
    q = (im >> 3).astype(np.uint16)
    idx = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    fgs = cv2.erode(fg_small, k) > 0; bgs = cv2.dilate(fg_small, k) == 0
    if fgs.sum() < 50 or bgs.sum() < 50: return None
    T = lab_lut()

    def loglik(sel):
        w = np.bincount(idx[sel], minlength=32768).astype(np.float64); n = w.sum()
        mu = (w @ T) / n; d = T - mu
        C = (w[:, None] * d).T @ d / n + np.eye(3) * 4.0
        return -0.5 * ((d @ np.linalg.inv(C)) * d).sum(1) - 0.5 * np.log(np.linalg.det(C))
    return (loglik(fgs) - loglik(bgs)).astype(np.float32)[idx]


def build_instances(boxes, scores, coefs, proto, r, padx, pady, roi, fx, fy, W0, H0, mask_thr=0.5, min_keep=0.4,
                    img=None, color_alpha=0.0, color_clip=6.0, ws_margin=0.0):
    """mask ต่อเม็ด → paint ลง canvas ขนาดภาพต้นฉบับ (เฉพาะในกรอบ) → (canvas, list ของ (idx_det, bbox ภาพเต็ม))

    1) logit ต่อเม็ด (proto → ภาพต้นฉบับ เฉพาะในกรอบ) แล้วคัดเม็ดซ้ำ: conf มาก→น้อย, ทิ้งถ้าส่วนที่ไม่ซ้ำ < min_keep
    ขัดขอบ (เมื่อส่ง img = ภาพที่อ่านมา; img=None = แบบเดิม "conf สูงทาสีก่อน"):
    2) color_alpha > 0: บวก prior สีของภาพนี้ (เม็ด vs พื้น) เข้า logit → ขอบเม็ดกับกระดาษตรงขึ้น
    3) พิกเซลที่หลายเม็ดทับ → เม็ดที่ logit สูงสุด
    4) ws_margin > 0: พิกเซลที่ logit อันดับ 1 กับ 2 ห่างกัน < ws_margin → watershed บนภาพตัดสิน (ขอบเม็ดชนเม็ด)"""
    P, ph, pw = proto.shape; st = 4.0                              # proto stride = 4 (S/4)
    canvas = np.zeros((H0, W0), np.uint8 if len(boxes) <= 255 else np.uint16)
    rx0, ry0 = roi[0], roi[1]
    to_x = lambda xl: ((xl - padx) / r + rx0) * fx                 # letterbox px → ภาพต้นฉบับ
    to_y = lambda yl: ((yl - pady) / r + ry0) * fy
    t0 = np.log(mask_thr / (1 - mask_thr))                          # sigmoid(x) > thr
    refine = img is not None
    if refine:                                                      # พื้นที่ทำงาน = ROI ในพิกัดภาพต้นฉบับ (ไม่จองทั้งภาพ)
        X0, Y0 = int(np.floor(rx0 * fx)), int(np.floor(ry0 * fy))
        X1, Y1 = min(int(np.ceil(roi[2] * fx)), W0), min(int(np.ceil(roi[3] * fy)), H0)
        occ = np.zeros((Y1 - Y0, X1 - X0), bool)
    else:
        X0, Y0, X1, Y1 = 0, 0, W0, H0
    pats, kept = [], []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        # patch proto เกินกรอบข้างละ 1 ช่อง → interpolate ที่ขอบกรอบได้ค่าจริง (ไม่ใช่ค่า replicate)
        px0, py0 = max(int(np.floor(x1 / st)) - 1, 0), max(int(np.floor(y1 / st)) - 1, 0)
        px1, py1 = min(int(np.ceil(x2 / st)) + 1, pw), min(int(np.ceil(y2 / st)) + 1, ph)
        if px1 <= px0 or py1 <= py0: continue
        sub = proto[:, py0:py1, px0:px1].reshape(P, -1)
        m = (coefs[i] @ sub).reshape(py1 - py0, px1 - px0)
        # พื้นที่ของ patch proto ในภาพต้นฉบับ
        FX0, FY0 = to_x(px0 * st), to_y(py0 * st); FX1, FY1 = to_x(px1 * st), to_y(py1 * st)
        bx0, by0 = int(max(np.floor(to_x(x1)), X0)), int(max(np.floor(to_y(y1)), Y0))   # กรอบเม็ด (crop mask แบบ ultralytics)
        bx1, by1 = int(min(np.ceil(to_x(x2)), X1)), int(min(np.ceil(to_y(y2)), Y1))
        if bx1 - bx0 < 2 or by1 - by0 < 2: continue
        # resize patch logit → พิกัดภาพเต็มเฉพาะช่วง [bx0,bx1)×[by0,by1) ด้วย affine (ไม่ทำทั้งภาพ)
        sx = (px1 - px0) / max(FX1 - FX0, 1e-6); sy = (py1 - py0) / max(FY1 - FY0, 1e-6)
        M = np.array([[sx, 0, (bx0 + 0.5 - FX0) * sx - 0.5], [0, sy, (by0 + 0.5 - FY0) * sy - 0.5]], np.float32)
        logit = cv2.warpAffine(m.astype(np.float32), M, (bx1 - bx0, by1 - by0),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        mk = logit > t0
        full = int(mk.sum())
        if full < 20: continue
        box = (bx0, by0, bx1, by1)
        if not refine:
            reg = canvas[by0:by1, bx0:bx1]
            free = mk & (reg == 0)                                  # conf สูงกว่า (ทาสีก่อน) ได้พิกเซลไปแล้ว
            if free.sum() < min_keep * full: continue               # ส่วนใหญ่ซ้ำกับเม็ดที่มั่นใจกว่า → ทิ้ง (กันทายเกิน)
            reg[free] = len(kept) + 1; kept.append((i, box)); continue
        reg = occ[by0 - Y0:by1 - Y0, bx0 - X0:bx1 - X0]
        if (mk & ~reg).sum() < min_keep * full: continue            # เกณฑ์คัดเม็ดซ้ำเดียวกับแบบเดิม
        reg |= mk; kept.append((i, box)); pats.append(logit)
    if not refine or not kept: return canvas, kept

    x0, y0, x1, y1 = roi
    if color_alpha > 0:                                             # (2) prior สี (คำนวณที่ความละเอียดภาพที่อ่าน)
        fg_small = cv2.resize(occ.view(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        llr = color_llr(np.ascontiguousarray(img[y0:y1, x0:x1]), fg_small)
        if llr is not None:
            llr = np.clip(llr, -color_clip, color_clip) * color_alpha
            for n, (i, (bx0, by0, bx1, by1)) in enumerate(kept):    # พิกัดภาพต้นฉบับ → พิกัด llr (ภาพที่อ่าน − roi)
                M = np.array([[1 / fx, 0, (bx0 + 0.5) / fx - 0.5 - x0], [0, 1 / fy, (by0 + 0.5) / fy - 0.5 - y0]], np.float32)
                pats[n] += cv2.warpAffine(llr, M, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                          borderMode=cv2.BORDER_REPLICATE)
    # (3) ทาสี: logit สูงสุดชนะ (เริ่มที่ t0 = ต้องผ่าน threshold ด้วย); เก็บ logit อันดับ 2 ไว้หาโซนแย่ง
    best = np.full((Y1 - Y0, X1 - X0), t0, np.float32)
    second = np.full_like(best, -1e9) if ws_margin > 0 else None
    cw = canvas[Y0:Y1, X0:X1]
    for n, (i, (bx0, by0, bx1, by1)) in enumerate(kept):
        lg = pats[n]; sl = (slice(by0 - Y0, by1 - Y0), slice(bx0 - X0, bx1 - X0))
        b = best[sl]; win = lg > b
        if second is not None:
            s2 = second[sl]; s2[win] = b[win]; lose = ~win & (lg > s2); s2[lose] = lg[lose]
        cw[sl][win] = n + 1; b[win] = lg[win]
    # (4) watershed บนภาพที่อ่าน (ภาพต้นฉบับย่อ 1/k) เฉพาะโซนแย่ง
    k = int(round(fx))
    if second is not None and k >= 1 and abs(fx - k) < 1e-3 and abs(fy - k) < 1e-3:
        ox, oy = x0 * k - X0, y0 * k - Y0                           # = 0 เมื่อ k เป็นจำนวนเต็ม
        hh, ww = min((Y1 - Y0 - oy) // k, y1 - y0), min((X1 - X0 - ox) // k, x1 - x0)
        cz = cw[oy:oy + hh * k:k, ox:ox + ww * k:k]
        bz = best[oy:oy + hh * k:k, ox:ox + ww * k:k]; sz = second[oy:oy + hh * k:k, ox:ox + ww * k:k]
        cont = (cz > 0) & (sz > t0) & (bz - sz < ws_margin)
        if cont.any():
            mk = cz.astype(np.int32); mk[cont] = 0
            far = cv2.dilate((cz > 0).view(np.uint8), np.ones((5, 5), np.uint8)) == 0
            mk[far] = 65535                                         # marker พื้น (ห่างจากเม็ด); พื้นติดเม็ด = unknown
            cv2.watershed(np.ascontiguousarray(img[y0:y0 + hh, x0:x0 + ww]), mk)
            ys, xs = np.nonzero(cont & (mk > 0) & (mk < 65535))
            if len(ys):
                V = cw[oy:oy + hh * k, ox:ox + ww * k].reshape(hh, k, ww, k)   # ขยายผลเป็นบล็อก k×k (ไม่ resize ทั้งภาพ)
                blk = V[ys, :, xs, :]                               # (N, k, k) สำเนา
                V[ys, :, xs, :] = np.where(blk > 0, mk[ys, xs].astype(canvas.dtype)[:, None, None], blk)
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


# ───────────── (ค) classifier รายเม็ด: crop ทีละเม็ด (นอก mask = สีเทา) → CNN เล็ก (NCNN) ─────────────
CROP_BASE, CROP_PAD, CROP_FILL = 96, 0.15, 128


def bean_crop(img, mk, ox, oy, gain=None, S=CROP_BASE, pad=CROP_PAD):
    """img: ภาพ BGR (ความละเอียดที่อ่าน) · mk: bool mask ของเม็ดเดียว วางที่ (ox, oy) ใน img
    → (crop S×S uint8 ยังไม่เติมเทา, mask S×S uint8) จัดกึ่งกลางกรอบเม็ด ด้านยาว = กรอบ × (1 + 2·pad)
    ใช้ฟังก์ชันเดียวกันทั้งตอนสร้างข้อมูล train และตอน infer"""
    ys, xs = np.nonzero(mk)
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2; s = S / (max(bx1 - bx0, by1 - by0) * (1 + 2 * pad))
    Mm = np.array([[s, 0, S / 2 - s * cx], [0, s, S / 2 - s * cy]], np.float32)
    Mi = Mm.copy(); Mi[0, 2] -= s * ox; Mi[1, 2] -= s * oy
    crop = cv2.warpAffine(img, Mi, (S, S), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(CROP_FILL,) * 3)
    if gain is not None: crop = np.clip(crop * gain, 0, 255).astype(np.uint8)
    m = cv2.warpAffine(mk.view(np.uint8), Mm, (S, S), flags=cv2.INTER_NEAREST)
    return crop, m


def fill_outside(crop, m, dil=1):
    """พิกเซลนอก mask (ขยาย dil px) = สีเทา → classifier เห็นแค่เม็ดนี้ ไม่เห็นเม็ดข้าง ๆ (ตัดบริบทของกอง)"""
    if dil > 0: m = cv2.dilate(m, np.ones((2 * dil + 1, 2 * dil + 1), np.uint8))
    out = crop.copy(); out[m == 0] = CROP_FILL
    return out


class BeanCls:
    """NCNN classifier รายเม็ด (export ด้วย pnnx; normalize อยู่ในกราฟแล้ว รับ RGB 0–255)"""
    def __init__(self, d, threads=4):
        import ncnn
        self.ncnn = ncnn; meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        self.classes, self.S = meta["classes"], int(meta["imgsz"])
        self.net = ncnn.Net(); self.net.opt.num_threads = threads; self.net.opt.use_vulkan_compute = False
        self.net.load_param(os.path.join(d, "model.ncnn.param")); self.net.load_model(os.path.join(d, "model.ncnn.bin"))

    def proba(self, crops):
        out = []
        for c in crops:
            mat = self.ncnn.Mat.from_pixels_resize(np.ascontiguousarray(c), self.ncnn.Mat.PixelType.PIXEL_BGR2RGB,
                                                   c.shape[1], c.shape[0], self.S, self.S)
            with self.net.create_extractor() as ex:
                ex.input("in0", mat); z = np.array(ex.extract("out0")[1]).reshape(-1)
            e = np.exp(z - z.max()); out.append(e / e.sum())
        return np.array(out)


def classes_from_crops(bc, img, canvas, objs, gain):
    """crop ทุกเม็ดจาก mask ใน canvas (ภาพต้นฉบับ) ที่ความละเอียดของภาพที่อ่าน → ความน่าจะเป็น [N, nc]"""
    H0, W0 = canvas.shape; k = max(1, int(round(W0 / img.shape[1]))); crops = []
    for o in objs:
        x0, y0, x1, y1 = o["_box"]; X0, Y0 = -(-x0 // k) * k, -(-y0 // k) * k        # จุดเริ่มที่หาร k ลงตัว
        mk = canvas[Y0:y1:k, X0:x1:k] == o["id"]
        if not mk.any(): mk = np.ones((1, 1), bool)
        c, m = bean_crop(img, mk, X0 // k, Y0 // k, gain); crops.append(fill_outside(c, m))
    return bc.proba(crops)


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
    bc = getattr(model, "bean_cls", None)
    if bc is not None:                                              # (ค) classifier รายเม็ด
        P = classes_from_crops(bc, img, canvas, objs, paper_gain(img))
        idx = np.full(len(objs), P.mean(0).argmax()) if vote else P.argmax(1)
        for o, i in zip(objs, idx): o["class"] = bc.classes[int(i)]
        return
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


# สีต่อคลาสตามสไลด์ Day 3 หน้า 4 (BGR) · ลำดับ id ของกรรมการ: 0 husk · 1 parchment · 2 immature · 3 good · 4 shell · 5 black · 6 broken
CLASS_COLORS = {"husk": (40, 85, 160), "parchment": (90, 200, 240), "immature": (115, 200, 110), "good": (170, 60, 150),
                "shell": (200, 160, 60), "black": (40, 40, 40), "broken": (60, 60, 220)}
LEGEND_ORDER = ["husk", "parchment", "immature", "good", "shell", "black", "broken"]


def draw_vis(img, canvas, objs, path, size=1024):
    """overlay ขนาดด้านยาว size px (ไว้ดูด้วยตา) — คลาสเดียวกันสีเดียวกันตามสไลด์ + ตารางจำนวนต่อคลาส"""
    h, w = img.shape[:2]; s = min(1.0, size / max(h, w)); tw, th = round(w * s), round(h * s)
    im = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    ids = cv2.resize(canvas, (tw, th), interpolation=cv2.INTER_NEAREST)
    pal = np.zeros((int(ids.max()) + 1 if ids.size else 1, 3), np.uint8)
    for o in objs:
        if o["id"] < len(pal): pal[o["id"]] = CLASS_COLORS.get(o["class"], (200, 200, 200))
    on = ids > 0
    vis = im.copy(); vis[on] = (im[on] * 0.45 + pal[ids[on]] * 0.55).astype(np.uint8)
    edge = np.zeros_like(on)
    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]; edge[1:] |= ids[1:] != ids[:-1]
    edge &= on | np.roll(on, 1, 0) | np.roll(on, 1, 1)
    eid = np.maximum(ids, np.maximum(np.roll(ids, 1, 0), np.roll(ids, 1, 1)))      # ขอบใช้สีคลาสแบบสว่างขึ้น
    vis[edge] = np.clip(pal[eid[edge]].astype(np.int16) + 60, 0, 255).astype(np.uint8)
    cnt = {c: 0 for c in LEGEND_ORDER}
    for o in objs: cnt[o["class"]] = cnt.get(o["class"], 0) + 1
    x0, y0, rh = 10, 10, 26                                                         # ตาราง: สี · id · คลาส · จำนวน
    cv2.rectangle(vis, (x0 - 4, y0 - 4), (x0 + 196, y0 + rh * 8 + 2), (255, 255, 255), -1)
    cv2.putText(vis, f"{len(objs)} beans", (x0 + 2, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    for i, c in enumerate(LEGEND_ORDER):
        y = y0 + rh * (i + 1)
        cv2.rectangle(vis, (x0 + 2, y + 4), (x0 + 20, y + 22), CLASS_COLORS[c], -1)
        cv2.putText(vis, f"{i} {c}", (x0 + 28, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(vis, str(cnt[c]), (x0 + 168, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2 if cnt[c] else 1)
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
                                   cfg["mask_thr"], cfg["min_keep"], img if cfg["refine"] else None,
                                   cfg["color_alpha"], cfg["color_clip"], cfg["ws_margin"])
    canvas, objs = finalize(canvas, kept, cls, scores, model.names)
    name = os.path.basename(path); stem = os.path.splitext(name)[0]
    if write:                                                        # PNG 12 MP เขียนขนานใน thread (cv2 ปล่อย GIL)
        fut = _POOL.submit(write_png, os.path.join(out, "seg_mask", stem + ".png"), canvas)
    assign_classes(model, img, canvas, objs, vote, (crop, roi, fx, fy))
    if write:
        draw_vis(img, canvas, objs, os.path.join(out, "vis", stem + ".jpg"))
        if not fut.result(): raise IOError(f"เขียน seg_mask ของ {name} ไม่ได้")   # รอให้เขียนเสร็จก่อนนับเวลาจบ
    return {"image_name": name, "objects": [{k: v for k, v in o.items() if not k.startswith("_") and k != "conf"} for o in objs]}, canvas


MQTT_CLASSES = ["good", "shell", "black", "husk", "broken", "parchment", "immature"]   # ลำดับ key ตาม PDF Day 3 หน้า 7
DEFECT = set(MQTT_CLASSES) - {"good"}                                # สมมติ: ทุกคลาสยกเว้น good = เม็ดบกพร่อง (PDF ไม่ระบุ)


def mqtt_message(rec, dt):
    """JSON ต่อภาพสำหรับ publish MQTT (PDF หน้า 7) · นับจาก objects เดียวกับ predictions.json/seg_mask → ตรงกันเสมอ"""
    cls = [o["class"] for o in rec["objects"]]
    counts = {c: cls.count(c) for c in MQTT_CLASSES}
    n = sum(counts.values())
    return {"image": rec["image_name"], "latency_ms": int(round(dt * 1000)), "counts": counts,
            "defect_ratio": round(sum(counts[c] for c in DEFECT) / n, 4) if n else 0.0}


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
    m.bean_cls = None
    if cfg.get("bean_cls"):                                          # (ค) classifier รายเม็ด
        bd = os.path.join(wdir, cfg["bean_cls"])
        if not os.path.isfile(os.path.join(bd, "model.ncnn.param")): fail(f"ไม่พบ bean classifier ใน {bd}")
        m.bean_cls = BeanCls(bd, cfg["threads"])
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
    if getattr(args, "no_refine", False): cfg["refine"] = False
    if getattr(args, "cls_model", None): cfg["cls_ncnn"] = args.cls_model
    if getattr(args, "no_cls_model", False): cfg["cls_ncnn"] = None
    if getattr(args, "bean_cls", None): cfg["bean_cls"] = args.bean_cls
    return cfg


def main():
    ap = argparse.ArgumentParser(description="TGR-20 Day 2 coffee bean instance segmentation (NCNN)")
    ap.add_argument("--input", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--vote", action="store_true", help="ใช้ majority vote ของคลาสทั้งภาพ")
    ap.add_argument("--ncnn", help="ชื่อโฟลเดอร์ NCNN ใน weights/ (ค่าเริ่มต้นจาก infer_config.json)")
    ap.add_argument("--conf", type=float); ap.add_argument("--iou", type=float)
    ap.add_argument("--min-keep", dest="min_keep", type=float); ap.add_argument("--mask-thr", dest="mask_thr", type=float)
    ap.add_argument("--cls-model", help="(ก+) โฟลเดอร์ NCNN หลายคลาสใน weights/ ใช้หา class (เช่น cls_ncnn_model_416)")
    ap.add_argument("--no-cls-model", action="store_true", help="ไม่ใช้ cls model แม้ตั้งไว้ใน infer_config.json")
    ap.add_argument("--bean-cls", help="(ค) โฟลเดอร์ NCNN classifier รายเม็ดใน weights/ (เช่น bean_cls_ncnn)")
    ap.add_argument("--decode-min-side", dest="decode_min_side", type=int, help="900 = อ่าน JPEG ย่อ 1/4 (เร็วกว่า)")
    ap.add_argument("--no-refine", action="store_true", help="ปิดการขัดขอบ mask (prior สี + watershed)")
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
    for d in ("seg_mask", "vis", "mqtt"): os.makedirs(os.path.join(a.out, d), exist_ok=True)
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
        msg = mqtt_message(rec, dt)
        with open(os.path.join(a.out, "mqtt", os.path.splitext(rec["image_name"])[0] + ".json"), "w", encoding="utf-8") as fp:
            json.dump(msg, fp, ensure_ascii=False, indent=2)
        print(f"{rec['image_name']}: {len(rec['objects'])} beans  {dt * 1000:.0f} ms  "
              + " ".join(f"{c}={v}" for c, v in msg["counts"].items() if v))
    t0 = time.perf_counter()
    with open(os.path.join(a.out, "predictions.json"), "w", encoding="utf-8") as fp:
        fp.write(json.dumps(preds, ensure_ascii=False, indent=2))   # แนวตั้ง indent 2 แบบ labels.json ของกรรมการ (dumps ทั้งก้อนเร็วกว่า dump แบบ stream)
    tj = time.perf_counter() - t0
    for f, rec in zip(files, preds):                                 # ตรวจผลลัพธ์ขั้นต่ำก่อนจบ
        stem = os.path.splitext(os.path.basename(f))[0]
        if not os.path.isfile(os.path.join(a.out, "seg_mask", stem + ".png")): fail(f"ไม่มี seg_mask ของ {rec['image_name']}")
        if [o["id"] for o in rec["objects"]] != list(range(1, len(rec["objects"]) + 1)): fail(f"id ไม่ต่อเนื่องใน {rec['image_name']}")
        if any(len(o["polygon"]) < 3 for o in rec["objects"]): fail(f"polygon < 3 จุดใน {rec['image_name']}")
        if not os.path.isfile(os.path.join(a.out, "mqtt", stem + ".json")): fail(f"ไม่มี mqtt json ของ {rec['image_name']}")
    print(f"done {len(files)} images  median={np.median(times) * 1000:.0f} ms  mean={np.mean(times) * 1000:.0f} ms  "
          f"max={np.max(times) * 1000:.0f} ms  (json {tj * 1000:.0f} ms)")
    return 0


if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace"); sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    sys.exit(main())
