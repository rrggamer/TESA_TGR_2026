"""ฟังก์ชันร่วมของ Day 2 (คัดลอก/ดัดแปลงจาก Day 1 crop_beans.py) — ใช้ได้ทั้ง train-side และ infer-side (ไม่ import torch)"""
import cv2, numpy as np
from PIL import Image, ImageOps

CLASSES = ["black", "broken", "good", "husk", "immature", "parchment", "shell"]   # = ชื่อโฟลเดอร์เรียงตามตัวอักษร


def load(path):
    """อ่านภาพ + แก้การหมุนจาก EXIF → BGR uint8"""
    return np.array(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))[:, :, ::-1].copy()


def white_balance(img):
    """ปรับให้กระดาษ (พิกเซลสว่าง 40% บน) เป็นสีเทากลาง"""
    g = img.mean(2); bgpx = img[g > np.percentile(g, 60)]
    gain = bgpx.mean() / bgpx.mean(0)
    return np.clip(img * gain, 0, 255).astype(np.uint8)


def bean_fg(img, work=1024):
    """foreground mask ของเมล็ด (ภาพย่อ work px) จาก flat-field + Lab b (Otsu) + dark → (mask uint8 0/255, scale)"""
    s = work / max(img.shape[:2]); small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    k = int(101 * work / 1024) | 1                                  # kernel ตามสเกล (1024 → 101 เหมือน Day 1)
    bg = cv2.medianBlur(small, k).astype(np.float32) + 1
    flat = np.clip(small / bg * 220, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB)
    b = cv2.GaussianBlur(lab[:, :, 2], (5, 5), 0)
    _, m = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (cv2.GaussianBlur(lab[:, :, 0], (5, 5), 0) < 110).astype(np.uint8) * 255
    m = cv2.bitwise_or(m, dark)
    r = max(1, round(work / 1024))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((4 * r + 1, 4 * r + 1), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((6 * r + 1, 6 * r + 1), np.uint8))
    return m, s


def find_beans(img, work=1024):
    """Day 1: เมล็ดที่วางแยกกัน → (list ของ bbox ภาพเต็ม, label map ภาพย่อ, list ของ label id, scale)"""
    m, s = bean_fg(img, work)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m)
    a = st[1:, cv2.CC_STAT_AREA]; med = np.median(a[a > 100]) if (a > 100).any() else 0
    keep = [i for i in range(1, n) if 0.3 * med < st[i, cv2.CC_STAT_AREA] < 3 * med]
    return [(st[i, :4] / s).astype(int) for i in keep], lbl, keep, s


def find_roi(img, work=512, margin=0.05, min_area_frac=1.5e-4):
    """กรอบรวมของเมล็ด/กองทั้งหมดในภาพ (ใช้ crop ก่อนเข้าโมเดล) → (x0, y0, x1, y1) พิกัดภาพเต็ม หรือ None

    ใช้สีเทียบกับกระดาษแบบ global (ไม่ใช้ flat-field แบบ median-blur เพราะในกองใหญ่ median ภายในกองจะเป็นสีเมล็ดเอง)
    เร็ว: ทำงานบนภาพย่อ work px (~5 ms)
    """
    H, W = img.shape[:2]; s = work / max(H, W)
    sm = cv2.resize(img, (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.int16)
    L, b = lab[..., 0], lab[..., 2]
    paper = L >= np.percentile(L, 70)                               # กระดาษ = ส่วนสว่าง (กองไม่น่าเกิน 30% ของภาพ)
    Lp, bp = np.median(L[paper]), np.median(b[paper])
    Lbg = cv2.blur(np.where(paper, L, Lp).astype(np.float32), (61, 61))   # แสงตกไม่สม่ำเสมอ: ระดับความสว่างกระดาษเฉพาะที่
    fg = (((b - bp) > 8) | (L < 0.62 * Lbg)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(fg)
    h, w = fg.shape; boxes = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < min_area_frac * h * w: continue                       # ฝุ่น / noise
        touch = x == 0 or y == 0 or x + bw == w or y + bh == h
        if touch and (max(bw, bh) / max(min(bw, bh), 1) > 4 or a / (bw * bh) < 0.3): continue   # ขอบโต๊ะ/กระดาษ
        boxes.append((x, y, x + bw, y + bh))
    if not boxes: return None
    bx = np.array(boxes); x0, y0 = bx[:, :2].min(0); x1, y1 = bx[:, 2:].max(0)
    m = margin * max(H, W)
    return (int(max(0, x0 / s - m)), int(max(0, y0 / s - m)), int(min(W, x1 / s + m)), int(min(H, y1 / s + m)))
