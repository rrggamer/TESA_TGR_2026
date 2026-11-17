import argparse, glob, os, cv2, numpy as np
from PIL import Image, ImageOps

def load(path):                                   # แก้การหมุนจาก EXIF ของกล้องมือถือ
    return np.array(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))[:, :, ::-1].copy()

def white_balance(img):                           # ปรับให้กระดาษเป็นสีเทากลาง (ใช้พิกเซลสว่างสุดเป็นพื้นหลัง)
    g = img.mean(2); bgpx = img[g > np.percentile(g, 60)]
    gain = bgpx.mean() / bgpx.mean(0)
    return np.clip(img * gain, 0, 255).astype(np.uint8)

def find_beans(img, work=1024):
    s = work / max(img.shape[:2]); small = cv2.resize(img, None, fx=s, fy=s)
    bg = cv2.medianBlur(small, 101).astype(np.float32) + 1          # flat-field ลบแสงไม่สม่ำเสมอ
    flat = np.clip(small / bg * 220, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB)
    b = cv2.GaussianBlur(lab[:, :, 2], (5, 5), 0)                    # ช่อง b: เมล็ดเหลือง/น้ำตาล, กระดาษ+เงา = กลาง
    _, m = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = (cv2.GaussianBlur(lab[:, :, 0], (5, 5), 0) < 110).astype(np.uint8) * 255  # เมล็ด black: สีไม่เหลืองแต่มืดมาก
    m = cv2.bitwise_or(m, dark)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m)
    a = st[1:, cv2.CC_STAT_AREA]; med = np.median(a[a > 100]) if (a > 100).any() else 0
    keep = [i for i in range(1, n) if 0.3 * med < st[i, cv2.CC_STAT_AREA] < 3 * med]  # ตัดฝุ่น/ขอบโต๊ะ
    mask = cv2.resize(np.isin(lbl, keep).astype(np.uint8) * 255, img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    return [(st[i, :4] / s).astype(int) for i in keep], mask

def square_crop(img, mask, box, pad=0.15, size=160, bg_val=200):
    x, y, w, h = box; c = max(w, h) * (1 + 2 * pad); cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(cx - c / 2), int(cy - c / 2); x1, y1 = int(x0 + c), int(y0 + c)
    P = int(c)                                                      # pad ขอบภาพกันหลุดเฟรม
    im = cv2.copyMakeBorder(img, P, P, P, P, cv2.BORDER_REPLICATE)[y0 + P:y1 + P, x0 + P:x1 + P]
    mk = cv2.copyMakeBorder(mask, P, P, P, P, cv2.BORDER_CONSTANT, 0)[y0 + P:y1 + P, x0 + P:x1 + P]
    mk = cv2.GaussianBlur(cv2.dilate(mk, np.ones((5, 5), np.uint8)), (7, 7), 0)[..., None] / 255.0
    im = (im * mk + bg_val * (1 - mk)).astype(np.uint8)            # แทนพื้นหลังด้วยสีเรียบ กันโมเดลจำลายกระดาษ/เงา
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)

FEAT_NAMES = ["L_mean", "a_mean", "b_mean", "L_std", "a_std", "b_std", "L_p10", "L_p90",
              "area", "aspect", "solidity", "circularity"]

def bean_features(img, mask, box):
    """feature สี + รูปทรงของเมล็ด 1 เม็ด (คำนวณจากภาพความละเอียดเต็มหลัง white balance)"""
    x, y, w, h = box
    mk = (mask[y:y + h, x:x + w] > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return [0.0] * len(FEAT_NAMES)
    c = max(cnts, key=cv2.contourArea)
    m1 = np.zeros_like(mk); cv2.drawContours(m1, [c], -1, 1, -1)
    lab = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2LAB)[m1 > 0].astype(np.float32)
    A = cv2.contourArea(c); P = cv2.arcLength(c, True); hull = cv2.contourArea(cv2.convexHull(c))
    (_, (rw, rh), _) = cv2.minAreaRect(c)
    return [*lab.mean(0), *lab.std(0), np.percentile(lab[:, 0], 10), np.percentile(lab[:, 0], 90),
            A / (img.shape[0] * img.shape[1]) * 1e4,          # ขนาดเทียบกับภาพ (กล้อง/ระยะเดิม)
            max(rw, rh) / max(min(rw, rh), 1),                 # ยาว/กว้าง
            A / max(hull, 1),                                  # solidity: ต่ำ = เว้า/แหว่ง (shell, broken)
            4 * np.pi * A / max(P * P, 1)]                     # circularity

def beans_from_image(path, size=160):
    """คืน list ของ (crop BGR size×size, feature 12 ค่า) ทีละเมล็ด"""
    img = white_balance(load(path)); boxes, mask = find_beans(img)
    return [(square_crop(img, mask, b, size=size), bean_features(img, mask, b)) for b in boxes]

def crops_from_image(path, size=160):
    return [c for c, _ in beans_from_image(path, size)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--src"); ap.add_argument("--dst"); ap.add_argument("--size", type=int, default=160)
    a = ap.parse_args()
    rows = []
    for path in sorted(glob.glob(os.path.join(a.src, "*", "*.*"))):
        cls = os.path.basename(os.path.dirname(path)); stem = os.path.splitext(os.path.basename(path))[0]
        os.makedirs(os.path.join(a.dst, cls), exist_ok=True)
        beans = beans_from_image(path, a.size)
        for i, (c, f) in enumerate(beans):
            name = f"{stem}_{i:02d}.jpg"
            cv2.imwrite(os.path.join(a.dst, cls, name), c)
            rows.append([f"{cls}/{name}", *[f"{v:.5f}" for v in f]])
        print(f"{path}: {len(beans)} beans")
    with open(os.path.join(a.dst, "features.csv"), "w") as fp:   # train.py --hand ใช้ไฟล์นี้
        fp.write(",".join(["crop", *FEAT_NAMES]) + "\n")
        for r in rows: fp.write(",".join(r) + "\n")