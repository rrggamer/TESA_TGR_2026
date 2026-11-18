"""ขั้นที่ 8: ตรวจผลลัพธ์ของ infer.py ให้ตรงสเปก

- ทุกภาพ input มี seg_mask/<stem>.png ขนาดเท่าภาพต้นฉบับ (หลังหมุนตาม EXIF), uint8 หรือ uint16
- predictions.json อ่านได้ เป็น list ของ {"image_name", "objects": [{"id", "class", "polygon"}]}
- id ใน json = ค่าพิกเซลใน seg_mask ทุกตัวพอดี (ไม่ขาดไม่เกิน), id ต่อเนื่อง 1..N
- polygon ไม่ว่าง (≥3 จุด), อยู่ในขอบภาพ, และตรงกับ contour ของ mask (findContours CHAIN_APPROX_NONE)
- class อยู่ในรายชื่อ 7 คลาส, มี vis/<stem>.jpg

python validate_outputs.py --input test_folder/ --out outputs/
"""
import argparse, json, os, sys
import cv2, numpy as np
from PIL import Image, ImageOps

CLASSES = {"black", "broken", "good", "husk", "immature", "parchment", "shell"}
EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    errs = []
    imgs = sorted(f for f in os.listdir(a.input) if f.lower().endswith(EXTS))
    try:
        preds = json.load(open(os.path.join(a.out, "predictions.json"), encoding="utf-8"))
    except Exception as e:
        print(f"FAIL: อ่าน predictions.json ไม่ได้: {e}"); return 1
    if not isinstance(preds, list): errs.append("predictions.json ต้องเป็น list")
    by = {p.get("image_name"): p for p in preds}
    if set(by) != set(imgs): errs.append(f"image_name ไม่ตรงกับภาพ input: ขาด {sorted(set(imgs) - set(by))} เกิน {sorted(set(by) - set(imgs))}")
    tot = 0
    for name in imgs:
        stem = os.path.splitext(name)[0]; p = by.get(name)
        if p is None: continue
        with Image.open(os.path.join(a.input, name)) as im: W, H = ImageOps.exif_transpose(im).size
        mp = os.path.join(a.out, "seg_mask", stem + ".png")
        m = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if m is None: errs.append(f"{name}: ไม่มี {mp}"); continue
        if m.shape != (H, W): errs.append(f"{name}: seg_mask {m.shape[::-1]} ≠ ภาพ {(W, H)}")
        if m.dtype not in (np.uint8, np.uint16): errs.append(f"{name}: dtype {m.dtype}")
        if not os.path.isfile(os.path.join(a.out, "vis", stem + ".jpg")): errs.append(f"{name}: ไม่มี vis")
        objs = p.get("objects")
        if not isinstance(objs, list): errs.append(f"{name}: objects ไม่ใช่ list"); continue
        ids = [o.get("id") for o in objs]
        if ids != list(range(1, len(objs) + 1)): errs.append(f"{name}: id ไม่ต่อเนื่อง 1..N")
        present = set(np.unique(m).tolist()) - {0}
        if present != set(ids): errs.append(f"{name}: id ใน json ≠ ค่าใน mask (เฉพาะ json {sorted(set(ids) - present)[:5]}, เฉพาะ mask {sorted(present - set(ids))[:5]})")
        for o in objs:
            if set(o) != {"id", "class", "polygon"}: errs.append(f"{name} id {o.get('id')}: key {sorted(o)}")
            if o.get("class") not in CLASSES: errs.append(f"{name} id {o.get('id')}: class '{o.get('class')}'")
            poly = np.asarray(o.get("polygon"), dtype=np.int64)
            if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
                errs.append(f"{name} id {o.get('id')}: polygon ว่าง/รูปแบบผิด"); continue
            if poly[:, 0].min() < 0 or poly[:, 1].min() < 0 or poly[:, 0].max() >= W or poly[:, 1].max() >= H:
                errs.append(f"{name} id {o.get('id')}: polygon ออกนอกภาพ")
            # polygon ต้องเป็น contour ของ mask เดียวกัน
            x0, y0 = poly.min(0); x1, y1 = poly.max(0) + 1
            sub = (m[y0:y1, x0:x1] == o["id"]).astype(np.uint8)
            cn, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(cn) != 1 or not np.array_equal(cn[0].reshape(-1, 2) + [x0, y0], poly):
                errs.append(f"{name} id {o.get('id')}: polygon ไม่ตรงกับ contour ของ mask ({len(cn)} ชิ้น)")
        tot += len(objs)
    print(f"{len(imgs)} images, {tot} objects")
    if errs:
        print(f"FAIL ({len(errs)} ปัญหา):"); [print("  -", e) for e in errs[:40]]; return 1
    print("PASS: ผลลัพธ์ตรงสเปกทุกข้อ"); return 0


if __name__ == "__main__":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
    sys.exit(main())
