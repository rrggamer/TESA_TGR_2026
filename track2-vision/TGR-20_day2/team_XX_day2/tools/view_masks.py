"""แปลง seg_mask (ค่า 0..N ซึ่งดูด้วยตาแล้วมืดเกือบดำ) เป็นภาพไว้ดู — ไม่แก้ไฟล์ seg_mask ต้นฉบับ

--mode white : ทุกเม็ดสีขาว พื้นดำ และมีเส้นดำคั่นระหว่างเม็ดที่ติดกัน (เห็นว่าแยกเม็ดถูกไหม)
--mode color : แต่ละ id คนละสี

python view_masks.py --masks outputs/seg_mask --out mask_view --mode white
** อย่าเขียนลงใน outputs/ — ไฟล์ส่งงานต้องเป็นค่า id 0..N ตามสเปก **
"""
import argparse, glob, os, sys
import cv2, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["white", "color"], default="white")
    ap.add_argument("--max-side", type=int, default=2048, help="ย่อภาพผลลัพธ์ (0 = ขนาดเดิม)")
    a = ap.parse_args()
    if os.path.abspath(a.out) == os.path.abspath(a.masks): sys.exit("ERROR: --out ต้องไม่ใช่โฟลเดอร์ seg_mask")
    os.makedirs(a.out, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(a.masks, "*.png"))):
        m = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.int32)
        if a.max_side and max(m.shape) > a.max_side:
            s = a.max_side / max(m.shape)
            m = cv2.resize(m.astype(np.float32), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST).astype(np.int32)
        edge = np.zeros(m.shape, bool)                                   # ขอบระหว่าง id ต่างกัน
        edge[:, 1:] |= m[:, 1:] != m[:, :-1]; edge[1:] |= m[1:] != m[:-1]
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
        if a.mode == "white":
            v = np.where(m > 0, 255, 0).astype(np.uint8); v[edge & (m > 0)] = 0
        else:
            pal = np.random.default_rng(0).integers(60, 256, (int(m.max()) + 1, 3)).astype(np.uint8); pal[0] = 0
            v = pal[m]; v[edge & (m > 0)] = 0
        cv2.imwrite(os.path.join(a.out, os.path.basename(p)), v)
        print(f"{os.path.basename(p)}: {len(np.unique(m)) - 1} beans")


if __name__ == "__main__":
    main()
