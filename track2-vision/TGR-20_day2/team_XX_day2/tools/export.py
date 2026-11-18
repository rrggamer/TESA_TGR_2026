"""export best.pt → NCNN หลายขนาด ลงใน weights/

python export.py --pt runs/seg/crop7/weights/best.pt --wdir weights --sizes 640 480 416 320 --default 640
→ weights/best.pt, weights/best_ncnn_model_<s>/ ทุกขนาด และ weights/best_ncnn_model/ (= ขนาด --default)
"""
import argparse, os, shutil


def one2one_copy(pt, dst):
    """NCNN export ปิดหัว end2end (ไม่รองรับ TopK) แล้วใช้หัว one-to-many ซึ่งต้อง NMS → ในกองเมล็ดที่กล่องซ้อนกันมาก
    NMS จะลบเม็ดจริงทิ้ง  วิธีแก้: คัดลอกน้ำหนักหัว one-to-one (NMS-free) ทับหัว one-to-many ก่อน export
    → graph/รูปแบบ output เหมือนเดิม (4+nc+32, anchors) แต่คะแนนมาจากหัว one-to-one: แค่ตัดด้วย conf ก็พอ"""
    import copy, torch
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    for key in ("model", "ema"):
        m = ck.get(key)
        if m is None: continue
        head = m.model[-1]
        for n in ("cv2", "cv3", "cv4"):
            if getattr(head, "one2one_" + n, None) is not None:
                setattr(head, n, copy.deepcopy(getattr(head, "one2one_" + n)))
    torch.save(ck, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True); ap.add_argument("--wdir", default="weights")
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 480, 416, 320])
    ap.add_argument("--default", type=int, default=640)
    ap.add_argument("--one2many", action="store_true",
                    help="ใช้หัว one-to-many เดิม (ต้อง NMS) แทนหัว one-to-one (NMS-free) ของ YOLO26")
    a = ap.parse_args()
    from ultralytics import YOLO
    os.makedirs(a.wdir, exist_ok=True)
    pt = os.path.join(a.wdir, "best.pt")
    if os.path.abspath(a.pt) != os.path.abspath(pt): shutil.copy2(a.pt, pt)
    src = pt if a.one2many else one2one_copy(pt, os.path.join(a.wdir, "_one2one.pt"))
    for s in a.sizes:
        tmp = os.path.join(a.wdir, f"_tmp_{s}"); os.makedirs(tmp, exist_ok=True)
        shutil.copy2(src, os.path.join(tmp, "best.pt"))
        out = YOLO(os.path.join(tmp, "best.pt")).export(format="ncnn", imgsz=s, device="cpu", half=False)
        dst = os.path.join(a.wdir, f"best_ncnn_model_{s}")
        if os.path.isdir(dst): shutil.rmtree(dst)
        shutil.move(out, dst); shutil.rmtree(tmp, ignore_errors=True)
        for f in ("model_ncnn.py",):                                  # สคริปต์ทดสอบที่ import torch → ไม่ต้องใช้
            p = os.path.join(dst, f)
            if os.path.isfile(p): os.remove(p)
        print("exported", dst)
    d = os.path.join(a.wdir, "best_ncnn_model")
    if os.path.isdir(d): shutil.rmtree(d)
    shutil.copytree(os.path.join(a.wdir, f"best_ncnn_model_{a.default}"), d)
    print("default →", d)
    if src != pt: os.remove(src)


if __name__ == "__main__":
    main()
