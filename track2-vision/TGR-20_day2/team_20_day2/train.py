"""ขั้นที่ 5: train YOLO26n-seg (fallback YOLO11n-seg) บนกองเมล็ดสังเคราะห์

fine-tune ทั้งโมเดลจาก COCO, imgsz 640, mosaic + HSV + copy_paste, early stopping ที่ val (กองสังเคราะห์จาก source ฝั่ง val)

python train.py --data datasets/crop/beans.yaml --name crop7                 # 7 คลาส
python train.py --data datasets/crop/beans.yaml --name crop1 --single-cls    # baseline 1 คลาส "bean"
python train.py ... --epochs 1 --name timing                                  # วัดเวลา/epoch ก่อน train จริง
"""
import argparse, os, shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/crop/beans.yaml")
    ap.add_argument("--model", default="yolo26n-seg.pt")
    ap.add_argument("--name", default="crop7")
    ap.add_argument("--single-cls", action="store_true", help="รวมทุกคลาสเป็น 'bean' (คะแนน P1/P2 ไม่คิดคลาส)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="0")
    ap.add_argument("--copy-paste", type=float, default=0.3)
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="runs/seg")
    ap.add_argument("--resume", action="store_true", help="train ต่อจาก runs/seg/<name>/weights/last.pt (เช่น หลัง crash)")
    ap.add_argument("--weights-out", help="คัดลอก best.pt ไปที่นี่เมื่อเสร็จ (เช่น weights/best.pt)")
    a = ap.parse_args()

    from ultralytics import YOLO
    if a.resume:
        last = os.path.join(os.path.abspath(a.project), a.name, "weights", "last.pt")
        YOLO(last).train(resume=True, workers=a.workers)
        print("best:", os.path.join(os.path.dirname(last), "best.pt")); return
    try:
        model = YOLO(a.model)
    except Exception as e:                                         # ไม่มี YOLO26 → YOLO11
        print(f"โหลด {a.model} ไม่ได้ ({e}) → ใช้ yolo11n-seg.pt"); model = YOLO("yolo11n-seg.pt")
    model.train(
        data=a.data, epochs=a.epochs, patience=a.patience, imgsz=a.imgsz, batch=a.batch, workers=a.workers,
        device=a.device, single_cls=a.single_cls, seed=a.seed, deterministic=True,
        project=os.path.abspath(a.project), name=a.name, exist_ok=True, pretrained=True,
        mosaic=a.mosaic, close_mosaic=10, copy_paste=a.copy_paste,
        hsv_h=0.01, hsv_s=0.4, hsv_v=0.3,                          # สีเป็นข้อมูลของคลาส → hue ขยับน้อย
        degrees=0.0, fliplr=0.5, flipud=0.5, scale=0.4, translate=0.1,
        max_det=300, plots=True, verbose=False,
    )
    best = os.path.join(os.path.abspath(a.project), a.name, "weights", "best.pt")
    print("best:", best)
    if a.weights_out and os.path.isfile(best):
        os.makedirs(os.path.dirname(a.weights_out) or ".", exist_ok=True); shutil.copy2(best, a.weights_out)
        print("copied →", a.weights_out)


if __name__ == "__main__":
    main()
