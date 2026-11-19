# TGR-20 Day 2 — Instance Segmentation เมล็ดกาแฟโรบัสต้า (กองเมล็ด)

YOLO26n-seg (1 คลาส "bean") → NCNN บน CPU ของ Raspberry Pi (ไม่ใช้ torch ตอน inference)
+ ตัวจำแนกคลาสต่อเม็ดแบบเบา (สี/รูปทรง 12 ค่า → Logistic Regression ใน numpy)

## โครงสร้าง

```
team_XX_day2/                 # ตาม DAY2_INFER_SPEC.md
├── README.md
├── requirements.txt          # สำหรับ inference เท่านั้น
├── train.py                  # train YOLO26n-seg บนกองเมล็ดสังเคราะห์
├── infer.py                  # ไฟล์เดียวที่ใช้ตอน test (ใช้แค่ไฟล์นี้ + weights/ ไม่ import ไฟล์อื่นในโปรเจกต์)
├── weights/
│   ├── infer_config.json           # infer.py อ่านไฟล์นี้: เลือกโมเดล, conf, iou, min_keep, การอ่านภาพ
│   ├── best_ncnn_model/            # NCNN ที่ใช้ (ค่าเริ่มต้น = 640)
│   ├── best_ncnn_model_{640,480,416,320}/
│   ├── cls_head.json               # ตัวจำแนกคลาสต่อเม็ด (StandardScaler + LogReg เป็น JSON, ใช้ numpy)
│   ├── best.pt                     # YOLO26n-seg 1 คลาส (PyTorch, ไว้ export ใหม่ — infer.py ไม่ใช้)
│   └── cls_best.pt, cls_ncnn_model_{416,320}/   # (สำรอง ก+) โมเดล 7 คลาสสำหรับหาคลาส
├── outputs/                  # ผลรันบนภาพ train จริง 21 ภาพ (ตัวอย่าง)
│   ├── seg_mask/<ชื่อภาพ>.png     ขนาดเท่าภาพต้นฉบับ (หลัง EXIF), 0=พื้น, 1..N=id (uint8; uint16 ถ้า N>255)
│   ├── predictions.json
│   └── vis/                       overlay สีต่อเม็ด (ย่อด้านยาว 1024 px)
└── tools/                    # ไม่ใช้ตอน test: เตรียมข้อมูล / จูน / วัดผล
    ├── common.py autolabel.py make_piles.py export.py tune.py train_cls.py
    ├── score.py              # local scorer (P1/P2 + F1 ที่ IoU 0.5–0.95)
    ├── bench_pi.py           # วัด median latency + อุณหภูมิชิปบน Pi
    ├── validate_outputs.py   # ตรวจผลตามสเปก (ขนาด mask, id ↔ json, polygon ↔ contour)
    └── view_masks.py         # แปลง seg_mask เป็นภาพขาว/สีไว้ดู (เขียนนอก outputs/)
```

## ติดตั้งบน Raspberry Pi (Raspberry Pi OS 64-bit)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # ncnn มี wheel aarch64 สำหรับ Python 3.11 / 3.13
```

ไม่ต้องติดตั้ง torch / ultralytics (infer.py ไม่ import) — ทดสอบแล้วใน venv ใหม่ที่ติดตั้งจาก requirements.txt อย่างเดียว

## รัน

```bash
python infer.py --input <test_folder/> --weights weights/ --out outputs/
```

| อาร์กิวเมนต์ | ความหมาย |
|---|---|
| `--vote` | ทุกเม็ดในภาพได้คลาสเดียวกัน (เฉลี่ยความน่าจะเป็นทั้งภาพ) — เหมาะกับภาพที่มีคลาสเดียว |
| `--ncnn best_ncnn_model_416` | เลือกขนาดโมเดล (ค่าเริ่มต้นอ่านจาก `weights/infer_config.json`) |
| `--decode-min-side 900` | อ่าน JPEG ย่อ 1/4 แทน 1/2 (เร็วขึ้นบน Pi, คะแนนลดเล็กน้อย) |
| `--cls-model cls_ncnn_model_416` | (ก+) ใช้โมเดล 7 คลาสหาคลาส แม่นกว่ามากแต่ช้าขึ้น |
| `--conf --iou --min-keep --threads --no-roi` | override ค่าใน infer_config.json |

โปรแกรมพิมพ์เวลาต่อภาพ (อ่านภาพ → เขียน seg_mask/vis เสร็จ) และ median ตอนจบ; warm-up 1 รอบด้วยภาพ dummy ก่อนเริ่มจับเวลา

ตรวจผล: `python tools/validate_outputs.py --input <test_folder/> --out outputs/`

### วัดบน Pi จริง

```bash
python tools/bench_pi.py --input <images/> --weights weights/ --rounds 3                       # median + อุณหภูมิ + throttling
python tools/bench_pi.py --input <images/> --weights weights/ --ncnn best_ncnn_model_416 --decode-min-side 900
```

ดู mask ด้วยตา: `python tools/view_masks.py --masks outputs/seg_mask --out mask_view --mode white` (ไฟล์ใน seg_mask เป็นค่า id 0..N จึงดูมืด — ห้ามแก้เป็นสีขาว)

**ถ้า median บน Pi 4 เกิน ~0.9 s** ให้แก้ `weights/infer_config.json` (ไม่ต้องแก้โค้ด) เช่น
`"ncnn": "best_ncnn_model_480", "decode_min_side": 900` แล้ววัดใหม่ — เลือกขนาดใหญ่สุดที่ median ≤ 0.9 s

## Pipeline ของ infer.py (ต่อภาพ)

1. อ่าน JPEG แบบย่อ 1/2 ด้วย libjpeg DCT scaling (`IMREAD_REDUCED_COLOR_2`) — เร็วกว่าอ่านเต็ม 12 MP มาก, ขนาดต้นฉบับอ่านจาก header (รองรับ EXIF)
2. **ROI crop**: หากรอบของกองจากภาพย่อ 512 px (สีต่างจากกระดาษใน Lab) → ส่งเฉพาะกองเข้าโมเดล เมล็ดจึงใหญ่ขึ้นใน input (หาไม่เจอ → ใช้ทั้งภาพ)
3. letterbox สี่เหลี่ยม → NCNN YOLO26n-seg (หัว **one-to-one / NMS-free**, ดูหมายเหตุ) → ตัดด้วย conf (+ NMS iou 0.7 กันพลาด)
4. mask ต่อเม็ด: proto × coefficient เฉพาะในกรอบของเม็ด แล้ว warp ไปพิกัดภาพเต็มเฉพาะกรอบนั้น (ไม่ทำทั้งภาพ N ครั้ง)
5. เรียงตาม conf มาก → น้อย: เม็ดที่มั่นใจกว่าได้พิกเซลก่อน; เม็ดที่เหลือพิกเซลของตัวเอง < 50% = ซ้ำ → ทิ้ง
6. แต่ละ id เก็บชิ้นใหญ่สุด → polygon = `findContours(CHAIN_APPROX_NONE)` จาก mask เดียวกัน, id ต่อเนื่อง 1..N
7. คลาสต่อเม็ด: feature สี/รูปทรง 12 ค่า (white balance ด้วยกระดาษ) → LogReg (numpy)
8. เขียน seg_mask (PNG filter-none + zlib RLE ใน thread ขนาน), vis (1024 px), เก็บ record json

**หมายเหตุ NCNN + YOLO26:** Ultralytics ปิดหัว end-to-end ตอน export NCNN (ไม่รองรับ TopK) แล้วใช้หัว one-to-many ที่ต้อง NMS
— ในกองเมล็ดกล่องซ้อนกันมาก NMS ลบเม็ดจริงทิ้ง (เช่น GT 101 → ทายได้ 86) `tools/export.py` จึงคัดลอกน้ำหนักหัว
one-to-one ทับหัว one-to-many ก่อน export: graph เหมือนเดิมแต่ไม่ต้องพึ่ง NMS → P1+P2 บน val เพิ่มจาก 12.83 เป็น 13.96

## ผลที่วัดได้

### คะแนน (local scorer = เกณฑ์ P1/P2 ของกรรมการ)
| ชุดทดสอบ | P1 | P2 | P1+P2 |
|---|---|---|---|
| val กองสังเคราะห์ 80 ภาพ (เมล็ด/พื้นหลังจาก source ที่ไม่ได้ใช้ train) | 6.60 | 7.36 | **13.96** / 15 |
| ภาพเต็ม 3072×4096 กองสังเคราะห์ 7 ภาพ (คลาสละภาพ, 85–118 เม็ด) | 6.51 | 7.40 | **13.91** / 15 |
| ภาพจริงเมล็ดวางแยก 7 ภาพ (source ฝั่ง val) | 7.20 | 7.50 | **14.70** / 15 |

### ขนาดโมเดล vs คะแนน vs เวลา (ภาพ 3072×4096, 7 ภาพ × 3 รอบ)
PC = Intel i5-14450HX, 4 threads; Pi 4 = ประมาณการ (ดูวิธีคิดด้านล่าง) — **ต้องยืนยันด้วย bench_pi.py บน Pi จริง**

| imgsz | อ่านภาพ | P1+P2 | median PC | ประมาณ Pi 4 |
|---|---|---|---|---|
| 640 | 1/2 | **13.91** | 168 ms | 1.45–1.8 s |
| 640 | 1/4 | 13.88 | 159 ms | 1.4–1.7 s |
| 480 | 1/2 | 13.34 | 144 ms | 1.2–1.55 s |
| 480 | 1/4 | 13.27 | 122 ms | 1.0–1.3 s |
| 416 | 1/4 | 12.92 | 117 ms | 1.0–1.25 s |
| 320 | 1/4 | 11.60 | 110 ms | 0.9–1.2 s |

วิธีประมาณ: เวลาโมเดล (ncnn 4 threads; PC 640/480/416/320 = 52/27/21/13 ms) × ~10
(Pi 4 ≈ 3 เท่าของ Pi 5 และ Pi 5 ≈ 3–3.5 เท่าของ PC นี้) + ส่วนที่ไม่ใช่โมเดล (decode, mask ภาพเต็ม, PNG 12 MP, json) × 8–11
— ส่วนที่ไม่ใช่โมเดลเป็นสัดส่วนใหญ่ ตัวคูณจริงบน ARM ไม่แน่นอน จึงเป็นช่วงกว้าง

### การหาคลาส (ไม่มีผลต่อ P1/P2 ของ Day 2)
| วิธี | แม่นรายเม็ด | แม่นรายภาพ (vote) สังเคราะห์ / จริง | เวลาเพิ่ม |
|---|---|---|---|
| ค่าเริ่มต้น: สี/รูปทรง + LogReg | 0.63 | 52/59 / 6/7 | ~15–25 ms (PC) |
| ก+ `--cls-model cls_ncnn_model_416` | 0.87 | 57/59 / 7/7 | + โมเดลที่ 416 อีกตัว |

## Training (บน PC ที่มี GPU) — ทำซ้ำได้ (ทุกสคริปต์ตั้ง seed)

```bash
pip install ultralytics scikit-learn scipy           # + torch (CUDA)
python tools/autolabel.py --src <day_1/> --out data/autolabel          # FastSAM-s + ตัวกรองสี → 175/175 เม็ด
python tools/make_piles.py --n-train 400 --n-val 80                    # กองสังเคราะห์ + datasets/{crop,full}
python train.py --data datasets/crop/beans.yaml --name crop1 --single-cls --epochs 60 --patience 15 --batch 8
python tools/export.py --pt runs/seg/crop1/weights/best.pt --wdir weights --sizes 640 480 416 320 --default 640
python tools/tune.py --weights weights --ncnn best_ncnn_model --save best_ncnn_model   # จูน conf/iou/min_keep บน val
python tools/train_cls.py --weights weights                            # ตัวจำแนกคลาส → weights/cls_head.json
python tools/score.py --gt data/piles/val/masks --pred <out>/seg_mask -v
```

- **ข้อมูล:** 21 ภาพ (7 คลาส × 3 layout) ภาพละ 8–9 เม็ดวางแยกกัน ไม่มี label → auto-label ด้วย FastSAM-s ตัดกับ foreground สี (Lab-b / dark) และเกลาขอบ ได้ 175/175 เม็ด
- **กองสังเคราะห์:** bean bank RGBA (ขอบ feather ด้านใน กัน halo), พื้นหลัง = ภาพจริงที่ inpaint เมล็ดออก (เก็บแสง/ขอบโต๊ะจริง), 30–125 เม็ด/ภาพ ซ้อนกันจริง, rotate/flip/scale ±15%/jitter/เงา, เม็ดถูกทับเหลือ < 30% หรือแตกเป็นชิ้น = ไม่ยอมวาง, 80% คลาสเดียว 20% ผสม
- **ไม่มี data leak:** แบ่ง train/val ตาม source image (คลาสละ 1 ภาพเป็น val) ทั้งเมล็ดและพื้นหลัง
- **เทียบโมเดล (val):** 1 คลาส 13.96 > 7 คลาส 13.26 (ไม่ต้องเสียความจุไปแยกคลาสที่ไม่นับคะแนน); crop ROI 12.89 > ทั้งภาพ 12.19 (โมเดล 7 คลาส หัวเดิม)

## ข้อจำกัด

- ยังไม่มีภาพกองจริง: วัดได้บนกองสังเคราะห์และภาพจริงที่เมล็ดวางแยกเท่านั้น กองจริง (เม็ดตั้งเอียง/เงาซ้อน 3D) อาจได้คะแนนต่ำกว่า
- latency บน Pi 4 เป็นค่าประมาณ — ใช้ `tools/bench_pi.py` เลือก config สุดท้าย
- คลาสจากสี/รูปทรงสับสน immature↔parchment, broken↔shell↔immature (ไม่มีผลกับคะแนน Day 2)
