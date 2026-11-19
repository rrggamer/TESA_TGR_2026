# team_XX_day3 — Coffee Bean Integration (Day 3)

แยกเมล็ดกาแฟรายเม็ด (instance segmentation) + ทายคลาส 7 คลาสต่อเม็ด จากกองเมล็ดคละคลาส บน Raspberry Pi
ส่งผลทาง HTTP (results.json + zip) และ MQTT (counts ต่อภาพ) พร้อมกันจาก `client.py`

## โมเดล
- YOLO26n-seg 7 คลาส (`black, broken, good, husk, immature, parchment, shell`) ตัวเดียวทำทั้ง mask และคลาส
- รันด้วย NCNN (CPU, ไม่ต้องใช้ torch บน Pi) ที่ imgsz 640 บนบริเวณที่มีเมล็ด (ครอป ROI อัตโนมัติก่อนเข้าโมเดล)
- train: fine-tune จากโมเดลที่ train บนกองเมล็ดสังเคราะห์คละคลาส (สร้างจากภาพ Day 1–2) + ภาพจริง Day 3 (หมุน 4 มุม)
- ผลบนภาพ Day 3 ที่ไม่ได้ใช้ train (3 ภาพ, 23 เม็ด): เจอครบ 23/23 · คลาสถูก 22/23 · ~110 ms/ภาพ บน PC

## ติดตั้ง (Raspberry Pi OS 64-bit)
```bash
pip install -r requirements.txt
```

## รัน inference
```bash
python infer.py --input <test_folder/> --weights weights/ --out outputs/
```
ผลลัพธ์ใน `outputs/`
| ไฟล์ | เนื้อหา |
|---|---|
| `seg_mask/<ภาพ>.png` | ต่อภาพ test · 0 = พื้น, 1..N = เม็ดที่ 1..N |
| `predictions.json` | ต่อภาพ: `image_name`, `objects` [`id`, `class`, `polygon`] (id ตรงกับ seg_mask) |
| `mqtt/<ภาพ>.json` | ข้อความที่ publish: `image`, `latency_ms`, `counts` (ครบ 7 key), `defect_ratio` |
| `vis/<ภาพ>.jpg` | ภาพ overlay mask |

`counts` ใน `mqtt/` นับจาก objects ชุดเดียวกับ `predictions.json` / `seg_mask` จึงตรงกันเสมอ
`defect_ratio` = เม็ดที่ไม่ใช่ `good` ÷ เม็ดทั้งหมด

ค่าตั้งของโมเดลอยู่ใน `weights/infer_config.json` (conf 0.10, iou 0.7, threads 4)

## client.py (HTTP + MQTT ในสคริปต์เดียว)
ตั้งค่า URL / credentials / topic ใน `client_config.json` แล้ว
```bash
python client.py                                    # GET ภาพ test → infer.py → POST results.json + zip และ publish MQTT พร้อมกัน
python client.py --dry-run --input <test_folder/>   # ทดสอบโดยไม่ต่อเครือข่าย
python client.py --skip-infer                       # ส่งซ้ำจาก outputs/ เดิม
```
- results.json = predictions.json · zip = `team_XX_Day3.zip` (โค้ด + weights + outputs)
- HTTP POST และ MQTT publish (QoS 1) ทำพร้อมกันใน 2 thread · ล้มเหลวลองซ้ำ 3 ครั้ง

## train
```bash
python train.py --data <dataset>/beans.yaml --model <pretrained>.pt --name crop7d3 --epochs 40 --batch 8
```
dataset เป็น YOLO-seg ของภาพที่ครอป ROI แบบเดียวกับ `infer.py` · export เป็น NCNN แล้ววางใน `weights/best_ncnn_model/`
