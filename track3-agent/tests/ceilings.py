"""พิสูจน์ว่า "เพดานลับ" ของผู้ซื้อแต่ละรายคงที่ไหม จาก log ทุกเกมที่เคยเล่น (ไม่ต่อเน็ต)

    python ceilings.py

หลักคิด: ratio = ราคาที่เสนอ ÷ TV ของเรา
  ACCEPTED / PARTIAL  -> ผู้ซื้อรับราคานี้ได้    -> เพดาน >= ratio  (ขอบล่าง)
  ABOVE_LIMIT         -> แพงเกินเพดาน           -> เพดาน <  ratio  (ขอบบน)
  OUTBID              -> ไม่บอกอะไรเรื่องเพดาน (เจอจริง: OUTBID ที่ราคาซึ่งเกินเพดาน)
ถ้าเพดานคงที่จริง ทุกเกมทุกรอบต้อง "ขอบล่าง < ขอบบน" เสมอ — เจอขัดกันแปลว่าไม่คงที่
"""
import collections
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).parent.parent
FILES = sorted(list((ROOT / "Coffee-TESA-starter" / "logs").glob("memory-*.json"))
               + list((ROOT / "tests" / "practice" / "logs").glob("memory-*.json")), key=lambda p: p.stat().st_mtime)

obs = collections.defaultdict(list)          # buyer -> [(game, round, ratio, status)]
for g, f in enumerate(FILES, 1):
    m = json.loads(f.read_text(encoding="utf-8"))
    for rnd in m["rounds"]:
        buyer = {s["rfq_id"]: s["buyer"] for s in rnd.get("market_summary", [])}
        sent = {(o["rfq_id"], o["lot_id"]): o for o in m["sent"].get(rnd["round_id"], [])}
        for o in rnd["my_offers"]:
            s = sent.get((o["rfq_id"], o["lot_id"]))
            if s and o["rfq_id"] in buyer:
                obs[buyer[o["rfq_id"]]].append((g, rnd["round_id"], s["markup"], o["status"]))

print(f"อ่าน {len(FILES)} เกม\n")
for b, rows in sorted(obs.items()):
    ok = [r for r in rows if r[3] in ("ACCEPTED", "PARTIAL")]
    no = [r for r in rows if r[3] == "ABOVE_LIMIT"]
    lo = max((r[2] for r in ok), default=None)
    hi = min((r[2] for r in no), default=None)
    games = sorted({r[0] for r in rows})
    if lo is not None and hi is not None and lo >= hi:
        verdict = "❌ ขัดกัน: เคยขายได้ที่ราคาสูงกว่าที่เคยโดน ABOVE_LIMIT -> เพดานไม่คงที่"
    elif lo is not None and hi is not None:
        verdict = f"✅ ไม่ขัดกัน · เพดานอยู่ในช่วง {lo:.3f} – {hi:.3f} (กว้าง {hi - lo:.3f})"
    else:
        verdict = f"ข้อมูลยังไม่พอ · ขอบล่าง {lo} · ขอบบน {hi}"
    print(f"{b}  (เกม {games}, {len(rows)} ข้อเสนอ)")
    print("   ", verdict)
    print("    ", " ".join(f"g{r[0]}{r[1]}:{r[2]}{'✓' if r[3] in ('ACCEPTED', 'PARTIAL') else '✗' if r[3] == 'ABOVE_LIMIT' else '·'}"
                          for r in rows))
    print()
print("✓ = ขายได้ (ขอบล่าง)  ✗ = ABOVE_LIMIT (ขอบบน)  · = OUTBID/อื่นๆ (ไม่นับ)")
