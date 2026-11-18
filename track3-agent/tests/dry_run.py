"""รัน strategy.decide() กับ snapshot แล้วพิมพ์ offer ที่ agent "จะ" ส่ง — ไม่ส่งจริง ไม่เรียก LLM (rule-based)

    python dry_run.py                          # snapshot ล่าสุด (หรือ example.json)
    python dry_run.py snap-1118-143000.json
    python dry_run.py --markup SPECIALTY=1.15 COMMERCIAL=1.05 INSTANT=0.95   # ลอง markup โดยไม่แก้ไฟล์จริง
"""
import random
import sys

from _setup import load_snapshot
import strategy
from appraisal import estimate_lots, grade

args = sys.argv[1:]
name = next((a for a in args if a.endswith(".json")), None)
if "--markup" in args:
    for a in args[args.index("--markup") + 1:]:
        if "=" in a:
            k, v = a.split("=")
            strategy.MARKUP[k.upper()] = float(v)

info, state = load_snapshot(name)
rules = info["rules"]
tv = estimate_lots(rules, state["lots"])
ctx = strategy.Context(rules, info["calendar"], tv, random.Random(0), None, None, state.get("seconds_left") or 999)

print("MARKUP:", strategy.MARKUP, "| round:", state["round"]["round_id"], "| cap left:", state["capacity_left_kg"])
print("\nLOTS")
for l in state["lots"]:
    s = l["sample"]
    print(f"  {l['lot_id']:12} avail {l['kg_available']:5} kg  d{s['defect_pct']:6} m{s['moisture_pct']:6} s{s['screen']:3}"
          f"  {grade(rules, s['defect_pct'])[0]:10} TV {tv[l['lot_id']]:7.2f}")
print("\nRFQS")
for r in state["rfqs"]:
    ok = [l["lot_id"] for l in state["lots"] if strategy.spec_ok(l["sample"], r["spec"])]
    print(f"  {r['rfq_id']:8} {r['buyer_type']:10} {r['kg']:5} kg  spec {r['spec']}  ผ่าน: {ok or '-'}")

offers, source = strategy.decide(state, ctx)
print(f"\nOFFERS ({source}) — ไม่ได้ส่งจริง")
for o in offers:
    print(f"  {o['rfq_id']:8} {o['lot_id']:12} {o['kg']:5} kg  {o['price_per_kg']:8.2f} ฿/kg"
          f"  (x{o['price_per_kg'] / tv[o['lot_id']]:.3f})")
print("  รวม", sum(o["kg"] for o in offers), "kg")
for n in ctx.notes:
    print("  note:", n)
