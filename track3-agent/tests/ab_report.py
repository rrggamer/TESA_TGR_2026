"""สรุปผล A/B จากบรรทัด '[ชื่อ] === จบเกม ... เก็บ reveal-XXXX.json' (อ่านจากไฟล์ output ของ monitor)

    python ab_report.py <ไฟล์ output> [คำนำหน้าเวอร์ชัน เช่น A G]
"""
import collections
import glob
import json
import os
import re
import statistics as S
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
P = Path(__file__).parent / "practice" / "logs"
text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
rows = re.findall(r"\[([A-Za-z_]+?)\d*\] === จบเกม .*?เก็บ (reveal-[\d-]+\.json)", text)
by = collections.defaultdict(list)
for ver, rf in rows:
    rev = json.loads((P / "autoplay" / rf).read_text(encoding="utf-8"))
    st_f = P / "autoplay" / rf.replace("reveal-", "states-")
    states = json.loads(st_f.read_text(encoding="utf-8")) if st_f.exists() else []
    t = os.path.getmtime(P / "autoplay" / rf)
    mf = min(glob.glob(str(P / "memory-sandbox-*.json")), key=lambda f: abs(os.path.getmtime(f) - t))
    mem = json.loads(Path(mf).read_text(encoding="utf-8"))
    if not states or not mem["rounds"] or mem["rounds"][0]["round_id"] != "R1":
        continue                                                     # เข้ากลางเกม -> ไม่นับ
    kg = {l["lot_id"]: l["kg_total"] for l in states[0]["lots"]}
    tvt = {l["lot_id"]: l["true_value_per_kg"] for l in rev["lots"]}
    rev_lot, sold = collections.Counter(), collections.Counter()
    offered = cap = 0
    stat = collections.Counter(); mk = []
    for r in mem["rounds"]:
        sent = {(o["rfq_id"], o["lot_id"]): o for o in mem["sent"].get(r["round_id"], [])}
        offered += sum(o["kg"] for o in r["my_offers"])
        for o in r["my_offers"]:
            rev_lot[o["lot_id"]] += o["filled_kg"] * o["price_per_kg"]; sold[o["lot_id"]] += o["filled_kg"]
            stat[o["status"]] += o["kg"]
            s = sent.get((o["rfq_id"], o["lot_id"]))
            if s and o["filled_kg"]:
                mk += [s["markup"]] * (o["filled_kg"] // 60)
    cap = sum((s.get("round_capacity_kg") or 0) for s in states[:8])
    good = [l["lot_id"] for l in rev["lots"] if l["true_grade"] != "REJECT" and l["lot_id"] in kg]
    pr_good = sum(rev_lot[i] + 0.8 * (kg[i] - sold[i]) * tvt[i] for i in good) / sum(kg[i] * tvt[i] for i in good)
    sold_good = sum(sold[i] for i in good) / sum(kg[i] for i in good)
    sc = rev["score"]
    by[ver].append(dict(pr=sc["pr"], pr_good=pr_good, rank=sc["rank"], pr_score=sc["pr_score"], sold_good=sold_good,
                        use=offered / cap if cap else 0, mk=S.mean(mk) if mk else 0,
                        outbid=stat["OUTBID"] / max(sum(stat.values()), 1), above=stat["ABOVE_LIMIT"] / max(sum(stat.values()), 1)))
print(f"{'':4s} {'n':>2s} {'PR':>6s} {'PR*':>6s} {'ขาย/80':>7s} {'ที่1':>4s} {'อันดับ':>6s} {'ขาย*%':>6s} {'ใช้ช่อง':>7s} {'ตัวคูณ':>6s} {'OUTBID':>6s} {'ABOVE':>6s}")
for ver, g in by.items():
    f = lambda k: S.mean(x[k] for x in g)
    print(f"{ver:4s} {len(g):2d} {f('pr'):6.3f} {f('pr_good'):6.3f} {f('pr_score'):7.1f} {sum(x['rank'] == 1 for x in g):4d} {f('rank'):6.1f}"
          f" {f('sold_good') * 100:5.0f}% {f('use') * 100:6.0f}% {f('mk'):6.3f} {f('outbid') * 100:5.0f}% {f('above') * 100:5.0f}%")
    print("     PR* แต่ละเกม:", " ".join(f"{x['pr_good']:.3f}" for x in g), "· อันดับ:", " ".join(str(x["rank"]) for x in g))
print("\nPR* = PR เฉพาะล็อตที่ไม่ใช่ REJECT (ตัดดวงล็อตขายไม่ได้) · ขาย* = % ขายได้ของล็อตเหล่านั้น · ใช้ช่อง = kg ที่เสนอ ÷ ลิมิตรวม")
