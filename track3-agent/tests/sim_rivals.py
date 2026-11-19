"""ตลาดจำลองออฟไลน์: strategy.py ตัวจริงแข่งกับคู่แข่งแบบต่าง ๆ (กดราคา / เก็งของ / starter) — ไม่ต่อตลาด

    python sim_rivals.py            # ทุกสถานการณ์ × 30 เกม
เป็นแบบจำลองจากข้อมูลตลาดฝึก 18 พ.ย. (เพดานผู้ซื้อ, ลิมิต 1.5×) ไม่ใช่ตลาดจริง — ใช้ดูว่ากลยุทธ์ "พัง" ในสถานการณ์ไหน
"""
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
import os  # noqa: E402
sys.path.insert(0, str(HERE / "practice") if os.environ.get("SIM_PRACTICE") else str(HERE.parent / "Coffee-TESA-starter"))   # SIM_PRACTICE=1 -> ตัวซ้อม
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
if os.environ.get("SIM_FILE"):                                       # SIM_FILE=practice/strategy_v4.py -> ไฟล์ที่ระบุ
    import importlib.util  # noqa: E402
    _spec = importlib.util.spec_from_file_location("strategy", str(HERE / os.environ["SIM_FILE"]))
    strategy = importlib.util.module_from_spec(_spec); sys.modules["strategy"] = strategy; _spec.loader.exec_module(strategy)
else:
    import strategy  # noqa: E402  ไฟล์แข่งจริง
import os  # noqa: E402
for _kv in filter(None, os.environ.get("EXP", "").split(";")):      # ทดลองค่า (ไม่แก้ไฟล์จริง)
    _k, _v = _kv.split("=", 1); setattr(strategy, _k.strip(), eval(_v))

BAG = 60
DAY20 = "--day20" in sys.argv          # จำลองแบบสไลด์วันที่ 20: ซ้อม 2 + จริง 15 รอบ มีเทศกาล/ใกล้ปิดตลาด
ROUNDS, PRACTICE = (15, 2) if DAY20 else (8, 0)
DEMAND = float(sys.argv[sys.argv.index("--demand") + 1]) if "--demand" in sys.argv else 1.0   # ความต้องการ ÷ ของที่มี
RULES = {"timeline": {"rounds": ROUNDS, "practice_rounds": PRACTICE}, "bag_kg": BAG}
# ⚠ ตัวเลขถ่วงความต้องการเป็นค่าสมมติของการจำลองเท่านั้น (สไลด์บอกแค่ทิศทาง ไม่บอกตัวเลข) · strategy อ่านจากปฏิทินจริงตอนแข่ง
FEST = float(sys.argv[sys.argv.index("--fest") + 1]) if "--fest" in sys.argv else 2.0     # Specialty ช่วงเทศกาล ×
CAL = ([{"rounds": "1-5", "label": "ปกติ", "demand_weight": 1.0, "specialty_share": 0.2},
        {"rounds": "6-8", "label": "เทศกาลกาแฟ", "demand_weight": 1.2, "specialty_share": 0.4},
        {"rounds": "9-12", "label": "ปกติ", "demand_weight": 1.0, "specialty_share": 0.2},
        {"rounds": "13-14", "label": "ใกล้ปิดตลาด", "demand_weight": 0.6, "specialty_share": 0.2},
        {"rounds": "15", "label": "รอบสุดท้าย", "demand_weight": 0.4, "specialty_share": 0.2}] if DAY20 else [])
W = {n: r["demand_weight"] for r in CAL for n in range(int(str(r["rounds"]).split("-")[0]), int(str(r["rounds"]).split("-")[-1]) + 1)}
STATES = sorted((HERE / "practice/logs/autoplay").glob("states-1118-*.json"))
CEIL = {"SPECIALTY": (1.20, 0.06), "COMMERCIAL": (1.06, 0.03), "INSTANT": (0.96, 0.03)}   # เพดานเฉลี่ย, ส่วนเบี่ยง (จากข้อมูลวันนี้)


def rival_markup(kind, rnd, btype):
    base = {"SPECIALTY": 1.10, "COMMERCIAL": 1.00, "INSTANT": 0.92}[btype]
    if kind == "starter":
        return 1.00
    if kind == "undercut":
        return max(0.85, base - 0.05 - 0.02 * (rnd - 1))
    if kind == "hoard":
        return 0.90 if rnd >= ROUNDS - 1 else None
    if kind == "smart":                                   # เหมือนเรา ๆ: ราคาดี ค่อย ๆ ลด
        return max(0.88, base - 0.01 * (rnd - 1))
    raise ValueError(kind)


class Mem:
    def __init__(self):
        self.rounds, self.sent = [], {}


def play(seed, mix, n_rivals=20):
    rng = random.Random(seed)
    base = json.loads(rng.choice(STATES).read_text(encoding="utf-8"))
    if not base:
        return None
    st0 = base[0]
    lots = [{**l, "kg_available": l["kg_total"], "kg_remaining": l["kg_total"]} for l in st0["lots"]]
    tv = {l["lot_id"]: 100.0 + rng.uniform(-15, 25) for l in lots}
    total_bags = sum(l["kg_total"] for l in lots) // BAG
    cap = math.ceil(1.5 * total_bags / ROUNDS) * BAG
    ceil = {}                                                 # เพดานลับต่อผู้ซื้อ
    rivals = [mix[i % len(mix)] for i in range(n_rivals)]
    rival_stock = [sum(l["kg_total"] for l in lots)] * n_rivals
    mem, revenue, sold = Mem(), 0.0, 0
    avg_tv = sum(tv.values()) / len(tv)
    schedule = [("practice", i) for i in range(1, PRACTICE + 1)] + [("real", i) for i in range(1, ROUNDS + 1)]
    saved = None
    for kind, rnd in schedule:
        if kind == "real" and saved is not None:          # รอบซ้อมไม่ตัดสต็อก: คืนสต็อกก่อนเริ่มรอบจริง
            for l, (a, r) in zip(lots, saved):
                l["kg_available"], l["kg_remaining"] = a, r
            rival_stock, revenue, sold, saved = saved_rival, 0.0, 0, None
        if kind == "practice" and saved is None:
            saved, saved_rival = [(l["kg_available"], l["kg_remaining"]) for l in lots], list(rival_stock)
        rid = f"P{rnd}" if kind == "practice" else f"R{rnd}"
        fest = W.get(rnd, 1.0) > 1.0 and kind == "real"
        src = base[(rnd - 1) % len(base)]
        rfqs = []
        # ความต้องการรวมต่อรอบ = DEMAND × (ของทุกทีม ÷ รอบ) กระจายตามสัดส่วน RFQ เดิม -> มีการแย่งกันจริง
        need = DEMAND * (n_rivals + 1) * sum(l["kg_total"] for l in lots) / ROUNDS * (W.get(rnd, 1.0) if kind == "real" else 1.0)
        orig = sum(q["kg"] for q in src["rfqs"]) or 1
        for q in src["rfqs"]:
            ceil.setdefault(q["buyer"], rng.gauss(*CEIL[q["buyer_type"]]))
            boost = FEST if fest and q["buyer_type"] == "SPECIALTY" else 1.0
            kg = max(BAG, int(need * q["kg"] / orig * boost) // BAG * BAG)
            rfqs.append({**q, "rfq_id": f"{rid}-{q['rfq_id'].split('-')[1]}", "kg": kg, "bags": kg // BAG})
        state = {"market": "sim", "phase": "ROUND_OPEN", "round": {"round_id": rid, "kind": kind, "number": rnd,
                 "of": PRACTICE if kind == "practice" else ROUNDS, "counts_for_score": kind == "real"}, "lots": lots, "rfqs": rfqs,
                 "capacity_left_kg": cap, "round_capacity_kg": cap}
        ctx = strategy.Context(RULES, CAL, tv, random.Random(seed * 100 + rnd), mem, None, 200)
        offers, _ = strategy.decide(state, ctx)
        mem.sent[rid] = [{**o, "buyer_type": next(q["buyer_type"] for q in rfqs if q["rfq_id"] == o["rfq_id"]),
                                "markup": round(o["price_per_kg"] / tv[o["lot_id"]], 3)} for o in offers]
        # คู่แข่ง: แต่ละรายเสนอ ~ลิมิตของตัวเอง กระจายไปยัง RFQ ที่รับประเภทนั้น
        book = {q["rfq_id"]: [] for q in rfqs}
        for i, kind in enumerate(rivals):
            left = min(cap, rival_stock[i])
            if kind == "hoard" and rnd >= ROUNDS - 1:
                left = min(rival_stock[i], cap)
            for q in rng.sample(rfqs, len(rfqs)):
                if left < BAG:
                    break
                m = rival_markup(kind, rnd, q["buyer_type"])
                if m is None:
                    break
                kg = min(left, rng.choice([60, 120, 180]))
                book[q["rfq_id"]].append({"who": i, "kg": kg, "m": m + rng.uniform(-0.01, 0.01)})
                left -= kg
        for o in offers:
            book[o["rfq_id"]].append({"who": "me", "kg": o["kg"], "m": o["price_per_kg"] / tv[o["lot_id"]], "o": o})
        my_res, summary = [], []
        for q in rfqs:
            want, got, prices = q["kg"], 0, []
            for b in sorted(book[q["rfq_id"]], key=lambda b: b["m"]):
                if b["m"] > ceil[q["buyer"]]:
                    status, fill = "ABOVE_LIMIT", 0
                elif got >= want:
                    status, fill = "OUTBID", 0
                else:
                    fill = min(b["kg"], want - got)
                    status = "ACCEPTED" if fill == b["kg"] else "PARTIAL"
                    got += fill; prices.append(b["m"])
                if b["who"] == "me":
                    o = b["o"]
                    my_res.append({"offer_id": "x", "round_id": rid, "rfq_id": q["rfq_id"], "lot_id": o["lot_id"],
                                   "kg": o["kg"], "price_per_kg": o["price_per_kg"], "status": status, "filled_kg": fill})
                    if fill:
                        revenue += fill * o["price_per_kg"]; sold += fill
                        for l in lots:
                            if l["lot_id"] == o["lot_id"]:
                                l["kg_available"] -= fill; l["kg_remaining"] -= fill
                elif fill:
                    rival_stock[b["who"]] -= fill
            summary.append({"rfq_id": q["rfq_id"], "buyer": q["buyer"], "buyer_type": q["buyer_type"], "wanted_kg": want,
                            "bought_kg": got, "offers": len(book[q["rfq_id"]]),
                            "price_min": min(prices) * avg_tv if prices else None,      # ประมาณเป็นบาท (ตัวคูณ × TV เฉลี่ย)
                            "price_max": max(prices) * avg_tv if prices else None,
                            "price_avg": sum(prices) / len(prices) * avg_tv if prices else None})
        mem.rounds.append({"round_id": rid, "kind": kind, "my_offers": my_res, "market_summary": summary})
    tv_total = sum(l["kg_total"] * tv[l["lot_id"]] for l in lots)
    unsold = sum(l["kg_remaining"] * tv[l["lot_id"]] for l in lots)
    return (revenue + 0.8 * unsold) / tv_total, sold / sum(l["kg_total"] for l in lots)


SCEN = {
    "บอทราคาดี (คล้ายตลาดฝึก)": ["smart"],
    "starter ทั้งสนาม (1.00)": ["starter"],
    "ผสม starter + กดราคา": ["starter", "undercut"],
    "กดราคาทั้งสนาม": ["undercut"],
    "เก็งของ (เทท้ายเกม) + starter": ["hoard", "starter"],
    "ผสมทุกแบบ": ["smart", "starter", "undercut", "hoard"],
}
if __name__ == "__main__":
    games = int(sys.argv[sys.argv.index("--games") + 1]) if "--games" in sys.argv else 30
    print(f"strategy: {strategy.__file__}\n")
    for name, mix in SCEN.items():
        res = [r for r in (play(s, mix) for s in range(games)) if r]
        pr = sum(r[0] for r in res) / len(res); sell = sum(r[1] for r in res) / len(res)
        print(f"{name:32s} PR {pr:.3f} · ขายได้ {sell * 100:3.0f}% · PR ต่ำสุด {min(r[0] for r in res):.3f}")
