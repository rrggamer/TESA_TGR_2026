"""Decide this round's offers.  <- เวอร์ชันซ้อม-พร้อมแข่ง (tests/practice) · ไฟล์แข่งจริงยังไม่ถูกแตะ

ทุกรอบ:
  1. candidates()   คู่ (ล็อต, RFQ) ที่ตัวอย่างผ่านสเปก (เผื่อ margin ได้)
  2. buyer_levels() ระดับราคารายผู้ซื้อ เรียนรู้จากผลทุกรอบใน memory (ขั้นบันได)
  3. llm_adjust()   LLM เลือก raise/hold/lower ต่อผู้ซื้อ + ปริมาณขาย · Python ค้านคำตอบที่ขัดข้อมูล · พลาด/ปิด = ไม่ปรับ
  4. rule_plan()    เลือกข้อเสนอ: ผู้ซื้อราคาดีได้ล็อตดีก่อน + กระสอบทดลอง 1 ใบ/ผู้ซื้อ/รอบ
  5. guard()        ตรวจทุกข้อเสนอก่อนส่ง (ของ starter เดิม)

price_per_kg = ตัวคูณ × TV ที่เราประเมิน
"""
import json
import math
import random
from dataclasses import dataclass, field

BAG = 60
# ตัวคูณตั้งต้นต่อประเภท (ใช้จนกว่าจะมีผลของผู้ซื้อรายนั้น)
# ตลาดฝึก 17 พ.ย.: Instant ×1.00 ได้ ABOVE_LIMIT แทบทุกครั้ง, ขายได้ราว ×0.92–0.95 · Commercial ขายได้ราว ×1.05–1.08
# SPECIALTY: ยังไม่เคยเห็น — สไลด์บอกว่าจ่ายมากสุด จึงไม่ตั้งต่ำกว่าระดับ Commercial ที่เรียนรู้ได้ (ดู type_floor)
MARKUP = {"SPECIALTY": 1.05, "COMMERCIAL": 1.00, "INSTANT": 0.90}   # v3: Specialty 1.10 -> 1.05 (ด่าน 2 ตลาดซื้อ Specialty ราว 1.00–1.03 ของ TV)
# 18 พ.ย. (ตลาดฝึกมีลิมิตกระสอบ/รอบ): เงินต่อช่องเสนอ = P(ขาย) × (ตัวคูณ − 0.80) สูงสุดที่ C 0.99 (ขาย 90%), I 0.90 (100%), S ~1.14
# เดิม C 1.05 ขายได้ 28% (แพ้คนถูกกว่า 48%) · Specialty ผู้ซื้อน้อยคนแย่ง ขายได้ 1.08–1.17
TYPE_ORDER = ["INSTANT", "COMMERCIAL", "SPECIALTY"]     # จ่ายน้อย -> มาก (สไลด์ 18)
MIN_MARKUP, MAX_MARKUP = 0.85, 1.30   # ของค้างได้ 0.80 × TV จึงไม่ลงใกล้ 0.80
PACE = 1.3                            # เสนอ 1.3 × ส่วนเฉลี่ยต่อรอบที่เหลือ
PACE_INSTANT_ONLY = 2.0               # ล็อตที่ผ่านแค่ Instant: ขายเร็วกว่า ก่อนทุกทีมแห่ไป Instant ตอนท้ายเกม
# (แข่งกับ 56 ทีม) starter เทของทั้งหมดในรอบสุดท้าย -> รอบนั้นคนขายล้น · เราขายมากขึ้นก่อนถึงรอบนั้น
PACE_PRE_FLOOD, PRE_FLOOD_ROUNDS = 1.5, 3     # รอบที่เหลือ 2–3 รอบ (ไม่นับรอบสุดท้าย) ใช้ PACE 1.5 (สไลด์ 18: capacity ≈ 1.5×)
# หลบเลขกลม: ทีมอื่นตั้ง 1.00 / 1.05 / 0.95 -> เราต่ำกว่านิดเดียว ผู้ซื้อเลือกคนถูกกว่าก่อน
ROUND_DODGE = 0.004
# ผู้ซื้อที่รอบก่อนได้ของไม่ครบ (คนเสนอน้อย) ได้ลำดับก่อน เท่ากับราคาสูงขึ้นเท่านี้ (ใช้แค่จัดลำดับ ไม่เปลี่ยนราคา)
UNDERSUPPLY_BONUS = 0.02
# เผื่อ noise ของตัวอย่าง (วันนี้ตลาดฝึกไม่มี noise = 0) · พรุ่งนี้วัดจาก reveal แล้วตั้ง เช่น defect 0.3, moisture 0.1
SPEC_MARGIN = {"defect_pct": 0.0, "moisture_pct": 0.0}
# ตลาดฝึก 18 พ.ย.: ค่าจริง − ตัวอย่าง คลาดถึง ±0.9 (ตำหนิ/ความชื้น) และตะแกรง ±1 -> ล็อตชิดขอบสเปก ส่งทีละ 1 กระสอบจนกว่าจะมีคนรับ
EDGE = {"defect_pct": 0.6, "moisture_pct": 0.6, "screen": 1}

# ขั้นบันไดรายผู้ซื้อ (ratio = ราคา ÷ TV ของเรา)
#   ขายได้ที่ ratio >= ระดับ -> ระดับ = ratio + UP · ABOVE_LIMIT ที่ ratio <= ระดับ -> ระดับ = ratio - DOWN
#   OUTBID ที่ ratio <= ระดับ -> ระดับ -= OUTBID_DOWN · สูงกว่าระดับแล้วพลาด (กระสอบทดลอง) -> ไม่เปลี่ยน
# เพดานลับสุ่มต่อ RFQ (พิสูจน์เกม 3) -> ระดับจะหยุดตรงที่ขายได้ราว UP/(UP+DOWN) = 75% ของครั้ง
UP, DOWN, OUTBID_DOWN = 0.01, 0.03, 0.02   # OUTBID 0.01 -> 0.02 (18 พ.ย.: แพ้คนถูกกว่าบ่อย ช่องเสนอมีจำกัด ต้องปรับเร็ว)
PROBE_UP = 0.02                       # กระสอบทดลอง 1 ใบ/ผู้ซื้อ/รอบ ที่ ระดับ + PROBE_UP (รอบสุดท้ายไม่ทดลอง)
LLM_MAX_ADJUST = 0.03                 # (ไม่ใช้แล้ว: LLM เลือกคำ ดู ACTION_DELTA) เก็บไว้ให้เทสเก่า
# (A) กระจายความเสี่ยง: ผู้ซื้อ 1 รายได้ไม่เกินสัดส่วนนี้ของกระสอบที่เสนอในรอบ (เกม 5–6: ทุ่มให้รายเดียวแล้วโดนปฏิเสธทั้งรอบ)
BUYER_SHARE = 0.5
# (B) รอบสุดท้าย: ของค้างได้แค่ 0.80 -> ลดราคาลงจากระดับเท่านี้ และเทของที่เหลือให้ RFQ ไหนก็ได้ที่ยังรับได้
LAST_DISCOUNT = 0.04
# (C) รอบก่อนขายไม่ได้เลยสักข้อเสนอ -> ลดทุกผู้ซื้อเพิ่มอีกเท่านี้ (เพดานทั้งตลาดตกพร้อมกัน)
SHOCK_DOWN = 0.02
# (18 พ.ย. วิเคราะห์ 15 เกม) ลิมิต/รอบ มีจำกัด แต่ R1–R7 เสนอแค่ ~87% ของลิมิต และ 13/15 เกมเหลือของในรอบ 8 มากกว่าลิมิตรอบ 8 -> ขายไม่ทันแน่
FILL_CAPACITY = True        # เสนอให้เต็มลิมิตทุกรอบ (ข้อเสนอที่ไม่ได้ขายไม่มีโทษ ราคาไม่ได้ลดเพราะเสนอเยอะ)
BEHIND_AT, BEHIND_DOWN = 0.7, 0.02   # ของเหลือ > 70% ของ (ลิมิต/รอบ × รอบที่เหลือ) -> ลดราคาเพิ่ม 0.02 ให้ขายทัน

# ---- v3 (หลังด่าน 2 ได้อันดับ 24: R1 และ R5 ขายไม่ได้เลย เพราะขึ้นราคาซ้อนกันหลังรอบที่ขายหมด) ----
MOVE_UP, MOVE_DOWN = 0.02, 0.04       # (1)(3) ตัวคูณหลักต่อผู้ซื้อ ขยับจากรอบก่อนได้ไม่เกิน +0.02 / -0.04 รวมทุกกฎ + LLM
LAST_MOVE_DOWN = 0.06                 #        รอบสุดท้ายลดได้ถึง -0.06 (ต้องเทของ)
MARKET_CAP = True                     # (2) ไม่เสนอแพงกว่าราคาสูงสุดที่ผู้ซื้อรายนี้ซื้อรอบก่อน (เมื่อรอบก่อนเขาได้ของครบ = มีคนแย่ง)
MARKET_CAP_MARGIN, MARKET_CAP_MAX_CUT = 0.005, 0.04   #     ต่ำกว่าราคานั้น 0.5% · ตัดลงได้ไม่เกิน 0.04 ของตัวคูณ (กันขายถูกเกิน)
PRACTICE_LADDER = (0.0, 0.04, -0.03, 0.08)
FESTIVAL_HOLD = True                  # (5) ก่อนเทศกาล: ล็อตที่เข้าสเปก Specialty ขายให้ Specialty เท่านั้น (เก็บไว้ขายช่วงเทศกาล/หลังเทศกาลได้อีก)   # (4) รอบซ้อม (ไม่นับคะแนน ไม่ตัดสต็อก): เสนอหลายราคา กระสอบละราคา เพื่อหาเพดาน


@dataclass
class Context:
    rules: dict                       # GET /v1/calendar -> rules
    calendar: list                    # GET /v1/calendar -> calendar
    tv: dict                          # lot_id -> our TV estimate (baht/kg)
    rng: random.Random
    memory: object = None
    llm: object = None
    seconds_left: float = 999         # time left in this round when the state was read
    volume: float = 1.0               # ตัวคูณปริมาณขายของรอบ (LLM เลือก sell_more/normal/hold_back)
    notes: list = field(default_factory=list)


# ---------- สเปก ----------
def spec_ok(sample: dict, spec: dict, margin: dict | None = None) -> bool:
    """ผ่านสเปกไหม · margin = เผื่อ noise (ใช้ตอนเลือก candidates) · guard ใช้แบบไม่เผื่อ"""
    m = margin or {}
    return (sample["defect_pct"] + m.get("defect_pct", 0) <= spec["max_defect_pct"]
            and sample["moisture_pct"] + m.get("moisture_pct", 0) <= spec["max_moisture_pct"]
            and (spec["min_screen"] is None or sample["screen"] >= spec["min_screen"]))


# ---------- ปริมาณที่ขายต่อรอบ ----------
def rounds_left(state: dict, ctx: Context) -> int:
    rnd, total = state["round"], ctx.rules["timeline"]["rounds"]
    return total if rnd["kind"] == "practice" else total - rnd["number"] + 1


def calendar_weight(state: dict, ctx: Context) -> float:
    """demand_weight ของรอบนี้ ÷ ค่าเฉลี่ยของรอบที่เหลือ (ปฏิทินจาก /v1/calendar) · อ่านไม่ได้ = 1"""
    try:
        rnd = state["round"]
        if rnd["kind"] == "practice":
            return 1.0
        weight = {}
        for row in ctx.calendar or []:
            a, _, b = str(row["rounds"]).partition("-")
            for n in range(int(a), int(b or a) + 1):
                weight[n] = float(row.get("demand_weight") or 1.0)
        now, total = rnd["number"], ctx.rules["timeline"]["rounds"]
        rest = [weight.get(n, 1.0) for n in range(now, total + 1)]
        return min(max(weight.get(now, 1.0) / (sum(rest) / len(rest)), 0.5), 2.0)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return 1.0


def calendar_weights(ctx: Context) -> dict:
    """รอบ -> demand_weight จากปฏิทิน (ไม่มีข้อมูล = 1.0)"""
    weight = {}
    try:
        for row in ctx.calendar or []:
            a, _, b = str(row["rounds"]).partition("-")
            for n in range(int(a), int(b or a) + 1):
                weight[n] = float(row.get("demand_weight") or 1.0)
    except (KeyError, TypeError, ValueError):
        return {}
    return weight


def effective_rounds_left(state: dict, ctx: Context) -> float:
    """(6) รอบที่เหลือถ่วงด้วยความต้องการ: ช่วงใกล้ปิดตลาดที่ความต้องการต่ำนับเป็นไม่ถึง 1 รอบ"""
    if state["round"]["kind"] == "practice":
        return float(rounds_left(state, ctx))
    w, now, total = calendar_weights(ctx), state["round"]["number"], ctx.rules["timeline"]["rounds"]
    return sum(min(w.get(n, 1.0), 1.0) for n in range(now, total + 1))


def festival_rounds(ctx: Context) -> set:
    """(5) รอบเทศกาลจากปฏิทิน: ป้ายมีคำว่า เทศกาล หรือ specialty_share สูงกว่ารอบอื่น"""
    rows, out = ctx.calendar or [], set()
    try:
        shares = [float(r["specialty_share"]) for r in rows if r.get("specialty_share") is not None]
        for row in rows:
            sh = row.get("specialty_share")
            if "เทศกาล" in str(row.get("label", "")) or (sh is not None and len(shares) > 1 and float(sh) == max(shares) > min(shares)):
                a, _, b = str(row["rounds"]).partition("-")
                out.update(range(int(a), int(b or a) + 1))
    except (KeyError, TypeError, ValueError):
        return set()
    return out


def pre_festival(state: dict, ctx: Context) -> bool:
    rnd, fest = state["round"], festival_rounds(ctx)
    return bool(FESTIVAL_HOLD and fest and rnd["kind"] != "practice" and rnd["number"] < min(fest))


def specialty_grade(lot: dict, rfqs: list[dict]) -> bool:
    """ล็อตนี้เข้าสเปกผู้ซื้อ Specialty สักรายในรอบนี้"""
    return any(r["buyer_type"] == "SPECIALTY" and spec_ok(lot["sample"], r["spec"], SPEC_MARGIN) for r in rfqs)


def instant_only(lot: dict, rfqs: list[dict]) -> bool:
    """ล็อตนี้ผ่านสเปกแค่ RFQ ของ Instant (รอบนี้)"""
    types = {r["buyer_type"] for r in rfqs if spec_ok(lot["sample"], r["spec"], SPEC_MARGIN)}
    return types == {"INSTANT"}


def sell_budget(state: dict, ctx: Context) -> dict[str, int]:
    """แต่ละล็อตเสนอขายกี่ kg รอบนี้: กระจายตามรอบที่เหลือ × ปฏิทิน · ล็อตที่ผ่านแค่ Instant ขายเร็วขึ้น"""
    left, cal = rounds_left(state, ctx), calendar_weight(state, ctx)
    out = {}
    for lot in state["lots"]:
        avail = lot["kg_available"] // BAG * BAG
        pace = PACE_INSTANT_ONLY if instant_only(lot, state["rfqs"]) else PACE
        if 1 < left <= PRE_FLOOD_ROUNDS and calendar_weights(ctx).get(state["round"]["number"], 1.0) >= 1.0:
            pace = max(pace, PACE_PRE_FLOOD)        # (6) ช่วงความต้องการต่ำ (ใกล้ปิดตลาด) ไม่เร่ง: ควรขายเสร็จก่อนหน้านั้น
        share = avail if left <= 1 else math.ceil(avail / left * pace * cal * ctx.volume / BAG) * BAG
        out[lot["lot_id"]] = min(avail, max(BAG, share)) if avail >= BAG else 0
    room = state.get("capacity_left_kg")
    if FILL_CAPACITY and room is not None and left > 1:          # เติมให้เต็มลิมิต: เพิ่มทีละกระสอบ ล็อตที่ของเหลือมากก่อน
        while sum(out.values()) < room:
            spare = [l for l in state["lots"] if l["kg_available"] // BAG * BAG - out[l["lot_id"]] >= BAG]
            if not spare:
                break
            lot = max(spare, key=lambda l: l["kg_available"] - out[l["lot_id"]])
            out[lot["lot_id"]] += BAG
    return out


def behind_schedule(state: dict, ctx: Context) -> bool:
    """ของเหลือมากเกินกว่าที่ลิมิตรอบที่เหลือจะขายทัน (คิดว่าขายได้ราว 70% ของที่เสนอ)"""
    cap, left = state.get("round_capacity_kg"), rounds_left(state, ctx)
    if not cap or left <= 1:
        return False
    stock = sum(l["kg_remaining"] for l in state["lots"])
    return stock > BEHIND_AT * cap * effective_rounds_left(state, ctx)     # (6) ถ่วงด้วยความต้องการที่เหลือ


def stricter_or_same(a: dict, b: dict) -> bool:
    """สเปก a เข้มเท่าหรือมากกว่า b ทุกข้อ (ล็อตไม่ผ่าน b -> ไม่ผ่าน a แน่)"""
    return (a["max_defect_pct"] <= b["max_defect_pct"] and a["max_moisture_pct"] <= b["max_moisture_pct"]
            and (a.get("min_screen") or 0) >= (b.get("min_screen") or 0))


def spec_history(state: dict, memory) -> tuple[dict, set]:
    """จากผลรอบก่อน: lot -> [สเปกที่ล็อตนี้โดน SPEC_MISMATCH] · {(lot, buyer) ที่เคยขายได้}
    memory ไม่เก็บสเปก จึงใช้สเปกของผู้ซื้อคนเดิมในรอบนี้ (ผู้ซื้อแต่ละรายใช้สเปกเดิมทั้งเกม)"""
    failed, passed = {}, set()
    if memory is None:
        return failed, passed
    spec_of = {r["buyer"]: r["spec"] for r in state["rfqs"]}
    for rnd in memory.rounds:
        buyer = {x["rfq_id"]: x["buyer"] for x in rnd.get("market_summary", [])}
        for o in rnd.get("my_offers", []):
            b = buyer.get(o["rfq_id"])
            if o["status"] == "SPEC_MISMATCH":
                if b in spec_of:
                    failed.setdefault(o["lot_id"], []).append(spec_of[b])
                failed.setdefault(o["lot_id"], []).append({"buyer": b})
            elif o["status"] in ("ACCEPTED", "PARTIAL", "OUTBID", "ABOVE_LIMIT") and b:
                passed.add((o["lot_id"], b))      # ไม่โดน SPEC_MISMATCH = คุณภาพจริงผ่านสเปกผู้ซื้อนี้
    return failed, passed


def near_edge(sample: dict, spec: dict) -> bool:
    return (spec["max_defect_pct"] - sample["defect_pct"] < EDGE["defect_pct"]
            or spec["max_moisture_pct"] - sample["moisture_pct"] < EDGE["moisture_pct"]
            or (spec.get("min_screen") is not None and sample["screen"] - spec["min_screen"] < EDGE["screen"]))


def candidates(state: dict, ctx: Context) -> list[dict]:
    budget = sell_budget(state, ctx)
    failed, passed = spec_history(state, ctx.memory)
    hold = pre_festival(state, ctx) and not behind_schedule(state, ctx)
    if hold:
        ctx.notes.append("festival ahead: specialty-grade lots offered to SPECIALTY only")
    out = []
    for lot in state["lots"]:
        if budget[lot["lot_id"]] < BAG:
            continue
        keep = hold and specialty_grade(lot, state["rfqs"])           # (5) เก็บไว้ขายช่วงเทศกาล
        for rfq in state["rfqs"]:
            if keep and rfq["buyer_type"] != "SPECIALTY":
                continue
            bad = failed.get(lot["lot_id"], [])
            if any(f.get("buyer") == rfq["buyer"] or ("max_defect_pct" in f and stricter_or_same(rfq["spec"], f))
                   for f in bad):
                continue                                   # เคยโดน SPEC_MISMATCH กับสเปกนี้ (หรือหลวมกว่า) แล้ว
            if spec_ok(lot["sample"], rfq["spec"], SPEC_MARGIN):
                risky = near_edge(lot["sample"], rfq["spec"]) and (lot["lot_id"], rfq["buyer"]) not in passed
                out.append({"risky": risky, "lot_id": lot["lot_id"], "rfq_id": rfq["rfq_id"], "buyer": rfq["buyer"],
                            "buyer_type": rfq["buyer_type"], "rfq_bags": rfq["bags"],
                            "max_bags": min(budget[lot["lot_id"]], rfq["kg"]) // BAG,
                            "tv": ctx.tv[lot["lot_id"]]})
    # สุ่มลำดับ (RFQ ใหญ่มีโอกาสมาก่อน) ทุกทีมจะได้ไม่แห่ไป RFQ แรกพร้อมกัน · ใช้ตัดสินตอนราคาเท่ากัน
    keyed = [(ctx.rng.random() ** (1 / c["rfq_bags"]), c) for c in out]
    return [c for _, c in sorted(keyed, key=lambda x: -x[0])]


# ---------- ขั้นบันไดราคารายผู้ซื้อ ----------
def buyer_levels(memory) -> dict:
    """buyer -> {"level", "type", "hist": ผลล่าสุดสั้นๆ} จากผลทุกรอบของเกมนี้ ตามลำดับเวลา"""
    out = {}
    if memory is None:
        return out
    for rnd in memory.rounds:
        info = {s["rfq_id"]: (s["buyer"], s["buyer_type"]) for s in rnd.get("market_summary", [])}
        sent = {(o["rfq_id"], o["lot_id"]): o for o in memory.sent.get(rnd["round_id"], [])}
        practice = rnd.get("kind") == "practice"
        start = {b: v["level"] for b, v in out.items()}             # ระดับต้นรอบ: ขายได้หลายกระสอบในรอบเดียวก็ขึ้นไม่เกิน MOVE_UP
        for o in rnd.get("my_offers", []):
            s, bt = sent.get((o["rfq_id"], o["lot_id"])), info.get(o["rfq_id"])
            if not s or not bt or bt[1] not in MARKUP:
                continue
            k = out.setdefault(bt[0], {"level": MARKUP[bt[1]], "type": bt[1], "hist": []})
            m, lv = s["markup"], k["level"]
            if o["status"] in ("ACCEPTED", "PARTIAL") and m >= lv - 1e-9:
                k["level"] = max(lv, m) if practice else max(lv, min(m + UP, start.get(bt[0], MARKUP[bt[1]]) + MOVE_UP))
            elif o["status"] == "ABOVE_LIMIT" and m <= lv + 1e-9:
                k["level"] = m - DOWN
            elif o["status"] == "OUTBID" and m <= lv + 1e-9:
                k["level"] = lv - OUTBID_DOWN
            k["level"] = round(min(max(k["level"], MIN_MARKUP), MAX_MARKUP), 3)
            k["hist"] = (k["hist"] + [f"{o['status'][:5]}@{m:.2f}"])[-4:]
    return out


def type_floor(buyer_type: str, levels: dict) -> float:
    """ตัวคูณตั้งต้นของผู้ซื้อที่ยังไม่มีข้อมูล: ไม่ต่ำกว่าระดับสูงสุดของประเภทที่จ่ายน้อยกว่า (สไลด์: S > C > I)"""
    below = TYPE_ORDER[:TYPE_ORDER.index(buyer_type)] if buyer_type in TYPE_ORDER else []
    learned = [v["level"] for v in levels.values() if v["type"] in below]
    return max([MARKUP.get(buyer_type, 1.0)] + [MARKUP[t] for t in below] + learned)


def price_for(buyer_type: str, level: dict | None, probing: bool, levels: dict | None = None,
              adjust: float = 0.0) -> tuple[float, float | None]:
    """(ตัวคูณราคาหลัก, ตัวคูณกระสอบทดลอง หรือ None)"""
    base = level["level"] if level else type_floor(buyer_type, levels or {})
    safe = min(max(base + adjust, MIN_MARKUP), MAX_MARKUP)
    if abs(safe * 20 - round(safe * 20)) < 1e-6:          # ลงท้าย .00 / .05 = เลขที่ทีมอื่นชอบตั้ง
        safe = max(safe - ROUND_DODGE, MIN_MARKUP)
    probe = round(safe + PROBE_UP, 3) if probing and safe + PROBE_UP <= MAX_MARKUP else None
    return round(safe, 3), probe


def last_round_view(memory) -> tuple[dict, dict, bool]:
    """รอบล่าสุดที่มีผล: buyer -> ตัวคูณหลัก (ต่ำสุด) ที่เราเสนอ · buyer -> ตัวคูณสูงสุดที่เราขายได้ · เป็นรอบซ้อมไหม"""
    if memory is None or not memory.rounds:
        return {}, {}, False
    rnd = memory.rounds[-1]
    buyer = {x["rfq_id"]: x["buyer"] for x in rnd.get("market_summary", [])}
    sent = {(o["rfq_id"], o["lot_id"]): o for o in memory.sent.get(rnd["round_id"], [])}
    offered, best = {}, {}
    for o in rnd.get("my_offers", []):
        b, s = buyer.get(o["rfq_id"]), sent.get((o["rfq_id"], o["lot_id"]))
        if not b or not s:
            continue
        offered[b] = min(offered.get(b, 9.0), s["markup"])
        if o["status"] in ("ACCEPTED", "PARTIAL"):
            best[b] = max(best.get(b, 0.0), s["markup"])
    return offered, best, rnd.get("kind") == "practice"


def move_limits(memory, last: bool) -> dict:
    """(1)(3) buyer -> (ต่ำสุด, สูงสุด) ของตัวคูณหลักรอบนี้ เทียบกับรอบก่อน · หลังรอบซ้อมใช้ราคาสูงสุดที่ขายได้เป็นฐานขาขึ้น"""
    offered, best, practice = last_round_view(memory)
    down = LAST_MOVE_DOWN if last else MOVE_DOWN
    return {b: (m - down, max(m + MOVE_UP, best.get(b, 0.0) if practice else 0.0)) for b, m in offered.items()}


def market_price_cap(memory) -> dict:
    """(2) buyer -> ราคาสูงสุด (บาท/kg) ที่ผู้ซื้อรายนี้ซื้อรอบก่อน เฉพาะเมื่อรอบก่อนเขาได้ของครบ (มีคนแย่ง ราคาสูงกว่านี้มักแพ้)"""
    if memory is None or not memory.rounds:
        return {}
    agg = {}
    for x in memory.rounds[-1].get("market_summary", []):
        a = agg.setdefault(x["buyer"], [0, 0, None])
        a[0] += x["wanted_kg"]; a[1] += x["bought_kg"]
        if x.get("price_max") is not None:
            a[2] = max(a[2] or 0.0, x["price_max"])
    return {b: pmax for b, (want, got, pmax) in agg.items() if pmax and want and got >= want}


def undersupplied_buyers(memory) -> set:
    """ผู้ซื้อที่รอบล่าสุดได้ของไม่ถึง 90% ของที่ต้องการ (คนแย่งน้อย) จาก market_summary"""
    if memory is None or not memory.rounds:
        return set()
    got = {}
    for x in memory.rounds[-1].get("market_summary", []):
        g = got.setdefault(x["buyer"], [0, 0])
        g[0] += x["bought_kg"]; g[1] += x["wanted_kg"]
    return {b for b, (bought, wanted) in got.items() if wanted and bought < 0.9 * wanted}


def last_round_sold_nothing(memory) -> bool:
    """รอบล่าสุดที่มีผล: เราส่งข้อเสนอแล้วไม่มีข้อไหนขายได้เลย"""
    if memory is None or not memory.rounds:
        return False
    mine = memory.rounds[-1].get("my_offers", [])
    return bool(mine) and not any(o["status"] in ("ACCEPTED", "PARTIAL") for o in mine)


def rule_plan(state: dict, ctx: Context, cands: list[dict], adjust: dict | None = None) -> list[dict]:
    """ราคาจากขั้นบันได (+ที่ LLM ปรับ) · ผู้ซื้อราคาดีได้ล็อตดีก่อน · ผู้ซื้อละ 1 กระสอบทดลองต่อรอบ
    (A) ผู้ซื้อละไม่เกิน BUYER_SHARE ของรอบ · (B) รอบสุดท้ายลดราคา + เทที่เหลือ · (C) รอบก่อนขายไม่ได้เลย -> ลดเพิ่ม"""
    budget = sell_budget(state, ctx)
    rfq_left = {r["rfq_id"]: r["bags"] for r in state["rfqs"]}
    levels = buyer_levels(ctx.memory)
    last = rounds_left(state, ctx) <= 1
    adjust = dict(adjust or {})
    shock = last_round_sold_nothing(ctx.memory)
    behind = behind_schedule(state, ctx)
    shift = (-LAST_DISCOUNT if last else 0.0) + (-SHOCK_DOWN if shock else 0.0) + (-BEHIND_DOWN if behind else 0.0)
    if shift:
        ctx.notes.append(f"shift all buyers {shift:+.2f}" + (" last-round" if last else "") + (" shock" if shock else "")
                         + (" behind" if behind else ""))
    practice = state["round"].get("kind") == "practice"
    price = {c["buyer"]: price_for(c["buyer_type"], levels.get(c["buyer"]), not last and not practice, levels,
                                   adjust.get(c["buyer"], 0.0) + shift) for c in cands}
    for b, (lo, hi) in move_limits(ctx.memory, last).items():         # (1)(3) รวมทุกกฎ + LLM แล้วขยับไม่เกินนี้
        if b in price:
            safe, probe = price[b]
            new = round(min(max(safe, lo, MIN_MARKUP), hi, MAX_MARKUP), 3)
            if new != safe:
                ctx.notes.append(f"move cap {b[:12]} {safe:.3f}->{new:.3f}")
                price[b] = (new, round(new + PROBE_UP, 3) if probe is not None else None)
    mcap = market_price_cap(ctx.memory) if MARKET_CAP and not practice else {}

    def capped(c, markup):                                                # (2) ไม่แพงกว่าราคาที่ตลาดเพิ่งซื้อ
        pmax, tv = mcap.get(c["buyer"]), ctx.tv.get(c["lot_id"])
        if not pmax or not tv:
            return markup
        cap = math.floor(pmax * (1 - MARKET_CAP_MARGIN) / tv * 1000) / 1000      # ปัดลง: ปัดขึ้นจะเกินราคาตลาด
        return round(max(min(markup, cap), markup - MARKET_CAP_MAX_CUT, MIN_MARKUP), 3)
    total_bags = sum(budget.values()) // BAG
    if state.get("capacity_left_kg") is not None:     # สไลด์ 18: เสนอรวมได้ราว 1.5 × ส่วนเฉลี่ยต่อรอบ -> แบ่ง 50% จากที่ตลาดให้เสนอจริง
        total_bags = min(total_bags, state["capacity_left_kg"] // BAG)
    n_buyers = len({c["buyer"] for c in cands})
    cap = max(1, math.ceil(total_bags * BUYER_SHARE)) if n_buyers > 1 else total_bags
    given, probed, plan = {}, set(), []
    short = undersupplied_buyers(ctx.memory)
    order = sorted(cands, key=lambda c: (-(price[c["buyer"]][0] + (UNDERSUPPLY_BONUS if c["buyer"] in short else 0)), -c["tv"]))
    for c in order:
        safe, probe = price[c["buyer"]]
        bags = min(budget[c["lot_id"]] // BAG, rfq_left[c["rfq_id"]], cap - given.get(c["buyer"], 0))
        if c.get("risky"):
            if (c["lot_id"], "risky") in given:
                continue
            bags = min(bags, 1)                       # ชิดขอบสเปก: ลอง 1 กระสอบต่อล็อตต่อรอบ
            given[(c["lot_id"], "risky")] = 1
        if bags <= 0:
            continue
        if practice:                                   # (4) รอบซ้อม: กระสอบละราคา ไล่ตาม PRACTICE_LADDER ต่อผู้ซื้อ
            k = given.get((c["buyer"], "ladder"), 0)
            bags, markup = 1, round(min(max(safe + PRACTICE_LADDER[k % len(PRACTICE_LADDER)], MIN_MARKUP), MAX_MARKUP), 3)
            given[(c["buyer"], "ladder")] = k + 1
        elif probe is not None and c["buyer"] not in probed:
            bags, markup = 1, capped(c, probe)
            probed.add(c["buyer"])
        else:
            markup = capped(c, safe)
        plan.append({"lot_id": c["lot_id"], "rfq_id": c["rfq_id"], "bags": bags, "markup": markup})
        budget[c["lot_id"]] -= bags * BAG
        rfq_left[c["rfq_id"]] -= bags
        given[c["buyer"]] = given.get(c["buyer"], 0) + bags
    if last:                                    # (B) ของที่ยังเหลือ: เทให้ RFQ ที่ยังรับได้ (ไม่จำกัดต่อผู้ซื้อ)
        used = {(p["lot_id"], p["rfq_id"]) for p in plan}
        for c in order:
            bags = min(budget[c["lot_id"]] // BAG, rfq_left[c["rfq_id"]])
            if bags <= 0 or (c["lot_id"], c["rfq_id"]) in used or c.get("risky"):
                continue
            plan.append({"lot_id": c["lot_id"], "rfq_id": c["rfq_id"], "bags": bags, "markup": capped(c, price[c["buyer"]][0])})
            budget[c["lot_id"]] -= bags * BAG
            rfq_left[c["rfq_id"]] -= bags
    ctx.notes.append("price: " + json.dumps({b: p[0] for b, p in price.items()}, ensure_ascii=False))
    return plan


# ---------- LLM: เลือก raise/hold/lower ต่อผู้ซื้อ + ปริมาณขายของรอบ (Python สรุปตัวเลข + ค้านคำตอบที่ขัดข้อมูล) ----------
# บทเรียน (17 พ.ย., qwen3:8b): ให้ตอบเป็นตัวเลข delta -> ตอบสลับ +,-,+,- ตามลำดับแถว ไม่อ่านข้อมูล, ดันราคาขึ้นจน ABOVE_LIMIT 42%
# จึงให้ "เลือกคำ" + เขียนหลักฐานก่อนตอบ + เกณฑ์ชัดใน system (แบบ tesa_prac: บอกขอบเขต/ผลที่โดนปฏิเสธให้ชัด)
ACTION_DELTA = {"raise": 0.01, "hold": 0.0, "lower": -0.02}
VOLUME_FACTOR = {"sell_more": 1.2, "normal": 1.0, "hold_back": 0.85}

SYSTEM = """You help a coffee seller set prices in a sealed-offer market. Python already did all the math.
For each buyer you get counts of my offers last rounds:
  sold = buyer bought at my price. too_expensive = ABOVE_LIMIT (price above the buyer's hidden ceiling).
  lost_to_cheaper = OUTBID (another seller was cheaper).
Rules for action:
  raise  only if sold >= 2 and too_expensive == 0 and lost_to_cheaper <= 1
  lower  if too_expensive >= 1 or lost_to_cheaper >= 2
  hold   otherwise, and when there is no data
Volume for this round: sell_more if demand_vs_average > 1.1 or stock is high with few rounds left;
hold_back only if most buyers are 'raise' and many rounds are left; otherwise normal.
Write the evidence (the numbers) before the action. Answer only JSON."""

SCHEMA = {"type": "object", "required": ["buyers", "volume", "reason"], "properties": {
    "buyers": {"type": "array", "items": {"type": "object", "required": ["id", "evidence", "action"], "properties": {
        "id": {"type": "string"}, "evidence": {"type": "string", "maxLength": 80},
        "action": {"type": "string", "enum": list(ACTION_DELTA)}}}},
    "volume": {"type": "string", "enum": list(VOLUME_FACTOR)},
    "reason": {"type": "string", "maxLength": 150}}}


def buyer_stats(memory, last_n: int = 3) -> dict:
    """buyer -> {"sold", "too_expensive", "lost_to_cheaper"} ของเราใน last_n รอบล่าสุด"""
    out = {}
    if memory is None:
        return out
    for rnd in memory.rounds[-last_n:]:
        info = {x["rfq_id"]: x["buyer"] for x in rnd.get("market_summary", [])}
        for o in rnd.get("my_offers", []):
            b = info.get(o["rfq_id"])
            if b is None:
                continue
            k = out.setdefault(b, {"sold": 0, "too_expensive": 0, "lost_to_cheaper": 0})
            if o["status"] in ("ACCEPTED", "PARTIAL"):
                k["sold"] += 1
            elif o["status"] == "ABOVE_LIMIT":
                k["too_expensive"] += 1
            elif o["status"] == "OUTBID":
                k["lost_to_cheaper"] += 1
    return out


def python_veto(action: str, st: dict | None) -> str:
    """Python คุมตัวเลข: คำตอบที่ขัดกับข้อมูลชัดๆ ถูกเปลี่ยน (สไลด์ 18: LLM เสนอ · Python ตรวจ)"""
    if st is None:
        return "hold" if action == "raise" else action           # ไม่มีข้อมูล ห้ามขึ้น
    if action == "raise" and (st["too_expensive"] > 0 or st["lost_to_cheaper"] >= 2):
        return "hold"
    if action == "lower" and st["sold"] >= 2 and st["too_expensive"] == 0 and st["lost_to_cheaper"] == 0:
        return "hold"
    return action


def llm_adjust(state: dict, ctx: Context, cands: list[dict]) -> dict | None:
    """buyer -> delta (จาก raise/hold/lower) · ตั้ง ctx.volume · None = LLM ไม่ได้ช่วยรอบนี้ (ใช้ขั้นบันไดล้วน)"""
    time_for_llm = ctx.seconds_left - 5
    if time_for_llm < 3:
        ctx.notes.append("no time for the LLM this round")
        return None
    stats, levels = buyer_stats(ctx.memory), buyer_levels(ctx.memory)
    buyers = sorted({(c["buyer"], c["buyer_type"]) for c in cands})
    ids = {f"B{i + 1}": b for i, (b, _) in enumerate(buyers)}          # ชื่อไทยกิน token -> ใช้รหัสสั้น
    market = {}
    last = ctx.memory.rounds[-1] if ctx.memory and ctx.memory.rounds else None
    for x in (last or {}).get("market_summary", []):
        m = market.setdefault(x["buyer"], [0, 0, 0])
        m[0] += x["offers"]; m[1] += x["wanted_kg"]; m[2] += x["bought_kg"]
    rows = []
    for bid, (name, btype) in zip(ids, buyers):
        st, mk = stats.get(name), market.get(name)
        mine = (f"sold={st['sold']} too_expensive={st['too_expensive']} lost_to_cheaper={st['lost_to_cheaper']}"
                if st else "no data yet")
        rows.append(f"{bid} ({btype}, price level {price_for(btype, levels.get(name), False, levels)[0]:.2f}): {mine}"
                    + (f"; market last round: {mk[0]} sellers, buyer got {mk[2] * 100 // max(mk[1], 1)}% of wanted" if mk else ""))
    rnd = state["round"]
    stock = sum(l["kg_remaining"] for l in state["lots"]), sum(l["kg_total"] for l in state["lots"])
    user = "\n".join([f"Round {rnd['number']} of {rnd['of']}, rounds left {rounds_left(state, ctx)}, "
                      f"demand_vs_average {calendar_weight(state, ctx):.2f} (from the market calendar), my stock left {stock[0] * 100 // max(stock[1], 1)}%.",
                      "My results in the last 3 rounds per buyer:", *rows,
                      "Give an action for every buyer id and the volume for this round."])
    answer = ctx.llm.ask_json(SYSTEM, user, SCHEMA, timeout=time_for_llm)
    if not isinstance(answer, dict) or not isinstance(answer.get("buyers"), list):
        ctx.notes.append(f"llm failed: {getattr(ctx.llm, 'last_error', None)}")
        return None
    out, said = {}, []
    for a in answer["buyers"]:
        try:
            name, act = ids[str(a["id"])], str(a["action"])
            if act not in ACTION_DELTA:
                raise ValueError(act)
            final = python_veto(act, stats.get(name))
            out[name] = ACTION_DELTA[final]
            said.append(f"{a['id']}:{act}" + (f"->{final}" if final != act else ""))
        except (KeyError, TypeError, ValueError):
            ctx.notes.append(f"llm bad answer ignored: {a}")
    if not out:                                          # ตอบมาแต่ใช้ไม่ได้สักข้อ = เหมือน LLM ล่ม
        ctx.notes.append("llm failed: no usable buyer action")
        return None
    vol = str(answer.get("volume"))
    behind = stock[0] / max(stock[1], 1) >= rounds_left(state, ctx) / ctx.rules["timeline"]["rounds"] - 0.05
    if vol == "hold_back" and behind:                    # ของเหลือมากกว่าที่ควร -> ห้ามชะลอการขาย (ขายให้หมดสำคัญสุด)
        said.append("volume:hold_back->normal")
        vol = "normal"
    ctx.volume = VOLUME_FACTOR.get(vol, 1.0) if rounds_left(state, ctx) > 1 else 1.0
    ctx.notes.append(f"llm: {' '.join(said)} volume={vol} · {str(answer.get('reason'))[:120]}")
    return out


# ---------- ตรวจก่อนส่ง (ของ starter เดิม) ----------
def guard(state: dict, ctx: Context, plan: list[dict]) -> list[dict]:
    """Turn a plan into offers the market will accept. Never trust the LLM's numbers as they are."""
    lots = {l["lot_id"]: l for l in state["lots"]}
    rfqs = {r["rfq_id"]: r for r in state["rfqs"]}
    room = state["capacity_left_kg"] if state["capacity_left_kg"] is not None else 10**9
    lot_left = {i: l["kg_available"] for i, l in lots.items()}
    rfq_left = {i: r["kg"] for i, r in rfqs.items()}
    offers, seen = [], set()
    for p in plan:
        lot, rfq = lots.get(p["lot_id"]), rfqs.get(p["rfq_id"])
        if lot is None or rfq is None or (p["lot_id"], p["rfq_id"]) in seen or not spec_ok(lot["sample"], rfq["spec"]):
            ctx.notes.append(f"guard dropped {p}")
            continue
        kg = min(p["bags"] * BAG, lot_left[lot["lot_id"]], rfq_left[rfq["rfq_id"]], room) // BAG * BAG
        if kg < BAG:
            continue
        markup = min(max(p["markup"], MIN_MARKUP), MAX_MARKUP)
        offers.append({"rfq_id": rfq["rfq_id"], "lot_id": lot["lot_id"], "kg": kg,
                       "price_per_kg": round(markup * ctx.tv[lot["lot_id"]], 2)})
        seen.add((p["lot_id"], p["rfq_id"]))
        lot_left[lot["lot_id"]] -= kg
        rfq_left[rfq["rfq_id"]] -= kg
        room -= kg
    return offers


def decide(state: dict, ctx: Context) -> tuple[list[dict], str]:
    """Returns (offers, source): "llm-adjust" (LLM ปรับระดับ) หรือ "rules" (ขั้นบันไดล้วน)"""
    cands = candidates(state, ctx)
    if not cands:
        return [], "none"
    adjust = None
    if ctx.llm:
        try:
            adjust = llm_adjust(state, ctx, cands)
        except Exception as e:                           # noqa: BLE001 — LLM พังแบบไหนก็ขายด้วยกฎต่อ
            ctx.notes.append(f"llm failed: {type(e).__name__}: {e}"[:200])
            ctx.volume, adjust = 1.0, None
    offers = guard(state, ctx, rule_plan(state, ctx, cands, adjust))
    return offers, "llm-adjust" if adjust is not None else "rules"
