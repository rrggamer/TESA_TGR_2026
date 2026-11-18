"""Decide this round's offers.  <- เวอร์ชันซ้อม-พร้อมแข่ง (tests/practice) · ไฟล์แข่งจริงยังไม่ถูกแตะ

ทุกรอบ:
  1. candidates()   คู่ (ล็อต, RFQ) ที่ตัวอย่างผ่านสเปก (เผื่อ margin ได้)
  2. buyer_levels() ระดับราคารายผู้ซื้อ เรียนรู้จากผลทุกรอบใน memory (ขั้นบันได)
  3. llm_adjust()   LLM ปรับระดับได้แค่ ±LLM_MAX_ADJUST ต่อผู้ซื้อ · พลาด/ปิด = ไม่ปรับ
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
MARKUP = {"SPECIALTY": 1.00, "COMMERCIAL": 1.05, "INSTANT": 0.92}
TYPE_ORDER = ["INSTANT", "COMMERCIAL", "SPECIALTY"]     # จ่ายน้อย -> มาก (สไลด์ 18)
MIN_MARKUP, MAX_MARKUP = 0.85, 1.30   # ของค้างได้ 0.80 × TV จึงไม่ลงใกล้ 0.80
PACE = 1.3                            # เสนอ 1.3 × ส่วนเฉลี่ยต่อรอบที่เหลือ
PACE_INSTANT_ONLY = 2.0               # ล็อตที่ผ่านแค่ Instant: ขายเร็วกว่า ก่อนทุกทีมแห่ไป Instant ตอนท้ายเกม
# เผื่อ noise ของตัวอย่าง (วันนี้ตลาดฝึกไม่มี noise = 0) · พรุ่งนี้วัดจาก reveal แล้วตั้ง เช่น defect 0.3, moisture 0.1
SPEC_MARGIN = {"defect_pct": 0.0, "moisture_pct": 0.0}

# ขั้นบันไดรายผู้ซื้อ (ratio = ราคา ÷ TV ของเรา)
#   ขายได้ที่ ratio >= ระดับ -> ระดับ = ratio + UP · ABOVE_LIMIT ที่ ratio <= ระดับ -> ระดับ = ratio - DOWN
#   OUTBID ที่ ratio <= ระดับ -> ระดับ -= OUTBID_DOWN · สูงกว่าระดับแล้วพลาด (กระสอบทดลอง) -> ไม่เปลี่ยน
# เพดานลับสุ่มต่อ RFQ (พิสูจน์เกม 3) -> ระดับจะหยุดตรงที่ขายได้ราว UP/(UP+DOWN) = 75% ของครั้ง
UP, DOWN, OUTBID_DOWN = 0.01, 0.03, 0.01
PROBE_UP = 0.02                       # กระสอบทดลอง 1 ใบ/ผู้ซื้อ/รอบ ที่ ระดับ + PROBE_UP (รอบสุดท้ายไม่ทดลอง)
LLM_MAX_ADJUST = 0.03                 # LLM ปรับระดับได้ไม่เกินเท่านี้ ไม่ว่าจะตอบอะไร
# (A) กระจายความเสี่ยง: ผู้ซื้อ 1 รายได้ไม่เกินสัดส่วนนี้ของกระสอบที่เสนอในรอบ (เกม 5–6: ทุ่มให้รายเดียวแล้วโดนปฏิเสธทั้งรอบ)
BUYER_SHARE = 0.5
# (B) รอบสุดท้าย: ของค้างได้แค่ 0.80 -> ลดราคาลงจากระดับเท่านี้ และเทของที่เหลือให้ RFQ ไหนก็ได้ที่ยังรับได้
LAST_DISCOUNT = 0.04
# (C) รอบก่อนขายไม่ได้เลยสักข้อเสนอ -> ลดทุกผู้ซื้อเพิ่มอีกเท่านี้ (เพดานทั้งตลาดตกพร้อมกัน)
SHOCK_DOWN = 0.02


@dataclass
class Context:
    rules: dict                       # GET /v1/calendar -> rules
    calendar: list                    # GET /v1/calendar -> calendar
    tv: dict                          # lot_id -> our TV estimate (baht/kg)
    rng: random.Random
    memory: object = None
    llm: object = None
    seconds_left: float = 999         # time left in this round when the state was read
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
        share = avail if left <= 1 else math.ceil(avail / left * pace * cal / BAG) * BAG
        out[lot["lot_id"]] = min(avail, max(BAG, share)) if avail >= BAG else 0
    return out


def candidates(state: dict, ctx: Context) -> list[dict]:
    budget = sell_budget(state, ctx)
    out = []
    for lot in state["lots"]:
        if budget[lot["lot_id"]] < BAG:
            continue
        for rfq in state["rfqs"]:
            if spec_ok(lot["sample"], rfq["spec"], SPEC_MARGIN):
                out.append({"lot_id": lot["lot_id"], "rfq_id": rfq["rfq_id"], "buyer": rfq["buyer"],
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
        for o in rnd.get("my_offers", []):
            s, bt = sent.get((o["rfq_id"], o["lot_id"])), info.get(o["rfq_id"])
            if not s or not bt or bt[1] not in MARKUP:
                continue
            k = out.setdefault(bt[0], {"level": MARKUP[bt[1]], "type": bt[1], "hist": []})
            m, lv = s["markup"], k["level"]
            if o["status"] in ("ACCEPTED", "PARTIAL") and m >= lv - 1e-9:
                k["level"] = m + UP
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
    probe = round(safe + PROBE_UP, 3) if probing and safe + PROBE_UP <= MAX_MARKUP else None
    return round(safe, 3), probe


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
    shift = (-LAST_DISCOUNT if last else 0.0) + (-SHOCK_DOWN if shock else 0.0)
    if shift:
        ctx.notes.append(f"shift all buyers {shift:+.2f}" + (" last-round" if last else "") + (" shock" if shock else ""))
    price = {c["buyer"]: price_for(c["buyer_type"], levels.get(c["buyer"]), not last, levels,
                                   adjust.get(c["buyer"], 0.0) + shift) for c in cands}
    total_bags = sum(budget.values()) // BAG
    n_buyers = len({c["buyer"] for c in cands})
    cap = max(1, math.ceil(total_bags * BUYER_SHARE)) if n_buyers > 1 else total_bags
    given, probed, plan = {}, set(), []
    order = sorted(cands, key=lambda c: (-price[c["buyer"]][0], -c["tv"]))
    for c in order:
        safe, probe = price[c["buyer"]]
        bags = min(budget[c["lot_id"]] // BAG, rfq_left[c["rfq_id"]], cap - given.get(c["buyer"], 0))
        if bags <= 0:
            continue
        if probe is not None and c["buyer"] not in probed:
            bags, markup = 1, probe
            probed.add(c["buyer"])
        else:
            markup = safe
        plan.append({"lot_id": c["lot_id"], "rfq_id": c["rfq_id"], "bags": bags, "markup": markup})
        budget[c["lot_id"]] -= bags * BAG
        rfq_left[c["rfq_id"]] -= bags
        given[c["buyer"]] = given.get(c["buyer"], 0) + bags
    if last:                                    # (B) ของที่ยังเหลือ: เทให้ RFQ ที่ยังรับได้ (ไม่จำกัดต่อผู้ซื้อ)
        used = {(p["lot_id"], p["rfq_id"]) for p in plan}
        for c in order:
            bags = min(budget[c["lot_id"]] // BAG, rfq_left[c["rfq_id"]])
            if bags <= 0 or (c["lot_id"], c["rfq_id"]) in used:
                continue
            plan.append({"lot_id": c["lot_id"], "rfq_id": c["rfq_id"], "bags": bags, "markup": price[c["buyer"]][0]})
            budget[c["lot_id"]] -= bags * BAG
            rfq_left[c["rfq_id"]] -= bags
    ctx.notes.append("price: " + json.dumps({b: p[0] for b, p in price.items()}, ensure_ascii=False))
    return plan


# ---------- LLM: ปรับระดับได้แค่ ±LLM_MAX_ADJUST ----------
SYSTEM = """You advise the pricing of a coffee seller in a sealed-offer market.
price = markup x true value. Each RFQ has a hidden ceiling (a markup); above it the buyer answers ABOVE_LIMIT.
Buyers take the cheapest offers first; OUTBID = another seller was cheaper. Unsold beans end at 0.80.
The program already learned a markup level per buyer from results. Suggest a small change per buyer:
raise if it keeps selling and few sellers compete, lower if ABOVE_LIMIT/OUTBID or many sellers compete.
delta must be between -0.03 and 0.03. Answer only JSON; reason in one short sentence."""

SCHEMA = {"type": "object", "required": ["adjust", "reason"], "properties": {
    "adjust": {"type": "array", "items": {"type": "object", "required": ["buyer", "delta"], "properties": {
        "buyer": {"type": "string"}, "delta": {"type": "number"}}}},
    "reason": {"type": "string", "maxLength": 200}}}


def llm_adjust(state: dict, ctx: Context, cands: list[dict]) -> dict | None:
    """buyer -> delta (บีบไว้ใน ±LLM_MAX_ADJUST) · None = LLM ไม่ได้ช่วยรอบนี้ (ใช้ขั้นบันไดล้วน)"""
    time_for_llm = ctx.seconds_left - 5
    if time_for_llm < 3:
        ctx.notes.append("no time for the LLM this round")
        return None
    levels = buyer_levels(ctx.memory)
    buyers = sorted({(c["buyer"], c["buyer_type"]) for c in cands})
    ids = {f"B{i + 1}": b for i, (b, _) in enumerate(buyers)}       # ชื่อไทยกิน token -> ใช้รหัสสั้น
    market = {}
    last = ctx.memory.rounds[-1] if ctx.memory and ctx.memory.rounds else None
    for s in (last or {}).get("market_summary", []):
        m = market.setdefault(s["buyer"], [0, 0, 0])
        m[0] += s["offers"]; m[1] += s["wanted_kg"]; m[2] += s["bought_kg"]
    rows = []
    for bid, (name, btype) in zip(ids, buyers):
        lv = levels.get(name)
        mk = market.get(name)
        rows.append(f"{bid} {btype} level {price_for(btype, lv, False, levels)[0]:.2f} "
                    f"my results [{' '.join(lv['hist']) if lv else 'none'}] "
                    f"last round market: {f'{mk[0]} offers, bought {mk[2] * 100 // max(mk[1], 1)}% of wanted' if mk else 'unknown'}")
    rnd = state["round"]
    stock = sum(l["kg_remaining"] for l in state["lots"]), sum(l["kg_total"] for l in state["lots"])
    user = "\n".join([f"Round {rnd['number']} of {rnd['of']} ({rnd['kind']}), calendar: {rnd.get('calendar')}. "
                      f"Stock left {stock[0] * 100 // max(stock[1], 1)}%.", *rows,
                      "Give a delta for each buyer id."])
    answer = ctx.llm.ask_json(SYSTEM, user, SCHEMA, timeout=time_for_llm)
    if not isinstance(answer, dict) or not isinstance(answer.get("adjust"), list):
        ctx.notes.append(f"llm failed: {getattr(ctx.llm, 'last_error', None)}")
        return None
    out = {}
    for a in answer["adjust"]:
        try:
            name = ids[str(a["buyer"])]
            out[name] = round(min(max(float(a["delta"]), -LLM_MAX_ADJUST), LLM_MAX_ADJUST), 3)
        except (KeyError, TypeError, ValueError):
            ctx.notes.append(f"llm bad adjust ignored: {a}")
    ctx.notes.append(f"llm adjust: {json.dumps(out, ensure_ascii=False)} · {str(answer.get('reason'))[:150]}")
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
    adjust = llm_adjust(state, ctx, cands) if ctx.llm else None
    offers = guard(state, ctx, rule_plan(state, ctx, cands, adjust))
    return offers, "llm-adjust" if adjust is not None else "rules"
