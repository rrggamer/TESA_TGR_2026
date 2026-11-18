"""Decide this round's offers.  <- ไฟล์หลักที่ทีมควรแก้

Every round:
  1. candidates()  every (lot, RFQ) pair whose sample meets the RFQ spec
  2. llm_plan()    the LLM picks offers from those candidates: which, how many bags, what markup
  3. rule_plan()   used instead when the LLM is off, too slow, or answers nonsense
  4. guard()       every offer, from the LLM or the rules, is checked before it is sent

price_per_kg = markup × our TV estimate. A markup of 1.00 asks exactly what we think the lot is worth.
"""
import json
import math
import random
from dataclasses import dataclass, field

BAG = 60
# Markup per buyer type. The starter asks 1.00 × TV from everyone.
# TODO: buyer types do not pay the same (Specialty pays the most, Instant the least), and these never change.
#       Price each type differently and move the markups after each round's results
#       (ACCEPTED -> ask a bit more, OUTBID / ABOVE_LIMIT -> ask less) — จุดนี้ปรับแล้วได้ผลมากที่สุด
# ซ้อมรอบ 1 (17 พ.ย.): เกม 1 Instant x1.00 ได้ ABOVE_LIMIT 16/17, Commercial x1.00 ได้ ACCEPTED 3/4
MARKUP = {"SPECIALTY": 1.00, "COMMERCIAL": 1.05, "INSTANT": 0.92}
MIN_MARKUP, MAX_MARKUP = 0.85, 1.30   # unsold beans go to salvage at 0.80 × TV, so never go near that
PACE = 1.3                            # offer 1.3 × an even share of each lot per remaining round


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


def spec_ok(sample: dict, spec: dict) -> bool:
    # TODO: the sample is noisy. A lot just inside the limit may really be outside -> SPEC_MISMATCH.
    #       ลองเว้น margin เล็กน้อยสำหรับล็อตที่ค่าใกล้ขอบสเปก
    return (sample["defect_pct"] <= spec["max_defect_pct"] and sample["moisture_pct"] <= spec["max_moisture_pct"]
            and (spec["min_screen"] is None or sample["screen"] >= spec["min_screen"]))


def rounds_left(state: dict, ctx: Context) -> int:
    rnd, total = state["round"], ctx.rules["timeline"]["rounds"]
    return total if rnd["kind"] == "practice" else total - rnd["number"] + 1


def sell_budget(state: dict, ctx: Context) -> dict[str, int]:
    """How many kg of each lot to offer this round: spread sales over the rounds that are left."""
    # TODO: use the Market Calendar — sell more in festival rounds (Specialty pays well), less when demand is low
    left = rounds_left(state, ctx)
    out = {}
    for lot in state["lots"]:
        avail = lot["kg_available"] // BAG * BAG
        share = avail if left <= 1 else math.ceil(avail / left * PACE / BAG) * BAG
        out[lot["lot_id"]] = min(avail, max(BAG, share)) if avail >= BAG else 0
    return out


def candidates(state: dict, ctx: Context) -> list[dict]:
    budget = sell_budget(state, ctx)
    out = []
    for lot in state["lots"]:
        if budget[lot["lot_id"]] < BAG:
            continue
        for rfq in state["rfqs"]:
            if spec_ok(lot["sample"], rfq["spec"]):
                out.append({"lot_id": lot["lot_id"], "rfq_id": rfq["rfq_id"], "buyer": rfq["buyer"],
                            "buyer_type": rfq["buyer_type"], "rfq_bags": rfq["bags"],
                            "max_bags": min(budget[lot["lot_id"]], rfq["kg"]) // BAG,
                            "tv": ctx.tv[lot["lot_id"]]})
    # Shuffle (weighted by RFQ size): if 40 teams all chase the first RFQ, they all get OUTBID.
    keyed = [(ctx.rng.random() ** (1 / c["rfq_bags"]), c) for c in out]
    return [c for _, c in sorted(keyed, key=lambda x: -x[0])]


# ---------- ขั้นบันไดราคารายผู้ซื้อ (ของเราเพิ่มเอง ไม่มีใน starter) ----------
# เพดานลับต่างกันในแต่ละ RFQ (พิสูจน์ในเกม 3) แต่ผู้ซื้อแต่ละรายมี "ระดับราคา" ของตัวเอง
# เลยไม่หาเพดานเป๊ะ แต่ขยับ "ระดับ" ของผู้ซื้อแต่ละรายจากผลทุกข้อเสนอ (ratio = ราคา ÷ TV ของเรา)
#   ขายได้ (ACCEPTED/PARTIAL) ที่ ratio >= ระดับ   -> ระดับ = ratio + UP
#   ABOVE_LIMIT ที่ ratio <= ระดับ                 -> ระดับ = ratio - DOWN
#   OUTBID ที่ ratio <= ระดับ                      -> ระดับ -= OUTBID_DOWN   (แพ้ทีมอื่น)
#   ข้อเสนอที่สูงกว่าระดับแล้วไม่ขาย (กระสอบทดลอง) -> ไม่เปลี่ยน
# ขึ้น 0.01 ลง 0.03 -> ระดับจะไปหยุดตรงที่ขายได้ราว 75% ของครั้ง (UP / (UP+DOWN) = 25% โดนปฏิเสธ)
UP, DOWN, OUTBID_DOWN = 0.01, 0.03, 0.01
PROBE_UP = 0.02      # กระสอบทดลอง 1 ใบ/ผู้ซื้อ/รอบ ที่ ระดับ + PROBE_UP


def buyer_levels(memory) -> dict:
    """buyer -> {"level": ตัวคูณที่ใช้รอบนี้, "type": ประเภท} จากผลทุกรอบของเกมนี้ ตามลำดับเวลา"""
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
            k = out.setdefault(bt[0], {"level": MARKUP[bt[1]], "type": bt[1]})
            m, lv = s["markup"], k["level"]
            if o["status"] in ("ACCEPTED", "PARTIAL") and m >= lv - 1e-9:
                k["level"] = m + UP
            elif o["status"] == "ABOVE_LIMIT" and m <= lv + 1e-9:
                k["level"] = m - DOWN
            elif o["status"] == "OUTBID" and m <= lv + 1e-9:
                k["level"] = lv - OUTBID_DOWN
            k["level"] = round(min(max(k["level"], MIN_MARKUP), MAX_MARKUP), 3)
    return out


def price_for(buyer_type: str, level: dict | None, probing: bool) -> tuple[float, float | None]:
    """(ตัวคูณราคาหลัก, ตัวคูณกระสอบทดลอง หรือ None)"""
    safe = level["level"] if level else MARKUP[buyer_type]
    probe = round(min(safe + PROBE_UP, MAX_MARKUP), 3) if probing and safe + PROBE_UP <= MAX_MARKUP else None
    return round(safe, 3), probe


def rule_plan(state: dict, ctx: Context, cands: list[dict]) -> list[dict]:
    """ไม่มี LLM: ราคาจากขั้นบันไดรายผู้ซื้อ · ผู้ซื้อที่จ่ายดีได้ล็อตดีก่อน · ผู้ซื้อละ 1 กระสอบทดลองต่อรอบ"""
    budget = sell_budget(state, ctx)
    rfq_left = {r["rfq_id"]: r["bags"] for r in state["rfqs"]}
    levels = buyer_levels(ctx.memory)
    probing = rounds_left(state, ctx) > 1             # รอบสุดท้ายไม่ทดลอง ไม่มีรอบหน้าให้ใช้ข้อมูล
    price = {c["buyer"]: price_for(c["buyer_type"], levels.get(c["buyer"]), probing) for c in cands}
    probed, plan = set(), []
    # ผู้ซื้อที่ราคาหลักสูงก่อน แล้วล็อต TV สูงก่อน (ลำดับสุ่มของ candidates() ใช้ตัดสินตอนเท่ากัน)
    for c in sorted(cands, key=lambda c: (-price[c["buyer"]][0], -c["tv"])):
        safe, probe = price[c["buyer"]]
        bags = min(budget[c["lot_id"]] // BAG, rfq_left[c["rfq_id"]])
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
    if levels:
        ctx.notes.append("levels: " + json.dumps({b: v["level"] for b, v in levels.items()}, ensure_ascii=False))
    return plan


SYSTEM = """You are the sales agent of a coffee farm selling green robusta beans in a sealed-offer market.
Each round buyers post RFQs. You choose which RFQs to offer your lots to, how many 60 kg bags, and a markup.
price = markup x the lot's True Value estimate (tv). Buyers rank offers by price / true value, cheapest first,
and reject offers above their hidden ceiling. Unsold beans are worth only 0.80 x tv at the end.
Buyer types pay differently: SPECIALTY the most, then COMMERCIAL, INSTANT the least. Ceilings are hidden;
learn them from the results (ABOVE_LIMIT means the price was too high for that buyer).
Answer only JSON matching the schema."""

SCHEMA = {"type": "object", "required": ["offers", "reason"], "properties": {
    "offers": {"type": "array", "items": {"type": "object", "required": ["candidate", "bags", "markup"], "properties": {
        "candidate": {"type": "integer"}, "bags": {"type": "integer"}, "markup": {"type": "number"}}}},
    "reason": {"type": "string", "maxLength": 300}}}   # short: a long reason can run past max_tokens and cut the JSON


def llm_plan(state: dict, ctx: Context, cands: list[dict]) -> list[dict] | None:
    rnd = state["round"]
    target = sum(sell_budget(state, ctx).values()) // BAG
    if state["capacity_left_kg"] is not None:
        target = min(target, state["capacity_left_kg"] // BAG)
    time_for_llm = ctx.seconds_left - 5          # keep 5 s to send the offers before the round closes
    if time_for_llm < 3:
        ctx.notes.append("no time for the LLM this round")
        return None
    table = [{"candidate": i, **{k: c[k] for k in ("lot_id", "buyer", "buyer_type", "rfq_bags", "max_bags", "tv")}}
             for i, c in enumerate(cands)]
    user = "\n".join([
        f"Round {rnd['round_id']} ({rnd['kind']}), market calendar: {rnd['calendar']}. Rounds left: {rounds_left(state, ctx)}.",
        f"You may offer at most {state['capacity_left_kg'] // BAG if state['capacity_left_kg'] is not None else 'any'} bags in total this round.",
        f"Plan to offer about {target} bags in total this round: unsold beans lose 20% of their value at the end.",
        f"Suggested markups: {json.dumps(MARKUP)}",
        "Recent results:", ctx.memory.summary() if ctx.memory else "none",
        "Candidates (lot x RFQ, already spec-checked):", json.dumps(table, ensure_ascii=False),
        "Pick offers. Use each candidate at most once; bags <= max_bags.",
    ])
    # TODO: give the LLM more to reason about — buyer personas' habits, the calendar ahead, price trends
    answer = ctx.llm.ask_json(SYSTEM, user, SCHEMA, timeout=time_for_llm)
    if answer is None or not isinstance(answer.get("offers"), list):
        ctx.notes.append(f"llm failed: {ctx.llm.last_error}")
        return None
    plan = []
    for o in answer["offers"]:
        try:
            c = cands[int(o["candidate"])]
            plan.append({"lot_id": c["lot_id"], "rfq_id": c["rfq_id"], "bags": int(o["bags"]), "markup": float(o["markup"])})
        except (KeyError, IndexError, TypeError, ValueError):
            ctx.notes.append(f"llm bad offer ignored: {o}")
    ctx.notes.append(f"llm reason: {str(answer.get('reason'))[:200]}")
    return plan


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
    """Returns (offers, source) where source is "llm" or "rules"."""
    cands = candidates(state, ctx)
    if not cands:
        return [], "none"
    plan = llm_plan(state, ctx, cands) if ctx.llm else None
    if plan:
        # The LLM tends to offer too little. Rules fill whatever volume it left unused
        # (guard skips lot×RFQ pairs the LLM already used and stops at the round capacity).
        offers = guard(state, ctx, plan + rule_plan(state, ctx, cands))
        n_llm = len(guard(state, Context(ctx.rules, ctx.calendar, ctx.tv, ctx.rng), plan))
        return offers, "llm" if len(offers) == n_llm else f"llm+rules ({n_llm}+{len(offers) - n_llm})"
    return guard(state, ctx, rule_plan(state, ctx, cands)), "rules"
