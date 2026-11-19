import os
"""เทสออฟไลน์ของโค้ดจริงใน Coffee-TESA-starter (ไม่ต่อเน็ต ไม่ต้องลง pytest)

    python test_strategy.py          # ใช้ snapshot ล่าสุด (หรือ example.json)
เช็กว่า offer ที่ agent จะส่ง ผ่านเงื่อนไขในคู่มือ-API.md เสมอ — แก้ strategy.py เสร็จให้รันอันนี้ก่อนทุกครั้ง
"""
import math
import random
import unittest

from _setup import load_snapshot
import strategy
from appraisal import estimate_lots, true_value

INFO, STATE = load_snapshot()
RULES = INFO["rules"]
BAG = RULES.get("bag_kg", 60)
SLIDE17 = RULES["p0"] == 120   # ตัวเลขสไลด์ 17; ถ้าวันแข่งเปลี่ยน เทสสูตรแบบตัวเลขตายตัวจะข้าม


def make_ctx(state, llm=None):
    tv = estimate_lots(RULES, state["lots"])
    return strategy.Context(RULES, INFO["calendar"], tv, random.Random(1), None, llm, state.get("seconds_left") or 999)


def check_offers(tc, state, ctx, offers):
    """เงื่อนไขจากคู่มือ-API.md หัวข้อ POST /v1/offers"""
    lots = {l["lot_id"]: l for l in state["lots"]}
    rfqs = {r["rfq_id"]: r for r in state["rfqs"]}
    used_lot = {}
    for o in offers:
        tc.assertIn(o["lot_id"], lots)
        tc.assertIn(o["rfq_id"], rfqs)
        tc.assertEqual(o["kg"] % BAG, 0, f"kg ต้องทวีคูณ {BAG}: {o}")
        tc.assertGreater(o["kg"], 0)
        tc.assertLessEqual(o["kg"], rfqs[o["rfq_id"]]["kg"], f"เกิน kg ของ RFQ: {o}")
        tc.assertTrue(0 < o["price_per_kg"] <= 1000, f"ราคานอกช่วง 0-1000: {o}")
        tc.assertTrue(strategy.spec_ok(lots[o["lot_id"]]["sample"], rfqs[o["rfq_id"]]["spec"]), f"ไม่ผ่านสเปก: {o}")
        m = o["price_per_kg"] / ctx.tv[o["lot_id"]]
        tc.assertTrue(strategy.MIN_MARKUP - 0.01 <= m <= strategy.MAX_MARKUP + 0.01, f"markup {m:.3f} หลุดกรอบ")
        used_lot[o["lot_id"]] = used_lot.get(o["lot_id"], 0) + o["kg"]
    for lid, kg in used_lot.items():
        tc.assertLessEqual(kg, lots[lid]["kg_available"], f"{lid} เสนอรวมเกินที่มี")
    if state["capacity_left_kg"] is not None:
        tc.assertLessEqual(sum(o["kg"] for o in offers), state["capacity_left_kg"], "เกิน capacity_left_kg")
    pairs = [(o["lot_id"], o["rfq_id"]) for o in offers]
    tc.assertEqual(len(pairs), len(set(pairs)), "lot×RFQ ซ้ำ")


class FakeLLM:
    """LLM ปลอม: คืนคำตอบที่กำหนดไว้ (จำลอง LLM ตอบมั่ว / ไม่ตอบ / timeout)"""
    def __init__(self, answer):
        self.answer, self.last_error = answer, "fake"

    def ask_json(self, system, user, schema, timeout=None):
        return self.answer


class TestAppraisal(unittest.TestCase):
    @unittest.skipUnless(SLIDE17, "rules ไม่ใช่ตัวเลขสไลด์ 17")
    def test_tv_formula_slide17(self):
        # TV = 120 × G × M × S   d=5.56 -> step 0.96, m=12.95, s=13 -> 0.97
        s = {"defect_pct": 5.56, "moisture_pct": 12.95, "screen": 13}
        g = (1 - 5.56 / 100) * 0.96
        m = (1 - 12.95 / 100) / 0.875 - 0.02 * (12.95 - 12.5)
        self.assertAlmostEqual(true_value(RULES, s), 120 * g * m * 0.97, places=4)

    @unittest.skipUnless(SLIDE17, "rules ไม่ใช่ตัวเลขสไลด์ 17")
    def test_moisture_over_limit_and_screen_clamp(self):
        # m > 15 -> M = 0.60 ; screen 20 -> clamp เป็น 18 = 1.06
        s = {"defect_pct": 2, "moisture_pct": 15.5, "screen": 20}
        self.assertAlmostEqual(true_value(RULES, s), 120 * 0.98 * 1.04 * 0.60 * 1.06, places=4)

    def test_all_lots_positive(self):
        for lid, v in estimate_lots(RULES, STATE["lots"]).items():
            self.assertGreater(v, 0, lid)


class TestRules(unittest.TestCase):
    def test_rule_plan_valid(self):
        ctx = make_ctx(STATE)
        offers, source = strategy.decide(STATE, ctx)
        self.assertIn(source, ("rules", "none"))
        check_offers(self, STATE, ctx, offers)

    def test_capacity_one_bag(self):
        st = {**STATE, "capacity_left_kg": BAG}
        ctx = make_ctx(st)
        offers, _ = strategy.decide(st, ctx)
        check_offers(self, st, ctx, offers)

    def test_no_rfqs(self):
        st = {**STATE, "rfqs": []}
        self.assertEqual(strategy.decide(st, make_ctx(st))[0], [])

    def test_last_round_offers_everything(self):
        rnd = {**STATE["round"], "kind": "real", "number": RULES["timeline"]["rounds"]}
        st = {**STATE, "round": rnd, "capacity_left_kg": None}
        budget = strategy.sell_budget(st, make_ctx(st))
        for l in st["lots"]:
            self.assertEqual(budget[l["lot_id"]], l["kg_available"] // BAG * BAG)


class TestLLMFallback(unittest.TestCase):
    def run_llm(self, answer):
        ctx = make_ctx(STATE, FakeLLM(answer))
        offers, source = strategy.decide(STATE, ctx)
        check_offers(self, STATE, ctx, offers)
        return offers, source

    def test_llm_no_answer(self):        # LLM timeout -> ต้องใช้ rules
        self.assertIn(self.run_llm(None)[1], ("rules", "none"))

    def test_llm_garbage(self):          # candidate ไม่มีจริง, bags ติดลบ, markup 50
        self.run_llm({"offers": [{"candidate": 999, "bags": 3, "markup": 1.0},
                                 {"candidate": 0, "bags": -5, "markup": 50},
                                 {"candidate": "x", "bags": 1, "markup": 1}], "reason": "?"})

    def test_llm_huge_markup_clamped(self):
        self.run_llm({"offers": [{"candidate": 0, "bags": 99, "markup": 9.9}], "reason": ""})


class FakeMemory:
    """memory ปลอม: ผู้ซื้อ "I1" (INSTANT) ขายได้ที่ 0.92 แต่ ABOVE_LIMIT ที่ 1.00 · "C1" ขายได้ที่ 1.05"""
    def __init__(self):
        self.sent = {"R1": [{"rfq_id": "R1-a", "lot_id": "L", "markup": 0.92},
                            {"rfq_id": "R1-b", "lot_id": "L", "markup": 1.00},
                            {"rfq_id": "R1-c", "lot_id": "L", "markup": 1.05},
                            {"rfq_id": "R1-d", "lot_id": "L", "markup": 1.20}]}
        self.rounds = [{"round_id": "R1",
                        "market_summary": [{"rfq_id": "R1-a", "buyer": "I1"}, {"rfq_id": "R1-b", "buyer": "I1"},
                                           {"rfq_id": "R1-c", "buyer": "C1"}, {"rfq_id": "R1-d", "buyer": "C1"}],
                        "my_offers": [{"rfq_id": "R1-a", "lot_id": "L", "status": "ACCEPTED"},
                                      {"rfq_id": "R1-b", "lot_id": "L", "status": "ABOVE_LIMIT"},
                                      {"rfq_id": "R1-c", "lot_id": "L", "status": "PARTIAL"},
                                      {"rfq_id": "R1-d", "lot_id": "L", "status": "OUTBID"}]}]


@unittest.skipUnless(hasattr(strategy, "buyer_levels"), "โค้ดนี้ไม่มีขั้นบันไดราคา")
class TestStaircase(unittest.TestCase):
    def test_levels_from_results(self):
        # ใช้ market_summary ของ FakeMemory: ใส่ buyer_type ให้ด้วย
        mem = FakeMemory()
        types = {"I1": "INSTANT", "C1": "COMMERCIAL"}
        for s in mem.rounds[0]["market_summary"]:
            s["buyer_type"] = types[s["buyer"]]
        lv = strategy.buyer_levels(mem)
        up, down = strategy.UP, strategy.DOWN
        # I1: เริ่ม MARKUP INSTANT · ขาย 0.92 (ถ้า >= ระดับ ขึ้น) · ABOVE_LIMIT 1.00 (ถ้า <= ระดับ ลง)
        start = strategy.MARKUP["INSTANT"]
        cap = getattr(strategy, "MOVE_UP", None)                    # v3: ขายได้ -> ขึ้นไม่เกิน MOVE_UP จากระดับเดิม
        rise = lambda m, lv: min(m + up, lv + cap) if cap is not None else m + up
        level = rise(0.92, start) if 0.92 >= start else start
        level = 1.00 - down if 1.00 <= level else level
        self.assertAlmostEqual(lv["I1"]["level"], round(max(level, strategy.MIN_MARKUP), 3))
        # C1: PARTIAL ที่ 1.05 -> ขึ้น · OUTBID ที่ 1.20 สูงกว่าระดับ -> ไม่เปลี่ยน
        self.assertAlmostEqual(lv["C1"]["level"], round(rise(1.05, strategy.MARKUP["COMMERCIAL"]), 3))

    def test_level_stays_in_bounds(self):
        mem = FakeMemory()
        for s in mem.rounds[0]["market_summary"]:
            s["buyer_type"] = "INSTANT"
        for o in mem.rounds[0]["my_offers"]:
            o["status"] = "ABOVE_LIMIT"
        for v in strategy.buyer_levels(mem).values():
            self.assertGreaterEqual(v["level"], strategy.MIN_MARKUP)

    def test_probe_only_when_not_last_round(self):
        safe, probe = strategy.price_for("INSTANT", {"level": 0.93}, True)
        self.assertEqual(safe, 0.93)
        self.assertAlmostEqual(probe, 0.93 + strategy.PROBE_UP)
        self.assertIsNone(strategy.price_for("INSTANT", {"level": 0.93}, False)[1])

    def test_one_probe_bag_per_buyer(self):
        ctx = make_ctx(STATE)
        cands = strategy.candidates(STATE, ctx)
        plan = strategy.rule_plan(STATE, ctx, cands)
        prior = {c["buyer"]: strategy.price_for(c["buyer_type"], None, False, {})[0] for c in cands}   # ราคาหลัก (ไม่ใช่กระสอบทดลอง)
        buyer_of = {r["rfq_id"]: r["buyer"] for r in STATE["rfqs"]}
        probes = [p for p in plan if p["markup"] > prior[buyer_of[p["rfq_id"]]] + 1e-9]
        self.assertTrue(all(p["bags"] == 1 for p in probes))
        self.assertEqual(len(probes), len({buyer_of[p["rfq_id"]] for p in probes}))
        check_offers(self, STATE, ctx, strategy.guard(STATE, ctx, plan))


@unittest.skipUnless(hasattr(strategy, "llm_adjust"), "โค้ดนี้ไม่มีเวอร์ชันพร้อมแข่ง")
class TestCompetitionVersion(unittest.TestCase):
    def test_spec_margin(self):
        spec = {"max_defect_pct": 5.0, "max_moisture_pct": 12.5, "min_screen": None}
        edge = {"defect_pct": 4.9, "moisture_pct": 12.5, "screen": 14}
        self.assertTrue(strategy.spec_ok(edge, spec))
        self.assertFalse(strategy.spec_ok(edge, spec, {"defect_pct": 0.3}))
        self.assertFalse(strategy.spec_ok(edge, spec, {"moisture_pct": 0.1}))

    def test_calendar_weight(self):
        cal = [{"rounds": "1-4", "demand_weight": 1.0}, {"rounds": "5-8", "demand_weight": 2.0}]
        rules = {**RULES, "timeline": {**RULES["timeline"], "rounds": 8}}
        ctx = strategy.Context(rules, cal, {}, random.Random(0))
        r = lambda n: {"round": {"kind": "real", "number": n}}
        self.assertAlmostEqual(strategy.calendar_weight(r(1), ctx), 1.0 / 1.5, places=3)   # รอบปกติ ก่อนเทศกาล: ขายน้อยลง
        self.assertAlmostEqual(strategy.calendar_weight(r(5), ctx), 1.0, places=3)
        bad = strategy.Context(rules, [{"rounds": "x"}], {}, random.Random(0))
        self.assertEqual(strategy.calendar_weight(r(1), bad), 1.0)                         # อ่านไม่ได้ = ไม่เปลี่ยน

    def test_instant_only_sells_faster(self):
        st = {**STATE, "rfqs": [r for r in STATE["rfqs"] if r["buyer_type"] == "INSTANT"]}
        ctx = make_ctx(st)
        fast = strategy.sell_budget(st, ctx)
        old, strategy.PACE_INSTANT_ONLY = strategy.PACE_INSTANT_ONLY, strategy.PACE
        try:
            normal = strategy.sell_budget(st, ctx)
        finally:
            strategy.PACE_INSTANT_ONLY = old
        self.assertTrue(all(fast[k] >= normal[k] for k in fast))
        self.assertTrue(any(fast[k] > normal[k] for k in fast))

    def test_specialty_floor(self):
        levels = {"C1": {"level": 1.12, "type": "COMMERCIAL", "hist": []}}
        self.assertGreaterEqual(strategy.type_floor("SPECIALTY", levels), 1.12)
        self.assertEqual(strategy.type_floor("INSTANT", levels), strategy.MARKUP["INSTANT"])
        # ยังไม่มีข้อมูลเลย: Specialty ต้องไม่ต่ำกว่าค่าตั้งต้นของ Commercial (เจอใน dry run example.json)
        self.assertGreaterEqual(strategy.type_floor("SPECIALTY", {}), strategy.MARKUP["COMMERCIAL"])

    def test_llm_actions(self):
        ans = {"buyers": [{"id": "B1", "evidence": "x", "action": "raise"}, {"id": "B2", "evidence": "x", "action": "lower"},
                          {"id": "B9", "evidence": "x", "action": "raise"}, {"id": "B3", "evidence": "x", "action": "+5"}],
               "volume": "sell_more", "reason": "x"}
        ctx = make_ctx(STATE, FakeLLM(ans))
        offers, source = strategy.decide(STATE, ctx)
        self.assertEqual(source, "llm-adjust")
        check_offers(self, STATE, ctx, offers)
        note = [n for n in ctx.notes if n.startswith("llm:")][0]
        self.assertIn("B1:raise->hold", note)          # ไม่มีข้อมูลผู้ซื้อ -> Python ไม่ให้ขึ้น
        self.assertEqual(ctx.volume, strategy.VOLUME_FACTOR["sell_more"])

    def test_python_veto(self):
        v = strategy.python_veto
        self.assertEqual(v("raise", {"sold": 3, "too_expensive": 1, "lost_to_cheaper": 0}), "hold")
        self.assertEqual(v("raise", {"sold": 3, "too_expensive": 0, "lost_to_cheaper": 0}), "raise")
        self.assertEqual(v("lower", {"sold": 3, "too_expensive": 0, "lost_to_cheaper": 0}), "hold")
        self.assertEqual(v("lower", {"sold": 0, "too_expensive": 2, "lost_to_cheaper": 0}), "lower")
        self.assertEqual(v("raise", None), "hold")

    def test_llm_wrong_format_falls_back(self):
        for ans in (None, {"offers": []}, {"buyers": "x"}, "text"):
            ctx = make_ctx(STATE, FakeLLM(ans))
            offers, source = strategy.decide(STATE, ctx)
            self.assertEqual(source, "rules", ans)
            check_offers(self, STATE, ctx, offers)


@unittest.skipUnless(hasattr(strategy, "BUYER_SHARE"), "ไม่มีแก้ A/B/C")
class TestSpreadAndLastRound(unittest.TestCase):
    def test_buyer_share_cap(self):                 # (A) ไม่ทุ่มให้ผู้ซื้อรายเดียว
        ctx = make_ctx(STATE)
        cands = strategy.candidates(STATE, ctx)
        if len({c["buyer"] for c in cands}) < 2:
            self.skipTest("snapshot มีผู้ซื้อรายเดียว")
        plan = strategy.rule_plan(STATE, ctx, cands)
        total = sum(strategy.sell_budget(STATE, ctx).values()) // BAG
        buyer = {c["rfq_id"]: c["buyer"] for c in cands}
        per = {}
        for p in plan:
            per[buyer[p["rfq_id"]]] = per.get(buyer[p["rfq_id"]], 0) + p["bags"]
        self.assertLessEqual(max(per.values()), math.ceil(total * strategy.BUYER_SHARE))

    def test_buyer_share_with_round_capacity(self):   # สไลด์ 18: จำกัดกระสอบต่อรอบ -> ยังต้องกระจายผู้ซื้อ
        st = {**STATE, "capacity_left_kg": 4 * BAG}
        ctx = make_ctx(st)
        cands = strategy.candidates(st, ctx)
        if len({c["buyer"] for c in cands}) < 2:
            self.skipTest("snapshot มีผู้ซื้อรายเดียว")
        offers = strategy.guard(st, ctx, strategy.rule_plan(st, ctx, cands))
        check_offers(self, st, ctx, offers)
        self.assertLessEqual(sum(o["kg"] for o in offers), 4 * BAG)
        buyer = {c["rfq_id"]: c["buyer"] for c in cands}
        self.assertGreaterEqual(len({buyer[o["rfq_id"]] for o in offers}), 2)     # ไม่ตกไปรายเดียว

    def test_last_round_cheaper_and_sells_more(self):   # (B)
        rules = {**RULES, "timeline": {**RULES["timeline"], "rounds": 6}}
        def run(n):
            st = {**STATE, "round": {**STATE["round"], "kind": "real", "number": n}}
            ctx = strategy.Context(rules, INFO["calendar"], estimate_lots(RULES, st["lots"]), random.Random(1))
            plan = strategy.rule_plan(st, ctx, strategy.candidates(st, ctx))
            check_offers(self, st, ctx, strategy.guard(st, ctx, plan))
            return plan, ctx
        mid, _ = run(3)
        last, ctx = run(6)
        self.assertLess(max(p["markup"] for p in last), max(p["markup"] for p in mid))
        self.assertGreaterEqual(sum(p["bags"] for p in last), sum(p["bags"] for p in mid))
        self.assertTrue(any("last-round" in n for n in ctx.notes))

    def test_shock_when_nothing_sold(self):         # (C)
        class M:
            rounds = [{"round_id": "R1", "my_offers": [{"rfq_id": "x", "lot_id": "y", "status": "ABOVE_LIMIT"}],
                       "market_summary": []}]
            sent = {}
        self.assertTrue(strategy.last_round_sold_nothing(M))
        M.rounds[0]["my_offers"].append({"rfq_id": "x", "lot_id": "z", "status": "PARTIAL"})
        self.assertFalse(strategy.last_round_sold_nothing(M))
        self.assertFalse(strategy.last_round_sold_nothing(None))


@unittest.skipUnless(hasattr(strategy, "PACE_PRE_FLOOD"), "ไม่มีชุดแข่งกับทีมอื่น")
class TestCompetitorEdges(unittest.TestCase):
    def test_round_dodge(self):
        self.assertAlmostEqual(strategy.price_for("COMMERCIAL", {"level": 1.05, "type": "COMMERCIAL"}, False)[0], 1.05 - strategy.ROUND_DODGE)
        self.assertAlmostEqual(strategy.price_for("INSTANT", {"level": 0.92, "type": "INSTANT"}, False)[0], 0.92)   # ไม่กลม ไม่ขยับ
        self.assertAlmostEqual(strategy.price_for("COMMERCIAL", {"level": 1.07, "type": "COMMERCIAL"}, False)[0], 1.07)

    def test_pre_flood_sells_more(self):
        rules = {**RULES, "timeline": {**RULES["timeline"], "rounds": 8}}
        st = {**STATE, "rfqs": [r for r in STATE["rfqs"] if r["buyer_type"] != "INSTANT"]}
        def budget(n):
            s = {**st, "round": {**st["round"], "kind": "real", "number": n}}
            return sum(strategy.sell_budget(s, strategy.Context(rules, [], {}, random.Random(0))).values())
        # 3 รอบเหลือ (รอบ 6) ต้องเสนอต่อรอบเป็นสัดส่วนมากกว่า PACE ปกติ
        old = strategy.PACE_PRE_FLOOD
        try:
            strategy.PACE_PRE_FLOOD = strategy.PACE
            normal = budget(6)
        finally:
            strategy.PACE_PRE_FLOOD = old
        self.assertGreaterEqual(budget(6), normal)

    def test_undersupplied(self):
        class M:
            rounds = [{"market_summary": [{"buyer": "A", "bought_kg": 60, "wanted_kg": 600},
                                          {"buyer": "B", "bought_kg": 600, "wanted_kg": 600}]}]
        self.assertEqual(strategy.undersupplied_buyers(M), {"A"})
        self.assertEqual(strategy.undersupplied_buyers(None), set())

    def test_hold_back_vetoed_when_behind(self):
        ans = {"buyers": [], "volume": "hold_back", "reason": "x"}
        ctx = make_ctx(STATE, FakeLLM(ans))
        strategy.decide(STATE, ctx)          # ของยังเหลือเกือบหมด -> ห้ามชะลอ
        self.assertEqual(ctx.volume, 1.0)


@unittest.skipUnless(hasattr(strategy, "spec_history"), "ไม่มีชุดจำ SPEC_MISMATCH")
class TestSpecLearning(unittest.TestCase):
    def test_no_resend_after_mismatch(self):
        lot = STATE["lots"][0]["lot_id"]
        rfq = next(r for r in STATE["rfqs"] if strategy.spec_ok(STATE["lots"][0]["sample"], r["spec"]))
        class M:
            sent = {}
            rounds = [{"round_id": "R0", "market_summary": [{"rfq_id": "X1", "buyer": rfq["buyer"], "buyer_type": rfq["buyer_type"],
                                                             "wanted_kg": 60, "bought_kg": 0}],
                       "my_offers": [{"rfq_id": "X1", "lot_id": lot, "status": "SPEC_MISMATCH", "kg": 60, "filled_kg": 0}]}]
        ctx = make_ctx(STATE); ctx.memory = M
        for c in strategy.candidates(STATE, ctx):
            if c["lot_id"] == lot:
                spec = next(r["spec"] for r in STATE["rfqs"] if r["rfq_id"] == c["rfq_id"])
                self.assertNotEqual(c["buyer"], rfq["buyer"])
                self.assertFalse(strategy.stricter_or_same(spec, rfq["spec"]))

    def test_near_edge(self):
        spec = {"max_defect_pct": 7.0, "max_moisture_pct": 12.5, "min_screen": 16}
        self.assertTrue(strategy.near_edge({"defect_pct": 2, "moisture_pct": 12.5, "screen": 18}, spec))
        self.assertTrue(strategy.near_edge({"defect_pct": 2, "moisture_pct": 11, "screen": 16}, spec))
        self.assertFalse(strategy.near_edge({"defect_pct": 2, "moisture_pct": 11, "screen": 17}, spec))



def _mem(rounds, sent):
    m = FakeMemory(); m.rounds, m.sent = rounds, sent; return m


@unittest.skipUnless(getattr(strategy, "CLEAN_RAISE", False), "v4 เท่านั้น")
class TestV4CleanRaise(unittest.TestCase):
    """19 พ.ย.: ราคาไต่ขึ้นทุกรอบที่ขายได้ แม้ราคาหลักบางข้อแพ้ -> ของค้าง ต้องเทลดราคาท้ายเกม"""
    def _mem(self, lost_status):
        summ = [{"rfq_id": "R4-1", "buyer": "C", "buyer_type": "COMMERCIAL", "wanted_kg": 600, "bought_kg": 600,
                 "offers": 20, "price_min": 100.0, "price_max": 126.0, "price_avg": 115.0}]
        sent = {"R4": [{"rfq_id": "R4-1", "lot_id": "L", "markup": 1.00}, {"rfq_id": "R4-1", "lot_id": "M", "markup": 1.00}]}
        offers = [{"rfq_id": "R4-1", "lot_id": "L", "status": "ACCEPTED", "filled_kg": 60},
                  {"rfq_id": "R4-1", "lot_id": "M", "status": lost_status, "filled_kg": 60 if lost_status == "ACCEPTED" else 0}]
        return _mem([{"round_id": "R4", "kind": "real", "my_offers": offers, "market_summary": summ}], sent)

    def test_hold_when_main_price_lost(self):
        lv = strategy.buyer_levels(self._mem("OUTBID"))["C"]["level"]
        self.assertLess(lv, 1.00 + 1e-9)                                       # ขายได้ 1 แพ้ 1 -> ไม่ขึ้น

    def test_raise_when_all_sold(self):
        lv = strategy.buyer_levels(self._mem("ACCEPTED"))["C"]["level"]
        if strategy.MOVE_UP > 0 and strategy.CLEAN_RAISE:                     # autofix ตั้ง MOVE_UP=0 = ไม่ขึ้นเลย (ตั้งใจ)
            self.assertGreater(lv, 1.00)
        else:
            self.assertLessEqual(lv, 1.00 + strategy.MOVE_UP + 0.01 + 1e-9)


@unittest.skipUnless(hasattr(strategy, "move_limits"), "v3 เท่านั้น")
class TestV3(unittest.TestCase):
    """บทเรียนด่าน 2: ขายหมดแล้วขึ้นราคาซ้อน (กฎ + กระสอบทดลอง + LLM) -> รอบถัดไปขายไม่ได้เลย"""
    def _full_accept(self, kind="real", rid="R4"):
        summ = [{"rfq_id": f"{rid}-1", "buyer": "C", "buyer_type": "COMMERCIAL", "wanted_kg": 600, "bought_kg": 600,
                 "offers": 20, "price_min": 100.0, "price_max": 126.0, "price_avg": 115.0}]
        sent = {rid: [{"rfq_id": f"{rid}-1", "lot_id": "L", "markup": 1.00}, {"rfq_id": f"{rid}-1", "lot_id": "M", "markup": 1.02}]}
        offers = [{"rfq_id": f"{rid}-1", "lot_id": "L", "status": "ACCEPTED", "filled_kg": 60},
                  {"rfq_id": f"{rid}-1", "lot_id": "M", "status": "ACCEPTED", "filled_kg": 60}]
        return _mem([{"round_id": rid, "kind": kind, "my_offers": offers, "market_summary": summ}], sent)

    def test_raise_after_full_accept_is_capped(self):
        lim = strategy.move_limits(self._full_accept(), last=False)
        self.assertAlmostEqual(lim["C"][1], 1.00 + strategy.MOVE_UP)          # ขึ้นได้ไม่เกิน +MOVE_UP จากตัวคูณหลัก
        lv = strategy.buyer_levels(self._full_accept())["C"]["level"]
        self.assertLessEqual(lv, 1.00 + strategy.MOVE_UP + 1e-9)              # กระสอบทดลองขายได้ไม่ทำให้กระโดด

    def test_down_cap(self):
        lim = strategy.move_limits(self._full_accept(), last=False)
        self.assertAlmostEqual(lim["C"][0], 1.00 - strategy.MOVE_DOWN)
        self.assertAlmostEqual(strategy.move_limits(self._full_accept(), last=True)["C"][0], 1.00 - strategy.LAST_MOVE_DOWN)

    def test_practice_sets_level_to_best_accepted(self):
        mem = self._full_accept(kind="practice", rid="P1")
        self.assertAlmostEqual(strategy.buyer_levels(mem)["C"]["level"], 1.02)   # ราคาสูงสุดที่ขายได้ ไม่ +UP
        self.assertGreaterEqual(strategy.move_limits(mem, last=False)["C"][1], 1.02)

    def test_market_cap_only_when_filled(self):
        mem = self._full_accept()
        self.assertEqual(strategy.market_price_cap(mem), {"C": 126.0})
        mem.rounds[0]["market_summary"][0]["bought_kg"] = 300                   # ผู้ซื้อยังได้ของไม่ครบ -> ไม่จำกัด
        self.assertEqual(strategy.market_price_cap(mem), {})

    def test_plan_respects_caps_with_llm_raise(self):
        """จำลองด่าน 2 R1: รอบก่อนขายหมด + LLM สั่ง raise -> ตัวคูณหลักต้องไม่เกิน +MOVE_UP และไม่แพงกว่าตลาด"""
        lot = {"lot_id": "L", "kg_total": 600, "kg_available": 600, "kg_remaining": 600,
               "sample": {"defect_pct": 2.0, "moisture_pct": 11.0, "screen": 16}}
        rfq = {"rfq_id": "R5-1", "buyer": "C", "buyer_type": "COMMERCIAL", "kg": 600, "bags": 10,
               "spec": {"max_defect_pct": 5, "max_moisture_pct": 12.5, "min_screen": 14}}
        state = {"market": "x", "phase": "ROUND_OPEN", "round": {"round_id": "R5", "kind": "real", "number": 5, "of": 15},
                 "lots": [lot], "rfqs": [rfq], "capacity_left_kg": 300, "round_capacity_kg": 300}
        ctx = strategy.Context({"timeline": {"rounds": 15}, "bag_kg": 60}, [], {"L": 125.0, "M": 125.0},
                               __import__("random").Random(1), self._full_accept(), None, 300)
        plan = strategy.rule_plan(state, ctx, strategy.candidates(state, ctx), {"C": strategy.ACTION_DELTA["raise"]})
        for p in plan:
            self.assertLessEqual(p["markup"], 126.0 * (1 - strategy.MARKET_CAP_MARGIN) / 125.0 + 1e-9)
        # max(.., 0.02): ถ้า autofix ลด MOVE_UP (เช่น 0) แผนนี้เหลือแค่กระสอบทดลอง (+PROBE_UP ถูกเพดานตลาดตัด) -> ใช้เกณฑ์ค่าเดิม
        self.assertLessEqual(min(p["markup"] for p in plan), 1.00 + max(strategy.MOVE_UP, 0.02) + 1e-9)

    def test_practice_ladder(self):
        lots = [{"lot_id": f"L{i}", "kg_total": 600, "kg_available": 600, "kg_remaining": 600,
                 "sample": {"defect_pct": 2.0, "moisture_pct": 11.0, "screen": 16}} for i in range(4)]
        rfq = {"rfq_id": "P1-1", "buyer": "C", "buyer_type": "COMMERCIAL", "kg": 600, "bags": 10,
               "spec": {"max_defect_pct": 5, "max_moisture_pct": 12.5, "min_screen": 14}}
        state = {"market": "x", "phase": "ROUND_OPEN", "round": {"round_id": "P1", "kind": "practice", "number": 1, "of": 2},
                 "lots": lots, "rfqs": [rfq], "capacity_left_kg": 600, "round_capacity_kg": 600}
        ctx = strategy.Context({"timeline": {"rounds": 15}, "bag_kg": 60}, [], {l["lot_id"]: 120.0 for l in lots},
                               __import__("random").Random(1), FakeMemory(), None, 300)
        ctx.memory.rounds, ctx.memory.sent = [], {}
        mk = sorted(p["markup"] for p in strategy.rule_plan(state, ctx, strategy.candidates(state, ctx)))
        self.assertEqual(len(set(mk)), len(mk))                                  # ทุกกระสอบคนละราคา
        self.assertGreater(mk[-1] - mk[0], 0.05)


CAL20 = [{"rounds": "1-5", "label": "ปกติ", "demand_weight": 1.0, "specialty_share": 0.2},
         {"rounds": "6-8", "label": "เทศกาลกาแฟ", "demand_weight": 1.2, "specialty_share": 0.4},
         {"rounds": "9-12", "label": "ปกติ", "demand_weight": 1.0, "specialty_share": 0.2},
         {"rounds": "13-14", "label": "ใกล้ปิดตลาด", "demand_weight": 0.6, "specialty_share": 0.2},
         {"rounds": "15", "label": "รอบสุดท้าย", "demand_weight": 0.4, "specialty_share": 0.2}]


@unittest.skipUnless(hasattr(strategy, "festival_rounds"), "v3 (ข้อ 5–6) เท่านั้น")
class TestCalendar20(unittest.TestCase):
    def _state(self, n, kind="real"):
        good = {"lot_id": "G", "kg_total": 600, "kg_available": 600, "kg_remaining": 600,
                "sample": {"defect_pct": 1.0, "moisture_pct": 10.5, "screen": 17}}
        rfqs = [{"rfq_id": "S", "buyer": "s", "buyer_type": "SPECIALTY", "kg": 300, "bags": 5,
                 "spec": {"max_defect_pct": 3, "max_moisture_pct": 12, "min_screen": 16}},
                {"rfq_id": "C", "buyer": "c", "buyer_type": "COMMERCIAL", "kg": 600, "bags": 10,
                 "spec": {"max_defect_pct": 6, "max_moisture_pct": 13, "min_screen": 14}}]
        return {"market": "x", "phase": "ROUND_OPEN", "round": {"round_id": f"R{n}", "kind": kind, "number": n, "of": 15},
                "lots": [good], "rfqs": rfqs, "capacity_left_kg": 300, "round_capacity_kg": 300}

    def _ctx(self, cal=CAL20):
        m = FakeMemory(); m.rounds, m.sent = [], {}
        return strategy.Context({"timeline": {"rounds": 15}, "bag_kg": 60}, cal, {"G": 120.0}, __import__("random").Random(1), m, None, 300)

    def test_festival_detected(self):
        self.assertEqual(strategy.festival_rounds(self._ctx()), {6, 7, 8})
        self.assertEqual(strategy.festival_rounds(self._ctx([{"rounds": "1-8", "label": "ปกติ", "demand_weight": 1.0, "specialty_share": None}])), set())

    def test_hold_before_festival_only(self):
        types = lambda n, kind="real": {c["buyer_type"] for c in strategy.candidates(self._state(n, kind), self._ctx())}
        # ก่อนเทศกาล: ล็อตเกรด Specialty ขายให้ Specialty เท่านั้น · autofix ปิด FESTIVAL_HOLD = ขายได้ทุกราย
        self.assertEqual(types(3), {"SPECIALTY"} if strategy.FESTIVAL_HOLD else {"SPECIALTY", "COMMERCIAL"})
        self.assertEqual(types(7), {"SPECIALTY", "COMMERCIAL"})         # ช่วง/หลังเทศกาล ขายได้ทุกราย
        self.assertEqual(types(10), {"SPECIALTY", "COMMERCIAL"})
        self.assertEqual(types(1, "practice"), {"SPECIALTY", "COMMERCIAL"})   # รอบซ้อมไม่เก็บ
        self.assertEqual({c["buyer_type"] for c in strategy.candidates(self._state(3), self._ctx([]))}, {"SPECIALTY", "COMMERCIAL"})

    def test_effective_rounds_left(self):
        self.assertAlmostEqual(strategy.effective_rounds_left(self._state(13), self._ctx()), 0.6 + 0.6 + 0.4)
        self.assertAlmostEqual(strategy.effective_rounds_left(self._state(13), self._ctx([])), 3)   # ไม่มีปฏิทิน = นับรอบปกติ


@unittest.skipUnless(hasattr(strategy, "apply_knobs"), "ปุ่มปรับ (v3) เท่านั้น")
class TestKnobs(unittest.TestCase):
    def tearDown(self):
        for k in ("KNOB_MAX_COMMERCIAL", "KNOB_MOVE_UP", "KNOB_FESTIVAL_HOLD"):
            os.environ.pop(k, None)
        strategy.apply_knobs(strategy.Context({}, [], {}, None))

    def test_defaults_and_bad_values(self):
        ctx = strategy.Context({}, [], {}, None)
        os.environ["KNOB_MOVE_UP"] = "abc"; os.environ["KNOB_MAX_COMMERCIAL"] = "5"
        strategy.apply_knobs(ctx)
        code = getattr(strategy, "_CODE_KNOBS", (0.02, True, {}))              # ค่าผิด -> กลับไปใช้ค่าในโค้ด
        self.assertEqual(strategy.MOVE_UP, code[0])
        self.assertEqual(strategy.KNOB_MAX.get("COMMERCIAL"), code[2].get("COMMERCIAL"))   # ไม่ใช่ 5
        self.assertTrue(any("knob ignored" in n for n in ctx.notes))

    def test_max_markup_applies(self):
        os.environ["KNOB_MAX_COMMERCIAL"] = "0.95"; os.environ["KNOB_FESTIVAL_HOLD"] = "0"
        lot = {"lot_id": "L", "kg_total": 600, "kg_available": 600, "kg_remaining": 600,
               "sample": {"defect_pct": 2.0, "moisture_pct": 11.0, "screen": 16}}
        rfq = {"rfq_id": "R1-1", "buyer": "C", "buyer_type": "COMMERCIAL", "kg": 600, "bags": 10,
               "spec": {"max_defect_pct": 5, "max_moisture_pct": 12.5, "min_screen": 14}}
        state = {"market": "x", "phase": "ROUND_OPEN", "round": {"round_id": "R1", "kind": "real", "number": 1, "of": 15},
                 "lots": [lot], "rfqs": [rfq], "capacity_left_kg": 300, "round_capacity_kg": 300}
        m = FakeMemory(); m.rounds, m.sent = [], {}
        ctx = strategy.Context({"timeline": {"rounds": 15}, "bag_kg": 60}, [], {"L": 120.0}, __import__("random").Random(1), m, None, 300)
        offers, _ = strategy.decide(state, ctx)
        self.assertTrue(offers)
        self.assertLessEqual(max(o["price_per_kg"] for o in offers), round((0.95 + strategy.PROBE_UP) * 120.0, 2))
        self.assertFalse(strategy.FESTIVAL_HOLD)
        self.assertTrue(any(n.startswith("knobs:") for n in ctx.notes))

    @unittest.skipUnless(hasattr(strategy, "_CODE_KNOBS"), "ค่าในโค้ดเป็นค่าเริ่มต้น (20 พ.ย.)")
    def test_code_constants_survive_without_env(self):
        """แก้ค่าในโค้ด (autofix) โดยไม่มีปุ่มใน .env -> apply_knobs ต้องไม่ทับกลับเป็น 0.02/True/{}"""
        saved = strategy._CODE_KNOBS
        try:
            strategy._CODE_KNOBS = (0.0, False, {"COMMERCIAL": 0.97})
            strategy.apply_knobs(strategy.Context({}, [], {}, None))
            self.assertEqual(strategy.MOVE_UP, 0.0)
            self.assertFalse(strategy.FESTIVAL_HOLD)
            self.assertEqual(strategy.KNOB_MAX, {"COMMERCIAL": 0.97})
            os.environ["KNOB_MAX_COMMERCIAL"] = "1.01"                      # ปุ่มใน .env ยังทับได้
            strategy.apply_knobs(strategy.Context({}, [], {}, None))
            self.assertEqual(strategy.KNOB_MAX["COMMERCIAL"], 1.01)
        finally:
            strategy._CODE_KNOBS = saved

if __name__ == "__main__":
    print("snapshot: p0 =", RULES["p0"], "| lots", len(STATE["lots"]), "| rfqs", len(STATE["rfqs"]))
    unittest.main(verbosity=2)
