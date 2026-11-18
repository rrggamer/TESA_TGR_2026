"""ย้อนดูด่าน 2 (memory-competition): ถ้าใช้ strategy ตัวซ้อม จะตั้งตัวคูณหลักต่อผู้ซื้อเท่าไหร่ เทียบกับที่ใช้จริง
ใช้ประวัติจริงทีละรอบ (ไม่ใช่การจำลองตลาด: ถ้าราคาเปลี่ยน ผลรอบต่อไปก็เปลี่ยน) · ไม่ใส่ LLM"""
import glob, json, os, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
H = Path(__file__).parent
sys.path.insert(0, str(H / "practice"))
import strategy  # ตัวซ้อม
class M: pass
f = max(glob.glob(str(H.parent / "Coffee-TESA-starter/logs/memory-competition-*.json")), key=os.path.getmtime)
d = json.load(open(f, encoding="utf-8"))
rows = [json.loads(l) for l in open(H.parent / "Coffee-TESA-starter/logs/decisions.jsonl", encoding="utf-8") if l.strip()]
tv = [r for r in rows if r.get("event") == "new_game" and r.get("market") == "competition"][-1]["tv"]
for k in range(1, len(d["rounds"])):
    mem = M(); mem.rounds, mem.sent = d["rounds"][:k], d["sent"]
    rnd = d["rounds"][k]; rid = rnd["round_id"]
    levels = strategy.buyer_levels(mem); lim = strategy.move_limits(mem, last=(k == len(d["rounds"]) - 1))
    mcap = strategy.market_price_cap(mem)
    info = {x["rfq_id"]: x for x in rnd["market_summary"]}
    sent = {(o["rfq_id"], o["lot_id"]): o for o in d["sent"][rid]}
    print(f"\n{rid}")
    for o in rnd["my_offers"]:
        q = info[o["rfq_id"]]; b = q["buyer"]; s = sent[(o["rfq_id"], o["lot_id"])]
        safe = strategy.price_for(q["buyer_type"], levels.get(b), False, levels)[0]
        if b in lim: safe = min(max(safe, lim[b][0]), lim[b][1])
        pm = mcap.get(b); t = tv[o["lot_id"]]
        v3 = max(min(safe, pm * (1 - strategy.MARKET_CAP_MARGIN) / t), safe - strategy.MARKET_CAP_MAX_CUT) if pm else safe
        acc = [sent[(x["rfq_id"], x["lot_id"])]["markup"] for x in rnd["my_offers"] if info[x["rfq_id"]]["buyer"] == b and x["status"] in ("ACCEPTED", "PARTIAL")]
        hint = ""
        if o["status"] in ("ACCEPTED", "PARTIAL"): hint = "ขายได้ แต่ v3 ถูกลง" if v3 < s["markup"] - 1e-3 else ("ไม่แน่ (v3 แพงขึ้น)" if v3 > s["markup"] + 1e-3 else "ขายได้")
        elif o["status"] == "ABOVE_LIMIT": hint = "ยังเกินเพดาน" if v3 >= s["markup"] - 1e-3 else "ต่ำลง (อาจผ่าน)"
        else: hint = "ต่ำลง (อาจชนะ)" if v3 < s["markup"] - 1e-3 else "ยังแพ้"
        print(f"  {q['buyer_type'][:4]} lot{o['lot_id'][-1]} จริง x{s['markup']:.3f} {o['status'][:6]:6s} → v3 x{v3:.3f} ({v3*t:6.1f}฿) · ตลาดซื้อ {q['price_min']}–{q['price_max']} · {hint}")
