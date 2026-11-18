"""ทดสอบไฟล์แข่งจริง (Coffee-TESA-starter): LLM ล่มแบบต่าง ๆ แล้วต้องยังขายด้วยกฎได้ — ไม่ส่งอะไรเข้าตลาด

    python fallback_real.py
ใช้สถานะตลาดที่ autoplay เก็บไว้ (practice/logs/autoplay/states-*.json) · LLM ปลอม/ที่อยู่ปลอมเท่านั้น (ไม่มี Ollama)
"""
import json, random, socket, sys, threading, time
from pathlib import Path

HERE = Path(__file__).parent
REAL = HERE.parent / "Coffee-TESA-starter"
sys.path.insert(0, str(REAL))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import strategy  # noqa: E402  (ไฟล์จริง)
from appraisal import estimate_lots  # noqa: E402
from llm import LLM  # noqa: E402
assert Path(strategy.__file__).resolve().parent == REAL.resolve(), strategy.__file__

rules = {"timeline": {"rounds": 8, "practice_rounds": 0}, "bag_kg": 60}
states = json.loads(sorted((HERE / "practice/logs/autoplay").glob("states-*.json"))[-1].read_text(encoding="utf-8"))
state = next(s for s in states if s["round"]["number"] == 3)
state["capacity_left_kg"] = state.get("round_capacity_kg")
cal_rules = json.loads((HERE / "practice/logs/autoplay").glob("reveal-*.json").__next__().read_text(encoding="utf-8")) and None


class Mem:
    rounds, sent = [], {}


def hang_server():                       # เซิร์ฟเวอร์ที่รับต่อแต่ไม่ตอบ = gateway ค้าง
    s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(5)
    def run():
        conns = []
        while True:
            c, _ = s.accept(); conns.append(c)
    threading.Thread(target=run, daemon=True).start()
    return s.getsockname()[1]


def err_server():                        # ตอบ HTTP 500 ทุกครั้ง = gateway พัง
    s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(5)
    def run():
        while True:
            c, _ = s.accept(); c.recv(65536)
            c.sendall(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"); c.close()
    threading.Thread(target=run, daemon=True).start()
    return s.getsockname()[1]


class Garbage:                           # ตอบมาแต่ผิดรูป
    last_error = None
    def __init__(self, ans): self.ans = ans
    def ask_json(self, *a, **k): return self.ans


class Boom:                              # โค้ด LLM โยน error เอง
    last_error = None
    def ask_json(self, *a, **k): raise RuntimeError("boom")


cases = [
    ("ที่อยู่ gateway ยังเป็นค่าปลอม (GATEWAY-ADDRESS)", LLM("http://GATEWAY-ADDRESS:4000/v1", "event-llm", "x", timeout=60), 60),
    ("gateway ค้าง ไม่ตอบ (timeout)", LLM(f"http://127.0.0.1:{hang_server()}/v1", "event-llm", "x", timeout=60), 12),
    ("gateway ตอบ HTTP 500", LLM(f"http://127.0.0.1:{err_server()}/v1", "event-llm", "x", timeout=60), 60),
    ("LLM ตอบ JSON ผิดรูป", Garbage({"buyers": "x"}), 60),
    ("LLM ตอบรหัสผู้ซื้อ/คำสั่งมั่ว", Garbage({"buyers": [{"id": "B99", "action": "fly"}, 5, "x"], "volume": "max"}), 60),
    ("LLM ตอบ None", Garbage(None), 60),
    ("โค้ด LLM โยน error", Boom(), 60),
    ("เวลาเหลือน้อย (ไม่ถาม LLM)", Garbage(None), 6),
]
tv = None
from market import Market  # noqa: E402
import os
sys.path.insert(0, str(HERE / "practice"))
from main import load_env  # noqa: E402
load_env(HERE / "practice/.env")
info = Market(os.environ["MARKET_URL"], os.environ["TEAM_KEY"]).calendar()      # อ่านกติกาอย่างเดียว
tv = estimate_lots(info["rules"], state["lots"])
ok_all = True
for name, llm, secs in cases:
    ctx = strategy.Context(info["rules"], info["calendar"], tv, random.Random(0), Mem, llm, secs)
    t = time.time()
    try:
        offers, source = strategy.decide(state, ctx)
        how = "decide"
    except Exception as e:                    # แบบเดียวกับ main.py จริง
        offers, source = strategy.guard(state, ctx, strategy.rule_plan(state, ctx, strategy.candidates(state, ctx))), "rules"
        how = f"main.py fallback ({type(e).__name__})"
    dt = time.time() - t
    kg = sum(o["kg"] for o in offers)
    good = source == "rules" and offers and all(o["kg"] % 60 == 0 for o in offers) and kg <= state["capacity_left_kg"] and dt < max(secs, 8)
    ok_all &= bool(good)
    print(f"{'✅' if good else '❌'} {name}: {source} · {len(offers)} ข้อเสนอ {kg} kg · {dt:.1f} วิ · {how}")
    for n in ctx.notes:
        if n.startswith(("llm", "no time")):
            print("     ", n[:150])
print("\nผ่านทุกกรณี" if ok_all else "\nมีกรณีไม่ผ่าน")
