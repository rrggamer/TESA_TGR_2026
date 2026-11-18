"""ลอง LLM กับข้อมูลตลาดจริงที่เก็บไว้ (ไม่ส่งอะไรเข้าตลาด) — วัดว่าตอบทัน / JSON ถูก / ปรับราคายังไง

    python llm_try.py                                   # ใช้ LLM_* ใน practice/.env (gateway ของงานเท่านั้น)
    python llm_try.py --times 5 --round 4
ไม่ยอมรัน LLM บนเครื่องตัวเอง (localhost / 127.0.0.1 / พอร์ต 11434) — วันแข่งใช้ได้เฉพาะ gateway ของงาน

ใช้เกมล่าสุดของ autoplay: states-*.json (สถานะต้นรอบ) + memory-*.json ของเกมนั้น (ผลก่อนรอบที่เลือก)
"""
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PRACTICE = HERE / "practice"
sys.path.insert(0, str(PRACTICE))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import strategy  # noqa: E402
from appraisal import estimate_lots  # noqa: E402
from llm import LLM  # noqa: E402
from main import load_env  # noqa: E402
from market import Market  # noqa: E402


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


load_env(PRACTICE / ".env")
url = arg("--url", os.environ.get("LLM_BASE_URL"))
model = arg("--model", os.environ.get("LLM_MODEL", "event-llm"))
key = os.environ.get("LLM_API_KEY") or "none"
times, want = int(arg("--times", 3)), int(arg("--round", 4))
if any(bad in (url or "").lower() for bad in ("localhost", "127.0.0.1", ":11434", "ollama")):
    sys.exit(f"หยุด: LLM_BASE_URL = {url} เป็น LLM บนเครื่องตัวเอง · ใช้ได้เฉพาะ gateway ของงาน")
if "<" in (url or "<") or "GATEWAY-ADDRESS" in url:
    sys.exit("ยังไม่ได้ใส่ที่อยู่ gateway ใน practice/.env (LLM_BASE_URL)")

states_file = sorted((PRACTICE / "logs" / "autoplay").glob("states-*.json"))[-1]
states = json.loads(states_file.read_text(encoding="utf-8"))
state = next((s for s in states if s["round"]["number"] == want), states[-1])
lots_key = tuple(sorted((l["lot_id"], json.dumps(l["sample"], sort_keys=True)) for l in state["lots"]))
tag = hashlib.sha1(repr(lots_key).encode()).hexdigest()[:8]            # ชื่อไฟล์ memory แบบเดียวกับ main.py


class Mem:                                     # memory ของเกมนี้ ตัดให้เหลือผลก่อนรอบที่เลือก
    rounds, sent = [], {}


mf = PRACTICE / "logs" / f"memory-{state['market']}-{tag}.json"
if mf.exists():
    m = json.loads(mf.read_text(encoding="utf-8"))
    Mem.rounds = [r for r in m["rounds"] if int(r["round_id"][1:]) < state["round"]["number"]]
    Mem.sent = m["sent"]

rules = Market(os.environ["MARKET_URL"], os.environ["TEAM_KEY"]).calendar()
info_rules, calendar = rules["rules"], rules["calendar"]
tv = estimate_lots(info_rules, state["lots"])
print(f"LLM {url} · {model} · สถานะ {states_file.name} รอบ {state['round']['round_id']} · ผลก่อนหน้า {len(Mem.rounds)} รอบ\n")

llm = LLM(url, model, key, timeout=float(arg("--timeout", 60)), thinking="--thinking" in sys.argv,
          max_tokens=int(arg("--max-tokens", 1500)), temperature=float(arg("--temp", 0.7)))
print("temperature", llm.temperature, "· thinking", llm.thinking, "· max_tokens", llm.max_tokens)
for i in range(times):
    ctx = strategy.Context(info_rules, calendar, tv, random.Random(i), Mem, llm, 60)
    t = time.time()
    offers, source = strategy.decide(state, ctx)
    print(f"--- ครั้งที่ {i + 1}: {time.time() - t:.1f} วิ · source={source} · {len(offers)} ข้อเสนอ")
    for n in ctx.notes:
        print("   ", n[:300])
