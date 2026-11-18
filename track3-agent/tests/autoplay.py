"""ซ้อมเร็ว: กด "ข้ามไปช่วงถัดไป" ให้เอง (POST /v1/sandbox/advance) — ตลาดฝึกเท่านั้น · ห้ามใช้ตอนแข่ง (เช็ก market == sandbox แล้วหยุดเองถ้าไม่ใช่)

    python autoplay.py --games 3

ต้องรัน agent ตัวซ้อมคู่กัน (practice/main.py --forever) · สคริปต์นี้ไม่ส่งข้อเสนอ แค่กดข้าม
กดข้ามเมื่อ:  APPRAISAL -> หลังส่งราคาประเมินแล้ว · ROUND_OPEN -> หลังตลาดเห็นข้อเสนอของเรา (หรือรอเกิน WAIT วิ)
              ช่วงอื่น (SETUP/CLEARING) -> ทันที · CLOSED -> เก็บ reveal + เก็บสถานะทุกรอบ แล้วเริ่มเกมใหม่
"""
import json
import os
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
import httpx  # noqa: E402
from main import load_env  # noqa: E402

load_env(PRACTICE / ".env")
URL, KEY = os.environ["MARKET_URL"].rstrip("/"), os.environ["TEAM_KEY"]
H = {"Authorization": f"Bearer {KEY}"}
WAIT = float(sys.argv[sys.argv.index("--wait") + 1]) if "--wait" in sys.argv else 8   # วิ (ใช้ LLM ให้ใส่ --wait 60): รอบไหนไม่มีข้อเสนอของเรา (เช่นไม่มีคู่ผ่านสเปก) รอเท่านี้แล้วข้าม
GAMES = int(sys.argv[sys.argv.index("--games") + 1]) if "--games" in sys.argv else 1
OUT = PRACTICE / "logs" / "autoplay"
OUT.mkdir(parents=True, exist_ok=True)


def get(path):
    return httpx.get(URL + path, headers=H, timeout=10).json()


def post(path):
    r = httpx.post(URL + path, headers=H, timeout=10)
    return r.status_code


def agent_appraised_since(t0: float) -> bool:
    """log ของ agent มี appraisal_sent หลังเวลา t0 ไหม (flag appraisal_submitted ของตลาดเชื่อไม่ได้ตอนกดข้ามเร็ว)"""
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0))
    try:
        lines = (PRACTICE / "logs" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[-30:]
    except OSError:
        return False
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("event") == "appraisal_sent" and r.get("time", "") >= since:
            return True
    return False


played, seen_at, states = 0, {}, []
print(f"autoplay: {GAMES} เกม · ตลาด {URL}")
first = get("/v1/state")
if first.get("market") != "sandbox":                         # ด่านกัน: ตลาดแข่งห้ามกดข้าม/เริ่มใหม่เด็ดขาด
    sys.exit(f"หยุด: ตลาดนี้คือ '{first.get('market')}' ไม่ใช่ตลาดฝึก (sandbox) · autoplay ใช้ได้แค่ตอนซ้อม")
if first["phase"] == "CLOSED":                    # เกมเก่าจบแล้ว: เริ่มเกมใหม่ก่อน
    print("reset:", post("/v1/sandbox/reset"), flush=True)
    time.sleep(3)
while played < GAMES:
    try:
        s = get("/v1/state")
    except Exception as e:  # noqa: BLE001
        print("อ่านตลาดไม่ได้:", e); time.sleep(2); continue
    if s.get("market") != "sandbox":
        sys.exit(f"หยุด: ตลาดเปลี่ยนเป็น '{s.get('market')}' · autoplay ใช้ได้แค่ตลาดฝึก")
    ph = s["phase"]
    rid = (s.get("round") or {}).get("round_id")
    key = (ph, rid)
    seen_at.setdefault(key, time.time())
    waited = time.time() - seen_at[key]

    go = False
    if ph == "APPRAISAL":
        go = agent_appraised_since(seen_at[key] - 2) or waited > 60   # รอจน log ของ agent บอกว่าส่งแล้ว
    elif ph == "ROUND_OPEN":
        if waited < 0.3:
            states.append(s)                                   # เก็บสถานะต้นรอบไว้วิเคราะห์ (สเปก RFQ ฯลฯ)
        mine = [o for o in s.get("my_open_offers", []) if o["round_id"] == rid]
        go = bool(mine) and waited > 1.0 or waited > WAIT
    elif ph == "CLOSED":
        played += 1
        tag = time.strftime("%m%d-%H%M%S")
        rev = get("/v1/sandbox/reveal")
        (OUT / f"reveal-{tag}.json").write_text(json.dumps(rev, ensure_ascii=False, indent=1), encoding="utf-8")
        (OUT / f"states-{tag}.json").write_text(json.dumps(states, ensure_ascii=False), encoding="utf-8")
        sc = rev.get("score", {})
        st = s.get("standings") or {}
        print(f"=== จบเกม {played}: PR {sc.get('pr')} · อันดับ {sc.get('rank')} · ขาย {st.get('kg_sold')}/{st.get('kg_total')} kg"
              f" · ประเมิน {sc.get('appraisal_score')} · เก็บ reveal-{tag}.json", flush=True)
        states, seen_at = [], {}
        if played < GAMES:
            time.sleep(3)                                      # ให้ agent บันทึก market_closed ก่อน
            print("reset:", post("/v1/sandbox/reset"), flush=True)
            time.sleep(3)
        continue
    else:                                                       # SETUP, CLEARING, ...
        go = waited > 0.5

    if go:
        code = post("/v1/sandbox/advance")
        print(time.strftime("%H:%M:%S"), f"{ph} {rid or ''} -> ข้าม ({code})", flush=True)
        time.sleep(1.2)                                         # ให้ agent (อ่านทุก 1 วิ) เห็นช่วงใหม่
    else:
        time.sleep(0.4)
