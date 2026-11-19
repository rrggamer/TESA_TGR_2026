"""หน้าจอดูสถานะระหว่างแข่ง — อ่านอย่างเดียว ไม่ส่งข้อเสนอ ไม่กดอะไรในตลาด

    cd D:\\TESA_TGR_2026\\track3-agent\\Coffee-TESA-starter
    .venv\\Scripts\\python ..\\tools\\watch.py              # ดู agent ตัวจริง (logs/ ของโฟลเดอร์นี้)
    .venv\\Scripts\\python ..\\tools\\watch.py --logs ..\\tests\\practice\\logs --env ..\\tests\\practice\\.env

อ่าน logs/decisions.jsonl + memory-<ตลาด>-*.json ของ agent · เรียก GET /v1/state ทุก 15 วิ (ไม่เกิน 1 คำขอ/15 วิ
ไกลจากเพดาน 5 คำขอ/วิ) เพื่อดูสต็อกที่เหลือและเวลา · ใช้ --no-api ถ้าไม่อยากเรียกตลาดเลย
คำแนะนำในหน้าจอเป็นแค่ข้อเสนอ: คนตัดสินใจเองว่าจะให้แก้โค้ดแล้วรีสตาร์ท agent หรือไม่ (ดู PLAYBOOK.md)
"""
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ARG = sys.argv
HERE = Path.cwd()
LOGS = Path(ARG[ARG.index("--logs") + 1]) if "--logs" in ARG else HERE / "logs"
ENV = Path(ARG[ARG.index("--env") + 1]) if "--env" in ARG else HERE / ".env"
EVERY = float(ARG[ARG.index("--every") + 1]) if "--every" in ARG else 15
USE_API = "--no-api" not in ARG
ONCE = "--once" in ARG


def env() -> dict:
    out = {}
    try:
        for line in ENV.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def read_log() -> list:
    try:
        lines = (LOGS / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[-3000:]
    except OSError:
        return []
    rows = []
    for l in lines:
        try:
            rows.append(json.loads(l))
        except ValueError:
            pass
    games = [i for i, r in enumerate(rows) if r.get("event") == "new_game"]
    return rows[games[-1]:] if games else rows


def read_memory(market: str):
    files = glob.glob(str(LOGS / f"memory-{market}-*.json"))
    if not files:
        return None
    try:
        return json.loads(Path(max(files, key=os.path.getmtime)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def get_state(e: dict):
    if not USE_API or not e.get("MARKET_URL") or not e.get("TEAM_KEY"):
        return None
    try:
        import httpx
        r = httpx.get(e["MARKET_URL"].rstrip("/") + "/v1/state", headers={"Authorization": f"Bearer {e['TEAM_KEY']}"}, timeout=8)
        return r.json() if r.status_code == 200 else {"_error": f"HTTP {r.status_code}"}
    except Exception as ex:  # noqa: BLE001
        return {"_error": str(ex)[:80]}


def round_rows(mem: dict) -> list:
    out = []
    for rnd in mem.get("rounds", []):
        rid = rnd["round_id"]
        info = {x["rfq_id"]: x for x in rnd.get("market_summary", [])}
        sent = {(o["rfq_id"], o["lot_id"]): o for o in mem.get("sent", {}).get(rid, [])}
        st, mk, by_type = Counter(), [], defaultdict(lambda: {"sold": 0, "kg": 0, "st": Counter(), "mk": []})
        for o in rnd.get("my_offers", []):
            t = info.get(o["rfq_id"], {}).get("buyer_type", "?")
            s = sent.get((o["rfq_id"], o["lot_id"]), {})
            st[o["status"]] += o["kg"]
            b = by_type[t]; b["kg"] += o["kg"]; b["sold"] += o["filled_kg"]; b["st"][o["status"]] += o["kg"]
            if "markup" in s:
                b["mk"].append(s["markup"])
                if o["filled_kg"]:
                    mk.append(s["markup"])
        market = defaultdict(lambda: [0, 0, [], []])
        for x in rnd.get("market_summary", []):
            m = market[x["buyer_type"]]; m[0] += x["wanted_kg"]; m[1] += x["bought_kg"]
            if x.get("price_avg") is not None:
                m[2].append(x["price_avg"]); m[3].append(x["price_max"])
        out.append({"rid": rid, "kind": rnd.get("kind"), "offered": sum(st.values()),
                    "sold": sum(o["filled_kg"] for o in rnd.get("my_offers", [])), "st": st,
                    "mk": sum(mk) / len(mk) if mk else None, "by_type": by_type, "market": market})
    return out


def advice(rows: list, state, notes_last: list) -> list:
    """กฎตัดสินใจที่เขียนไว้ล่วงหน้า (PLAYBOOK.md) — แค่แนะนำ"""
    tips = []
    real = [r for r in rows if r["kind"] != "practice"]
    if len(real) >= 2 and real[-1]["offered"] and real[-1]["sold"] == 0 and real[-2]["sold"] == 0:
        tips.append("🔴 ขายไม่ได้เลย 2 รอบติด → ข้อ A ใน PLAYBOOK: ให้ Claude ใส่ KNOB_MAX ใน strategy.py = ตัวคูณที่เสนอล่าสุด − 0.03")
    elif real and real[-1]["offered"] and real[-1]["sold"] == 0:
        tips.append("🟠 รอบล่าสุดขายไม่ได้เลย (ยังไม่ต้องทำอะไร agent ลดราคาเอง) · ถ้ารอบหน้าเป็นอีก → ข้อ A")
    for r in real[-1:]:
        for t, b in r["by_type"].items():
            lost = b["st"]["OUTBID"] + b["st"]["ABOVE_LIMIT"]
            if b["kg"] >= 120 and lost >= 0.8 * b["kg"]:
                tips.append(f"🟠 {t}: แพ้ {lost}/{b['kg']} kg รอบล่าสุด (ตัวคูณที่เสนอ {min(b['mk']):.3f}–{max(b['mk']):.3f})")
    if state and not state.get("_error") and state.get("round"):
        rnd = state["round"]
        if rnd.get("kind") == "real":
            left = rnd["of"] - rnd["number"] + 1
            stock = sum(l["kg_remaining"] for l in state.get("lots", []))
            cap = state.get("round_capacity_kg") or 0
            if cap and stock > 0.7 * cap * max(left - 2, 1) and left <= 6:
                tips.append(f"🟠 ของเหลือ {stock} kg แต่เหลือ {left} รอบ (รอบท้ายความต้องการต่ำ) → ข้อ C ใน PLAYBOOK")
    if any(n.startswith("llm failed") for n in notes_last):
        tips.append("ℹ️ LLM ล่มรอบนี้ agent ขายด้วยกฎ (ปกติ ไม่ต้องทำอะไร · ห้ามสลับไปใช้ LLM อื่น · ข้อ E)")
    return tips or ["🟢 ปกติ ไม่ต้องทำอะไร"]


def render():
    e = env()
    log = read_log()
    market = ARG[ARG.index("--market") + 1] if "--market" in ARG else next((r.get("market") for r in reversed(log) if r.get("event") == "new_game"), "competition")
    mem = read_memory(market) or {"rounds": [], "sent": {}}
    state = get_state(e)
    rows = round_rows(mem)
    sent_ev = [r for r in log if r.get("event") == "offers_sent"]
    errs = [r for r in log if r.get("event") in ("state_error", "strategy_crashed", "batch_rejected", "offer_rejected", "appraisal_error")]
    out = [f"=== {time.strftime('%H:%M:%S')} · ตลาด {market} · log {LOGS} ==="]
    knobs = {k: v for k, v in e.items() if k.startswith("KNOB_") and v}
    out.append(f".env: USE_LLM={e.get('USE_LLM')} · ปุ่มปรับ: {knobs or 'ไม่มี (ค่าเดิม)'}")
    if state:
        if state.get("_error"):
            out.append(f"⚠ อ่านตลาดไม่ได้: {state['_error']}")
        else:
            rnd = state.get("round") or {}
            stock = sum(l["kg_remaining"] for l in state.get("lots", []))
            total = sum(l["kg_total"] for l in state.get("lots", []))
            out.append(f"ตลาด: {state.get('phase')} {rnd.get('round_id', '')} ({rnd.get('calendar', '')}) · เหลือ {state.get('seconds_left')} วิ"
                       f" · สต็อก {stock}/{total} kg ({stock * 100 // max(total, 1)}%) · ลิมิตรอบ {state.get('round_capacity_kg')} kg"
                       + (" · ⏸ ตลาดหยุดชั่วคราว" if state.get("paused") else ""))
    appr = any(r.get("event") == "appraisal_sent" for r in log)
    out.append(f"ส่งผลประเมิน: {'✅' if appr else '❌ ยังไม่ส่ง'} · ส่งข้อเสนอแล้ว {len(sent_ev)} รอบ · error {len(errs)}"
               + (f" (ล่าสุด {errs[-1]['time'][11:19]} {errs[-1]['event']})" if errs else ""))
    out.append("")
    out.append(f"{'รอบ':5s} {'เสนอ':>5s} {'ขาย':>5s} {'ตัวคูณขาย':>9s}  ผลของเรา (kg)                        | ตลาด: ผู้ซื้อได้ของ % · ราคาเฉลี่ย ฿")
    for r in rows[-10:]:
        st = " ".join(f"{k[:5]}{v}" for k, v in r["st"].items())
        mk = f"{r['mk']:.3f}" if r["mk"] else "  -  "
        mkt = " ".join(f"{t[:4]} {m[1] * 100 // max(m[0], 1)}% {sum(m[2]) / len(m[2]):.0f}" if m[2] else f"{t[:4]} {m[1] * 100 // max(m[0], 1)}%"
                       for t, m in sorted(r["market"].items()))
        flag = " 🔴" if r["offered"] and r["sold"] == 0 and r["kind"] != "practice" else ""
        out.append(f"{r['rid']:5s} {r['offered']:5d} {r['sold']:5d} {mk:>9s}  {st:36s} | {mkt}{flag}")
    if sent_ev:
        last = sent_ev[-1]
        out.append("")
        out.append(f"ล่าสุด {last['time'][11:19]} {last['round']}: {last['source']} · {last['n']} ข้อเสนอ {last['kg']} kg")
        for n in last.get("notes", [])[:6]:
            if n.startswith(("llm", "shift", "knob", "move cap", "festival")):
                out.append("   " + n[:150])
    out.append("")
    out += advice(rows, state, sent_ev[-1].get("notes", []) if sent_ev else [])
    return "\n".join(out)


if __name__ == "__main__":
    while True:
        text = render()
        if not ONCE:
            os.system("cls" if os.name == "nt" else "clear")
        print(text, flush=True)
        if ONCE:
            break
        time.sleep(EVERY)
