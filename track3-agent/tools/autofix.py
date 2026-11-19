"""แก้ค่าคงที่ใน strategy.py แบบมีกันพลาด (ใช้โดย Claude ระหว่างแข่ง) · ไม่ส่งข้อเสนอ ไม่แตะตลาด ไม่แตะ .env/log

    cd D:\\TESA_TGR_2026\\track3-agent\\tests
    ..\\Coffee-TESA-starter\\.venv\\Scripts\\python.exe ..\\tools\\autofix.py status  [--real]
    ..\\Coffee-TESA-starter\\.venv\\Scripts\\python.exe ..\\tools\\autofix.py set MOVE_UP 0 --why "R9 ของค้าง ข้อ C" [--real]
    ..\\Coffee-TESA-starter\\.venv\\Scripts\\python.exe ..\\tools\\autofix.py set KNOB_MAX '{"COMMERCIAL": 0.97}' --why "..." [--real]
    ..\\Coffee-TESA-starter\\.venv\\Scripts\\python.exe ..\\tools\\autofix.py revert --why "แก้แล้วแย่ลง" [--real]

ไม่ใส่ --real = แก้ตัวซ้อม tests/practice/strategy.py · --real = ไฟล์แข่ง Coffee-TESA-starter/strategy.py
main.py (ที่มี reload_strategy) โหลดไฟล์ใหม่เองตอนต้นรอบถัดไป ไม่ต้องรีสตาร์ท

ขั้นตอนของ set (ล้มขั้นไหน = ไฟล์จริงไม่เปลี่ยน):
  1. ชื่อต้องอยู่ใน ALLOWED และค่าต้องอยู่ในช่วง
  2. ห้ามแก้ถี่: ต้องมีอย่างน้อย MIN_ROUNDS_BETWEEN รอบ (offers_sent ใน log) หลังการแก้ครั้งก่อน (--force ข้ามได้)
  3. สร้างไฟล์ใหม่ในโฟลเดอร์ชั่วคราว แล้วรันชุดทดสอบกับไฟล์นั้น (TEST_STRATEGY_FILE)
  4. สำรองไฟล์เดิมไว้ใน bk/ แล้วเขียนไฟล์ใหม่แบบ atomic (os.replace: agent ไม่มีวันเห็นไฟล์ครึ่งๆ)
  5. รันชุดทดสอบกับไฟล์จริงอีกรอบ · ไม่ผ่าน = คืนไฟล์สำรองทันที
  6. จดลง tools/CHANGELOG-day20.md และ tools/autofix_state.json
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
TESTS = ROOT / "tests"
BK = ROOT / "bk"
CHANGELOG = TOOLS / "CHANGELOG-day20.md"
STATE = TOOLS / "autofix_state.json"
TARGETS = {"practice": TESTS / "practice", "real": ROOT / "Coffee-TESA-starter"}
MIN_ROUNDS_BETWEEN = 2
TYPES = ("SPECIALTY", "COMMERCIAL", "INSTANT")


def _float(lo, hi):
    def check(v):
        x = float(v)
        if not lo <= x <= hi:
            raise ValueError(f"ต้องอยู่ในช่วง {lo}–{hi}")
        return x
    return check


def _bool(v):
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise ValueError("ต้องเป็น true/false หรือ 1/0")


def _knob_max(v):
    d = json.loads(v) if isinstance(v, str) else v
    if not isinstance(d, dict):
        raise ValueError('ต้องเป็น dict เช่น {"COMMERCIAL": 0.97}')
    out = {}
    for k, x in d.items():
        if k not in TYPES:
            raise ValueError(f"ประเภทผู้ซื้อ {k!r} ไม่รู้จัก (ใช้ {', '.join(TYPES)})")
        out[k] = _float(0.85, 1.30)(x)
    return dict(sorted(out.items()))


ALLOWED = {                       # ชื่อ -> ตัวตรวจค่า · แก้ได้แค่นี้ (ตามที่ผู้ใช้เลือก: เฉพาะค่าคงที่)
    "KNOB_MAX": _knob_max,        # เพดานตัวคูณหลักต่อประเภท 0.85–1.30
    "MOVE_UP": _float(0.0, 0.02),  # ขึ้นราคาได้ต่อรอบ (ลดได้อย่างเดียว ไม่ให้เกินค่าเดิม 0.02)
    "FESTIVAL_HOLD": _bool,
    "CLEAN_RAISE": _bool,
}


def line_re(name: str) -> re.Pattern:
    return re.compile(rf"^({re.escape(name)} = )([^#\n]*?)(\s*#.*)?$", re.M)


def read_values(text: str) -> dict:
    out = {}
    for name in ALLOWED:
        found = line_re(name).findall(text)
        out[name] = ast.literal_eval(found[0][1].strip()) if len(found) == 1 else "<หาไม่เจอ/ซ้ำ>"
    return out


def set_value(text: str, name: str, value) -> str:
    pat = line_re(name)
    if len(pat.findall(text)) != 1:
        raise SystemExit(f"❌ หาบรรทัด '{name} = ...' ไม่เจอหรือมีมากกว่า 1 บรรทัด: ไม่แก้")
    return pat.sub(lambda m: m.group(1) + repr(value) + (m.group(3) or ""), text, count=1)


def run_tests(target: str, strategy_file: Path | None) -> tuple[bool, str]:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("TEST_STRATEGY_FILE", None)
    if strategy_file is not None:
        env["TEST_STRATEGY_FILE"] = str(strategy_file)
    cmd = [sys.executable, "test_strategy.py"] + (["--real"] if target == "real" else [])
    p = subprocess.run(cmd, cwd=TESTS, env=env, capture_output=True, text=True, encoding="utf-8", timeout=300)
    out = (p.stdout + p.stderr).strip().splitlines()
    summary = next((l for l in reversed(out) if l.startswith("Ran ")), "?")
    ok = p.returncode == 0
    return ok, summary + (" OK" if ok else " FAILED\n" + "\n".join(out[-25:]))


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def log_rows(target: str) -> list:
    try:
        lines = (TARGETS[target] / "logs" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()[-3000:]
    except OSError:
        return []
    rows = []
    for l in lines:
        try:
            rows.append(json.loads(l))
        except ValueError:
            pass
    return rows


def rounds_since(target: str, iso: str | None) -> tuple[int, str]:
    """จำนวนรอบที่ agent ส่งข้อเสนอหลังเวลา iso (ในเกมปัจจุบัน) และรอบล่าสุด"""
    rows = log_rows(target)
    games = [i for i, r in enumerate(rows) if r.get("event") == "new_game"]
    game_start = rows[games[-1]]["time"] if games else ""
    sent = [r for r in rows if r.get("event") == "offers_sent" and r.get("time", "") >= game_start]
    last = sent[-1]["round"] if sent else "-"
    if not iso or iso < game_start:
        return 99, last
    return sum(1 for r in sent if r["time"] > iso), last


def changelog(lines: list[str]) -> None:
    new = not CHANGELOG.exists()
    with open(CHANGELOG, "a", encoding="utf-8") as f:
        if new:
            f.write("# บันทึกการแก้โค้ดระหว่างแข่ง ด่าน 3\n\n")
        f.write("\n".join(lines) + "\n\n")


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".autofix-tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def cmd_status(target: str) -> None:
    path = TARGETS[target] / "strategy.py"
    st = load_state().get(target, {})
    n, last = rounds_since(target, st.get("last_change"))
    print(f"ไฟล์: {path}")
    for k, v in read_values(path.read_text(encoding="utf-8")).items():
        print(f"  {k} = {v!r}")
    print(f"รอบล่าสุดที่ส่ง: {last} · แก้ครั้งก่อน: {st.get('last_change', '-')} ({st.get('last_what', '-')})"
          f" · ผ่านไป {n if n != 99 else '-'} รอบ · แก้ได้อีก: {'✅' if n >= MIN_ROUNDS_BETWEEN else '❌ รออีก ' + str(MIN_ROUNDS_BETWEEN - n) + ' รอบ'}")


def cmd_set(target: str, name: str, raw: str, why: str, force: bool) -> None:
    if name not in ALLOWED:
        raise SystemExit(f"❌ แก้ได้แค่ {', '.join(ALLOWED)}")
    try:
        value = ALLOWED[name](raw)
    except (ValueError, TypeError) as e:
        raise SystemExit(f"❌ ค่า {name}={raw!r} ใช้ไม่ได้: {e}")
    if not why.strip():
        raise SystemExit("❌ ต้องใส่ --why \"เหตุผล\" (กรรมการอาจถาม)")
    path = TARGETS[target] / "strategy.py"
    old_text = path.read_text(encoding="utf-8")
    old = read_values(old_text)[name]
    if old == value:
        raise SystemExit(f"ไม่ต้องแก้: {name} = {value!r} อยู่แล้ว")
    st_all = load_state()
    st = st_all.get(target, {})
    n, last = rounds_since(target, st.get("last_change"))
    if n < MIN_ROUNDS_BETWEEN and not force:
        raise SystemExit(f"❌ เพิ่งแก้ไป {n} รอบ (ต้องรอ {MIN_ROUNDS_BETWEEN} รอบ ให้เห็นผลก่อน) · ใช้ --force ถ้าจำเป็นจริง")
    new_text = set_value(old_text, name, value)

    with tempfile.TemporaryDirectory() as td:          # 3. ทดสอบไฟล์ใหม่ก่อนแตะไฟล์จริง
        cand = Path(td) / "strategy.py"
        cand.write_text(new_text, encoding="utf-8")
        ok, res = run_tests(target, cand)
    if not ok:
        raise SystemExit(f"❌ ชุดทดสอบไม่ผ่านกับค่าใหม่ · ไฟล์จริงไม่เปลี่ยน\n{res}")

    now = datetime.now()
    BK.mkdir(exist_ok=True)
    backup = BK / f"strategy_{target}_{now:%m%d-%H%M%S}.py"
    shutil.copy2(path, backup)                          # 4.
    atomic_write(path, new_text)
    ok2, res2 = run_tests(target, None)                 # 5.
    if not ok2:
        atomic_write(path, old_text)
        changelog([f"## {now:%H:%M:%S} ❌ ยกเลิก ({target}) {name}: {old!r} -> {value!r}",
                   f"- เหตุผล: {why}", f"- ชุดทดสอบกับไฟล์จริงไม่ผ่าน -> คืนไฟล์เดิมแล้ว", f"- {res2.splitlines()[0]}"])
        raise SystemExit(f"❌ ทดสอบไฟล์จริงไม่ผ่าน -> คืนไฟล์เดิมแล้ว\n{res2}")

    hist = st.get("history", [])
    hist.append({"time": now.isoformat(timespec="seconds"), "name": name, "old": old, "new": value,
                 "backup": str(backup), "why": why, "round": last})
    st.update(last_change=now.isoformat(timespec="seconds"), last_what=f"{name}={value!r}", history=hist)
    st_all[target] = st
    save_state(st_all)
    changelog([f"## {now:%H:%M:%S} ({target}) {name}: {old!r} -> {value!r}",
               f"- เหตุผล: {why}",
               f"- รอบล่าสุดที่ส่งก่อนแก้: {last} · มีผลตั้งแต่รอบถัดไป (main.py โหลดใหม่เอง)",
               f"- สำรอง: `{backup.relative_to(ROOT)}` · ชุดทดสอบ: {res2.splitlines()[0]}"])
    print(f"✅ {name}: {old!r} -> {value!r} ({target}) · ทดสอบ {res2.splitlines()[0]}")
    print(f"   สำรอง {backup.relative_to(ROOT)} · มีผลตั้งแต่รอบถัดไป (ดู log: strategy_reloaded)")


def cmd_revert(target: str, why: str) -> None:
    st_all = load_state()
    st = st_all.get(target, {})
    hist = st.get("history", [])
    if not hist:
        raise SystemExit("ไม่มีการแก้ให้ย้อน")
    h = hist[-1]
    path = TARGETS[target] / "strategy.py"
    backup = Path(h["backup"])
    text = backup.read_text(encoding="utf-8")
    now = datetime.now()
    keep = BK / f"strategy_{target}_{now:%m%d-%H%M%S}_before_revert.py"
    shutil.copy2(path, keep)
    atomic_write(path, text)
    ok, res = run_tests(target, None)
    hist.pop()
    st.update(history=hist, last_change=now.isoformat(timespec="seconds"), last_what=f"revert {h['name']}")
    st_all[target] = st
    save_state(st_all)
    changelog([f"## {now:%H:%M:%S} ↩ ย้อน ({target}) {h['name']}: {h['new']!r} -> {h['old']!r}",
               f"- เหตุผล: {why or '-'}", f"- คืนจาก `{backup.relative_to(ROOT)}` · ชุดทดสอบ: {res.splitlines()[0]}"])
    print(f"{'✅' if ok else '⚠️'} ย้อน {h['name']} -> {h['old']!r} · ทดสอบ {res.splitlines()[0]}")


def main() -> None:
    args = sys.argv[1:]
    target = "real" if "--real" in args else "practice"
    force = "--force" in args
    why = ""
    if "--why" in args:
        i = args.index("--why")
        why = args[i + 1] if i + 1 < len(args) else ""
        del args[i:i + 2]
    args = [a for a in args if a not in ("--real", "--force")]
    if not args or args[0] == "status":
        cmd_status(target)
    elif args[0] == "set" and len(args) == 3:
        cmd_set(target, args[1], args[2], why, force)
    elif args[0] == "revert":
        cmd_revert(target, why)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
