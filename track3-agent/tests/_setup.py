"""ให้ไฟล์ในโฟลเดอร์ tests import โค้ดจริงจาก Coffee-TESA-starter ได้ (ไม่ copy ไม่แก้ไฟล์ต้นฉบับ)"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
REAL = HERE.parent / "Coffee-TESA-starter"      # ไฟล์แข่งจริง
PRACTICE = HERE / "practice"                     # ตัวซ้อม (สำเนา) แก้ได้ตามใจ
SNAP = HERE / "snapshots"
# ค่าเริ่มต้น = เทสตัวซ้อม · ใส่ --real เพื่อเทสไฟล์แข่งจริง
if "--real" in sys.argv:
    sys.argv.remove("--real")
    STARTER = REAL
else:
    STARTER = PRACTICE if (PRACTICE / "strategy.py").exists() else REAL
sys.path.insert(0, str(STARTER))
print("โค้ดที่เทส:", STARTER.name if STARTER == REAL else "tests/practice (ตัวซ้อม)")
try:
    sys.stdout.reconfigure(encoding="utf-8")   # ให้ print ภาษาไทยบน Windows ได้
except Exception:
    pass


def load_snapshot(name: str | None = None) -> tuple[dict, dict]:
    """คืน (calendar_info, state). ใช้ snapshot จริงอันล่าสุดที่มี RFQ (จับตอน ROUND_OPEN) ไม่งั้นใช้ example.json"""
    if name is None:
        path = SNAP / "example.json"
        for p in sorted(SNAP.glob("snap-*.json"), reverse=True):
            if json.loads(p.read_text(encoding="utf-8"))["state"]["rfqs"]:
                path = p
                break
        print("snapshot:", path.name)
    else:
        path = SNAP / name
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["calendar"], data["state"]
