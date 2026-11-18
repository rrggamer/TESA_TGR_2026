"""ดึง /v1/calendar + /v1/state ของจริงมาเก็บเป็น snapshot (GET อย่างเดียว: ไม่ส่ง offer ไม่ส่ง appraisal)

    python capture.py      # ใช้ MARKET_URL / TEAM_KEY จาก Coffee-TESA-starter/.env
เก็บตอน phase = ROUND_OPEN จะได้มี RFQ ให้เทส
"""
import json
import os
from datetime import datetime

from _setup import SNAP, STARTER
from main import load_env
from market import Market

load_env(STARTER / ".env")
api = Market(os.environ["MARKET_URL"], os.environ["TEAM_KEY"])
data = {"calendar": api.calendar(), "state": api.state()}
path = SNAP / f"snap-{datetime.now():%m%d-%H%M%S}.json"
path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved", path.name, "| phase:", data["state"]["phase"], "| lots:", len(data["state"]["lots"]),
      "| rfqs:", len(data["state"]["rfqs"]))
