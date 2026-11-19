"""เทส reload_strategy() ใน main.py: แก้ strategy.py ระหว่างรัน -> รอบถัดไปใช้ไฟล์ใหม่ · ไฟล์พัง -> ใช้ตัวเดิมต่อ

    python test_reload.py          # main.py ตัวซ้อม (tests/practice)
    python test_reload.py --real   # main.py ตัวจริง (Coffee-TESA-starter)

ไม่ต่อเน็ต · ไม่แตะ strategy.py/log จริง (ใช้สำเนาในโฟลเดอร์ชั่วคราว)
"""
import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path

import _setup  # noqa: F401  (ใส่ path ของ starter ให้)
import main as M

M.LOGS = Path(tempfile.mkdtemp())          # log ของเทสไม่ปน log จริง


class TestReload(unittest.TestCase):
    def setUp(self):
        if not hasattr(M, "reload_strategy"):
            self.skipTest("main.py นี้ยังไม่มี reload_strategy")
        self.orig = (M.strategy, M._STRATEGY_PATH, M._strategy_mtime)
        self.src = Path(M.strategy.__file__).read_text(encoding="utf-8")
        self.f = Path(tempfile.mkdtemp()) / "strategy.py"
        self.f.write_text(self.src, encoding="utf-8")
        M._STRATEGY_PATH, M._strategy_mtime = self.f, self.f.stat().st_mtime

    def tearDown(self):
        M.strategy, M._STRATEGY_PATH, M._strategy_mtime = self.orig
        sys.modules["strategy"] = M.strategy

    def _write(self, text):
        time.sleep(0.02)
        self.f.write_text(text, encoding="utf-8")
        os.utime(self.f, None)

    def _events(self):
        p = M.LOGS / "decisions.jsonl"
        return re.findall(r'"event": "(\w+)"', p.read_text(encoding="utf-8")) if p.exists() else []

    def test_no_change_no_reload(self):
        before = M.strategy
        M.reload_strategy()
        self.assertIs(M.strategy, before)

    def test_good_edit_is_picked_up(self):
        self._write(re.sub(r"^MOVE_UP = .*$", "MOVE_UP = 0.013", self.src, flags=re.M))
        M.reload_strategy()
        self.assertEqual(M.strategy.MOVE_UP, 0.013)
        self.assertIs(sys.modules["strategy"], M.strategy)
        self.assertEqual(self._events()[-1], "strategy_reloaded")

    def test_broken_edit_keeps_old(self):
        self._write(re.sub(r"^MOVE_UP = .*$", "MOVE_UP = 0.011", self.src, flags=re.M))
        M.reload_strategy()
        good = M.strategy
        self._write(self.src + "\ndef broken(:\n")
        M.reload_strategy()
        self.assertIs(M.strategy, good)
        self.assertEqual(M.strategy.MOVE_UP, 0.011)
        self.assertEqual(self._events()[-1], "strategy_reload_failed")
        n = len(self._events())
        M.reload_strategy()                                   # ไฟล์ไม่เปลี่ยน -> ไม่ลองซ้ำ ไม่เขียน log ซ้ำ
        self.assertEqual(len(self._events()), n)

    def test_missing_function_keeps_old(self):
        before = M.strategy
        self._write(self.src.replace("def decide(", "def decide_renamed("))
        M.reload_strategy()
        self.assertIs(M.strategy, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
