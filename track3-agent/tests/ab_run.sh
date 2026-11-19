#!/bin/bash
# ทดลองหลายเวอร์ชันในตลาดฝึก (ตัวซ้อมเท่านั้น): bash ab_run.sh <ชื่อ> <เกม> <USE_LLM> "<EXP>"
name=$1; games=$2; llm=$3; exp=$4
cd "$(dirname "$0")/practice"
d="logs/old_memory/$(date +%H%M%S)-$name"; mkdir -p "$d"; mv logs/memory-sandbox-*.json "$d"/ 2>/dev/null   # ตลาดฝึกวนชุดล็อตเดิมได้: ไม่ให้โหลด memory เกมเก่า
USE_LLM=$llm EXP="$exp" STRATEGY_FILE="$STRATEGY_FILE" PYTHONIOENCODING=utf-8 ../../Coffee-TESA-starter/.venv/Scripts/python.exe -u main.py --forever > "logs/ab-$name.log" 2>&1 &
agent=$!
sleep 4
cd ..
PYTHONIOENCODING=utf-8 ../Coffee-TESA-starter/.venv/Scripts/python.exe -u autoplay.py --games $games --wait 30 2>&1 | grep --line-buffered "===" | sed -u "s/^/[$name] /"
kill $agent 2>/dev/null; sleep 2
