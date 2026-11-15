# TESA_TGR_2026
**Chumphon Robusta AIoT Platform Toward The Next-Gen Market**

TESA Top Gun Rally 2026 — KMITL Prince of Chumphon Campus

## Tracks

| Track | Folder | Branch | Hardware / Stack | Goal |
|---|---|---|---|---|
| 1. Embedded (ESP32 & Sensor) | [track1-sensing/](track1-sensing/) | `track1-sensing` | ESP32 + protoboard, self-built sensor, soldering | Sensing node that collects sensor data and publishes it to the broker |
| 2. Edge AI (RPi & AI) | [track2-vision/](track2-vision/) | `track2-vision` | Raspberry Pi 4 (4GB+) | Classify coffee beans into ~7 grades (good → bad) using the provided labeled dataset |
| 3. The Next-Gen Market (Python & AI) | [track3-agent/](track3-agent/) | `track3-agent` | Python + Qwen ~9B LLM (provided) via API | AI agent that sets a selling strategy to price coffee above the reference price |
| Pitching | [pitch/](pitch/) | `pitch` | — | Business model |

## System

- **Track 1** reads sensor data → publishes to the broker (likely MQTT, details TBA by organizer)
- **Track 2** Pi camera → bean grading model → publishes results to the broker
- **Track 3** lot/grade data + market formula → LLM analysis → offer price
  - Each team owns multiple coffee lots across several grades; customers choose whom to buy from, and teams compete on price
  - Each grade has a reference price: selling above it adds points, selling below it reduces points on a sliding scale
- **Integration** combines all three tracks into one platform (task announced later)
- Deployment runs on organizer infrastructure; no MATLAB; UI is not a focus

## Branches

```
main  ← merged via Pull Request
 ├── track1-sensing
 ├── track2-vision
 ├── track3-agent
 └── pitch
```

References: [docs/](docs/)
