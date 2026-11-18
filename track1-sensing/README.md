# Track 1 — Sensing (ESP32-S3-WROOM-1 N16R8)

PlatformIO project used to verify the toolchain ↔ board link before real sensing code lands.

## The `pio` command

Fixed on 2026-11-17: `~/.local/bin/pio` is a symlink to the working **PlatformIO Core 6.2.0**
(`~/.platformio/penv/bin/pio`), and `~/.local/bin` comes first in `PATH`, so plain `pio` works:

```bash
pio run -e esp32s3_usb
```

Background: the apt package `platformio` 4.3.4-3 installs a **broken** `/usr/bin/pio`
(`AttributeError: 'PlatformioCLI' object has no attribute 'resultcallback'` — incompatible
`click` version) and is far too old for ESP32-S3 anyway. 6.2.0 is the latest release on PyPI,
so nothing needs upgrading — only the `PATH` shadowing had to be resolved.

Still to do (needs sudo, run it yourself):

```bash
sudo apt remove platformio
```

Nothing depends on that package, so it removes cleanly on its own. **Do not** run
`apt autoremove` afterwards — on this machine it would also pull out `intel-microcode`,
`amd64-microcode` and `thermald`, which are unrelated.

`./pio` in this folder is a fallback wrapper that calls the penv binary directly; it keeps
working even if the symlink or `PATH` changes. Override with `PLATFORMIO_BIN=/path/to/pio`.

## Commands

| Task | Command |
|------|---------|
| Build + upload competition firmware | `pio run -e esp32s3_usb -t upload` |
| Build + upload calibration firmware | `pio run -e esp32s3_cal -t upload` |
| Serial monitor (9600 baud) | `pio device monitor -e esp32s3_usb` |
| Clean | `pio run -t clean` |
| List ports | `pio device list` |

Board as detected by `pio device list`:

```
/dev/ttyACM0
Hardware ID: USB VID:PID=303A:1001 SER=E0:72:A1:F3:CE:90 LOCATION=5-1:1.0
Description: USB JTAG/serial debug unit
```

## Environments

| Env | Purpose |
|-----|---------|
| `esp32s3_usb` (default) | **Competition firmware.** Prints only the required format. Native USB port `/dev/ttyACM0` (USB ID `303a:1001`), `ARDUINO_USB_CDC_ON_BOOT=1`. |
| `esp32s3_cal` | **Calibration firmware.** Same measurement, but prints raw timing stats (`t`, `sd`, `min`, `max`, `C`, elapsed ms). Also triggers on any key sent over serial, so you can measure without pressing the button. |
| `esp32s3_uart` | Competition firmware over the UART bridge port (`/dev/ttyUSB0`), `ARDUINO_USB_CDC_ON_BOOT=0`. |

All envs: `monitor_speed = 9600` (rule 3.2) and `CORE_DEBUG_LEVEL=0` so no ESP log lines
leak into the judged output.

Native USB CDC ignores the baud setting (any terminal speed works); if judges need a genuine
9600-baud UART, use `esp32s3_uart` on the bridge port.

## N16R8 configuration

`esp32-s3-devkitc-1` ships as an N8 (8 MB, no PSRAM) board definition, so `platformio.ini`
overrides it:

```ini
board_build.arduino.memory_type = qio_opi   ; QIO flash + OPI (octal) PSRAM
board_build.partitions = default_16MB.csv
board_upload.flash_size = 16MB
board_upload.maximum_size = 16777216
build_flags = -DBOARD_HAS_PSRAM
```

If `PSRAM: not detected` shows up in the monitor, `memory_type` is the first thing to check.

## What `src/main.cpp` does

Competition firmware for **method 1 (inverting comparator)**, chosen in
[docs/competition-requirements.md](docs/competition-requirements.md).

Wiring (LM393 powered from 5 V, everything the ESP32 sees stays at 3.3 V):

```
GPIO5 (CHARGE) ── R 1MΩ ──┬── LM393 pin 2 (IN1−)
                          └── Cx ── GND
3.3V ── trimpot 10k ── GND,  wiper (≈2.08 V) ── LM393 pin 3 (IN1+)   [+ optional 0.1µF to GND]
LM393 pin 1 (OUT1) ── R 10k ── 3.3V      ← NEVER pull up to 5 V
                   └── GPIO6 (COMP)
LM393 pin 8 ── 5V,  pin 4 ── GND,  pin 5 ── GND,  pin 6 ── pin 3    (unused half locked)
GPIO4 (BUTTON) ── switch ── GND           (debounced in software)
GPIO48 = on-board RGB LED
```

Per press-and-release:

1. Software debounce (30 ms stable), fires on **release** only; presses during a measurement
   are discarded (`buttonResync()`).
2. Wait 200 ms so the hand is away from the sensor.
3. 10 warm-up + **500 charge cycles**: discharge 1 ms, then with interrupts off, set CHARGE
   high and count CPU cycles (240 MHz, 4.17 ns/tick) until the comparator pulls COMP low.
   500 µs timeout per cycle.
4. Sort, drop 10 % from each end, take the mean → `t` in µs.
5. `C = CAL_SLOPE_PF_PER_US · t + CAL_OFFSET_PF` (theory with 1 MΩ and Vref = 0.632·Vcc:
   1 µs = 1 pF; replace with the fit from the 5/10/15/20 pF reference caps).
6. Print the exact format, then light the LED white for 200 ms. If fewer than half the cycles
   switched, nothing is printed (no made-up values).

Edit the `CONFIG` block at the top of `src/main.cpp`:

- `TEAM_CODE` — your real team code
- `CAL_SLOPE_PF_PER_US`, `CAL_OFFSET_PF` — from calibration
- `OUTPUT_MOISTURE` — `0` for 18 Nov (pF), `1` for 20 Nov (moisture %), plus the
  `MOIST_*` fit

Output (rule 3.5):

```
"TGR-20"
Capacitance = 10.00 pF

```

The earlier hardware self-test lives in `examples/selftest/main.cpp` (not built).

## Self-test result (2026-11-17, `examples/selftest`)

```
[1] chip : ESP32-S3 rev0, 2 cores @ 240 MHz
    flash: 16.00 MB @ 80 MHz      psram: 8.00 MB
    mac  : E0:72:A1:F3:CE:90      sdk: v4.4.7 / Arduino 2.0.17
[2] boot count: 2 (written + read back 2)  [OK]   -> 3, 4 on later resets
[3] 4 MB write 120 ms (33.3 MB/s), verify 75 ms (53.3 MB/s), mismatches 0  [OK]
[4] SDA=GPIO8 SCL=GPIO9 -> 0 device(s) (bus idle, nothing wired yet)
[5] GPIO48 -> red/green/blue/yellow/off
[6] >> ECHO: HELLO FROM PLATFORMIO
```

RAM 5.9% (19,372 B), Flash 4.5% (297,085 B of 6.25 MB app partition).

## Board quirks (YD-ESP32-S3 style, CH343P + native USB)

The board is a DevKitC-1 clone with two USB-C ports. Two solder pads ship **open** and
both had to be bridged:

| Pad | Symptom while open | After bridging |
|-----|--------------------|----------------|
| `IN-OUT` (next to the 5V pin) | 5V pin only gives ~1.8 V → LM393 unpowered, COMP never switches | 5V pin outputs USB 5 V |
| `RGB` (next to the RGB LED) | GPIO48 not connected to the LED → nothing lights | LED on GPIO48 works |

After bridging `IN-OUT`, never feed external 5 V into the 5V pin while USB is plugged in.

The CH343 port (USB ID `1a86:55d3`) also enumerates as `/dev/ttyACM0` (cdc_acm driver).
Use the `*_uart` envs on that port. The TX/RX LEDs next to it blink on serial traffic and
are not controllable from firmware. At reset the ROM prints a boot log at 115200, seen as
one garbage line at 9600 — open the serial monitor after the board has booted.

## Notes

- The on-board LED is the addressable RGB LED on `GPIO48` (DevKitC-1 v1.1; v1.0 boards use
  GPIO38). Change `LED_PIN` if it does not light — rule 3.4 requires it to be visible.
- Pins to avoid on N16R8: 19/20 (USB), 26–32 (flash), 35–37 (octal PSRAM), 0/3/45/46
  (strapping), 43/44 (UART0).
- Toolchains and platform packages live in the shared `~/.platformio` core dir
  (platform-espressif32 **6.12.0**, Arduino core **2.0.17** / ESP-IDF 4.4.7). Kept as-is
  deliberately: it is the last release of the official `platformio/platform-espressif32`.
  Arduino core 3.x (ESP-IDF 5.x) only exists in the community `pioarduino` fork — switching
  would mean a ~500 MB toolchain download and some libraries not compiling, so it is not
  worth it unless a library demands the 3.x API.
