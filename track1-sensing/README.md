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
| Build | `pio run -e esp32s3_usb` |
| Upload | `pio run -e esp32s3_usb -t upload` |
| Serial monitor | `pio device monitor -e esp32s3_usb` |
| Clean | `pio run -t clean` |
| List ports | `pio device list` |

Board as detected by `pio device list`:

```
/dev/ttyACM0
Hardware ID: USB VID:PID=303A:1001 SER=E0:72:A1:F3:CE:90 LOCATION=5-1:1.0
Description: USB JTAG/serial debug unit
```

## Environments

- **`esp32s3_usb`** (default) — native USB-OTG port, enumerates as `/dev/ttyACM0`
  (USB ID `303a:1001`, "Espressif USB JTAG/serial debug unit"). `ARDUINO_USB_CDC_ON_BOOT=1`,
  so `Serial` goes over USB CDC.
- **`esp32s3_uart`** — the UART bridge port (`/dev/ttyUSB0`, CP2102/CH343) if you use the
  other USB socket. `ARDUINO_USB_CDC_ON_BOOT=0`.

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

A six-stage self-test, run once at boot, then an idle loop with a 2 s status line:

1. **Identity** — chip, flash, PSRAM, MAC, SDK, build timestamp, reset reason
2. **NVS / flash write** — a boot counter in NVS; it must increase by 1 on every reset
3. **PSRAM stress** — `ps_malloc(4 MB)`, write a hash pattern, verify every word
4. **I2C scan** — `SDA=GPIO8 SCL=GPIO9`, lists any device at 0x08..0x77
5. **RGB LED** — cycles red/green/blue/yellow/off on `GPIO48`
6. **Serial echo** — anything you type comes back UPPERCASED (host → device direction)

Adjust `RGB_LED_PIN`, `I2C_SDA`, `I2C_SCL` at the top of the file to match your wiring.

## Verified result (2026-11-17)

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

## Notes

- The on-board LED pin is set to `LED_PIN 2`. DevKitC-1 boards use an addressable RGB LED on
  GPIO 38/48 depending on revision — if nothing blinks, the serial heartbeat is still the
  authoritative connectivity check.
- Toolchains and platform packages live in the shared `~/.platformio` core dir
  (platform-espressif32 **6.12.0**, Arduino core **2.0.17** / ESP-IDF 4.4.7). Kept as-is
  deliberately: it is the last release of the official `platformio/platform-espressif32`.
  Arduino core 3.x (ESP-IDF 5.x) only exists in the community `pioarduino` fork — switching
  would mean a ~500 MB toolchain download and some libraries not compiling, so it is not
  worth it unless a library demands the 3.x API.
