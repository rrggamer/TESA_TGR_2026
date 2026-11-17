/*
 * Track 1 - Sensing : upload + hardware self-test
 * Target: ESP32-S3-WROOM-1 N16R8 (16 MB flash / 8 MB octal PSRAM)
 *
 * Proves, in order:
 *   1. chip / flash / PSRAM identity
 *   2. NVS (flash write) survives a reboot  -> boot counter
 *   3. 4 MB PSRAM write + verify            -> OPI PSRAM really works
 *   4. I2C bus scan                         -> ready for real sensors
 *   5. RGB LED                              -> GPIO output
 *   6. serial echo                          -> host -> device direction
 */

#include <Arduino.h>
#include <Wire.h>
#include <Preferences.h>
#include <esp_chip_info.h>
#include <esp_flash.h>

// ESP32-S3-DevKitC-1 defaults; change to match your wiring.
#define RGB_LED_PIN 48
#define I2C_SDA      8
#define I2C_SCL      9

static Preferences prefs;
static uint32_t bootCount = 0;

static void hr() { Serial.println(F("--------------------------------------------")); }

// ---------- 1. identity ----------
static void testIdentity() {
  esp_chip_info_t chip;
  esp_chip_info(&chip);
  uint32_t flashSize = 0;
  esp_flash_get_size(nullptr, &flashSize);
  uint64_t mac = ESP.getEfuseMac();

  Serial.println(F("[1] IDENTITY"));
  Serial.printf("    chip      : %s rev%d, %d cores @ %lu MHz\n",
                ESP.getChipModel(), ESP.getChipRevision(), chip.cores,
                (unsigned long)getCpuFrequencyMhz());
  Serial.printf("    flash     : %.2f MB @ %lu MHz\n", flashSize / (1024.0 * 1024.0),
                (unsigned long)(ESP.getFlashChipSpeed() / 1000000UL));
  Serial.printf("    psram     : %.2f MB\n", ESP.getPsramSize() / (1024.0 * 1024.0));
  Serial.printf("    mac       : %02X:%02X:%02X:%02X:%02X:%02X\n",
                (uint8_t)(mac >> 0), (uint8_t)(mac >> 8), (uint8_t)(mac >> 16),
                (uint8_t)(mac >> 24), (uint8_t)(mac >> 32), (uint8_t)(mac >> 40));
  Serial.printf("    sdk       : %s / Arduino %d.%d.%d\n", ESP.getSdkVersion(),
                ESP_ARDUINO_VERSION_MAJOR, ESP_ARDUINO_VERSION_MINOR, ESP_ARDUINO_VERSION_PATCH);
  Serial.printf("    built     : %s %s\n", __DATE__, __TIME__);
  Serial.printf("    reset why : %d\n", (int)esp_reset_reason());
}

// ---------- 2. flash write survives reboot ----------
static void testNvs() {
  Serial.println(F("[2] NVS / FLASH WRITE"));
  if (!prefs.begin("selftest", false)) {
    Serial.println(F("    FAIL: cannot open NVS namespace"));
    return;
  }
  bootCount = prefs.getUInt("boots", 0) + 1;
  prefs.putUInt("boots", bootCount);
  uint32_t readBack = prefs.getUInt("boots", 0);
  prefs.end();

  Serial.printf("    boot count: %lu (written + read back %lu)  %s\n",
                (unsigned long)bootCount, (unsigned long)readBack,
                readBack == bootCount ? "[OK]" : "[FAIL]");
  Serial.println(F("    -> this number must grow by 1 on every reset"));
}

// ---------- 3. PSRAM write + verify ----------
static void testPsram() {
  Serial.println(F("[3] PSRAM STRESS"));
  if (ESP.getPsramSize() == 0) {
    Serial.println(F("    FAIL: no PSRAM (check memory_type = qio_opi)"));
    return;
  }

  const size_t blockSize = 4 * 1024 * 1024;  // 4 MB
  uint32_t *buf = (uint32_t *)ps_malloc(blockSize);
  if (!buf) {
    Serial.println(F("    FAIL: ps_malloc(4 MB) returned null"));
    return;
  }

  const size_t words = blockSize / sizeof(uint32_t);
  uint32_t t0 = millis();
  for (size_t i = 0; i < words; i++) buf[i] = (uint32_t)(i * 2654435761u);
  uint32_t tWrite = millis() - t0;

  t0 = millis();
  size_t errors = 0;
  for (size_t i = 0; i < words; i++) {
    if (buf[i] != (uint32_t)(i * 2654435761u)) errors++;
  }
  uint32_t tRead = millis() - t0;

  Serial.printf("    4 MB write: %lu ms (%.1f MB/s)\n", (unsigned long)tWrite,
                tWrite ? 4000.0 / tWrite : 0.0);
  Serial.printf("    4 MB verify: %lu ms (%.1f MB/s)\n", (unsigned long)tRead,
                tRead ? 4000.0 / tRead : 0.0);
  Serial.printf("    mismatches : %u  %s\n", (unsigned)errors, errors == 0 ? "[OK]" : "[FAIL]");

  free(buf);
  Serial.printf("    psram free after free(): %.2f MB\n", ESP.getFreePsram() / (1024.0 * 1024.0));
}

// ---------- 4. I2C scan ----------
static void testI2c() {
  Serial.println(F("[4] I2C SCAN"));
  Wire.begin(I2C_SDA, I2C_SCL, 100000);
  int found = 0;
  for (uint8_t addr = 0x08; addr < 0x78; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("    device @ 0x%02X\n", addr);
      found++;
    }
  }
  Serial.printf("    SDA=GPIO%d SCL=GPIO%d -> %d device(s) %s\n", I2C_SDA, I2C_SCL, found,
                found ? "[OK]" : "(bus idle - no sensor wired yet, this is fine)");
}

// ---------- 5. RGB LED ----------
static void testLed() {
  Serial.println(F("[5] RGB LED"));
  const uint8_t colors[][3] = {{64, 0, 0}, {0, 64, 0}, {0, 0, 64}, {48, 48, 0}, {0, 0, 0}};
  const char *names[] = {"red", "green", "blue", "yellow", "off"};
  for (size_t i = 0; i < 5; i++) {
    neopixelWrite(RGB_LED_PIN, colors[i][0], colors[i][1], colors[i][2]);
    Serial.printf("    GPIO%d -> %s\n", RGB_LED_PIN, names[i]);
    delay(250);
  }
  Serial.println(F("    -> if the on-board LED did not change, the pin differs on your board"));
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 3000) delay(10);
  delay(300);

  Serial.println();
  Serial.println(F("############################################"));
  Serial.println(F("  TESA TGR 2026 / Track 1 - SELF TEST"));
  Serial.println(F("############################################"));

  testIdentity();   hr();
  testNvs();        hr();
  testPsram();      hr();
  testI2c();        hr();
  testLed();        hr();

  Serial.println(F("[6] SERIAL ECHO - type anything, it comes back UPPERCASED"));
  Serial.println(F("    (proves the host -> device direction works too)"));
  hr();
  Serial.println(F("ALL TESTS DONE - entering idle loop"));
}

void loop() {
  // 6. echo whatever the host sends
  static String line;
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (line.length()) {
        line.toUpperCase();
        Serial.printf(">> ECHO: %s\n", line.c_str());
        line = "";
      }
    } else if (line.length() < 120) {
      line += c;
    }
  }

  // slow status pulse so the board visibly stays alive
  static uint32_t last = 0;
  static uint32_t beat = 0;
  if (millis() - last >= 2000) {
    last = millis();
    beat++;
    neopixelWrite(RGB_LED_PIN, 0, (beat % 2) ? 12 : 0, 0);
    Serial.printf("[boot #%lu | up %lus] heap=%lu psram=%lu temp=%.1fC\n",
                  (unsigned long)bootCount, (unsigned long)(millis() / 1000),
                  (unsigned long)ESP.getFreeHeap(), (unsigned long)ESP.getFreePsram(),
                  temperatureRead());
  }
}
