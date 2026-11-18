#include <Arduino.h>
#include <algorithm>
#include <math.h>
#include <soc/gpio_reg.h>

// ============================================================
//  CONFIG - edit these
// ============================================================

// Team code exactly as it must appear in the output (rule 3.5).
#define TEAM_CODE "TGR-20"

// What the competition firmware reports:
//   0 = capacitance contest (18 Nov)   -> Capacitance = xx.xx pF
//   1 = moisture contest    (20 Nov)   -> Moisture Level = xx.xx %
#ifndef OUTPUT_MOISTURE
#define OUTPUT_MOISTURE 0
#endif

#ifndef CAL_MODE
#define CAL_MODE 0
#endif

// Calibration fit:  C[pF] = CAL_SLOPE * t[us] + CAL_OFFSET
// With R1 = 1 MOhm and Vref = 0.632 * Vcc, theory gives 1 us == 1 pF.
// Replace with the line fitted from the 5/10/15/20 pF reference caps;
// CAL_OFFSET absorbs stray capacitance and comparator delay.
// Interim one-point fit (2026-11-17): slope from theory with Vref = 2.088 V,
// Vcc = 3.3 V; offset from the 20 pF cap (mean t = 33.3695 us over 6 runs).
// Replace with the 5/10/15/20 pF multi-point fit.
static const float CAL_SLOPE_PF_PER_US = 0.99835f;
static const float CAL_OFFSET_PF       = -13.314f;

// Moisture fit:  M[%] = MOIST_SLOPE * C[pF] + MOIST_OFFSET  (TODO: calibrate)
static const float MOIST_SLOPE_PCT_PER_PF = 1.0f;
static const float MOIST_OFFSET_PCT       = 0.0f;

// ============================================================
//  HARDWARE
// ============================================================

static const uint8_t BUTTON_PIN = 4;
static const uint8_t CHARGE_PIN = 5;
static const uint8_t COMP_PIN   = 6;
static const uint8_t LED_PIN    = 48;  // on-board addressable RGB LED

static const uint32_t SERIAL_BAUD = 9600;

// ============================================================
//  MEASUREMENT TUNING
// ============================================================

static const int      SAMPLE_COUNT  = 500;   // charge cycles averaged per result
static const int      WARMUP_COUNT  = 10;    // discarded before sampling
static const float    TRIM_FRACTION = 0.10f; // drop this share from each end
static const uint32_t DISCHARGE_US  = 1000;  // >= 9 tau at 100 pF total
static const uint32_t DITHER_TICKS  = 240;   // polling-phase sweep, 1 us at 240 MHz
static const uint32_t TIMEOUT_US    = 500;   // give up on a single charge
static const uint32_t SETTLE_MS     = 200;   // let the hand move away first

static const uint32_t DEBOUNCE_MS = 30;
static const uint32_t LED_ON_MS   = 200;

static const uint32_t CHARGE_MASK = 1UL << CHARGE_PIN;
static const uint32_t COMP_MASK   = 1UL << COMP_PIN;

static portMUX_TYPE timingMux = portMUX_INITIALIZER_UNLOCKED;
static int          failNotReady = 0;   // COMP already low before charging
static int          failTimeout  = 0;   // COMP never went low while charging
static uint32_t     ticksPerUs = 240;
static uint32_t     samples[SAMPLE_COUNT];

struct Measurement {
  bool  ok;
  int   valid;     // charge cycles that crossed Vref in time
  float meanUs;    // trimmed mean charge time
  float sdUs;      // standard deviation of the trimmed set
  float minUs;
  float maxUs;
  float pF;
};

// ============================================================
//  BUTTON (software debounce, fires on release)
// ============================================================

static bool     btnStable     = HIGH;  // HIGH = not pressed (INPUT_PULLUP)
static bool     btnLastRaw    = HIGH;
static uint32_t btnLastChange = 0;

#if CAL_MODE
static bool     btnTestOnly   = true;  // 'b' toggles: button logs only / button measures
static uint32_t btnRawEdges   = 0;     // raw level changes seen, bounces included
static uint32_t btnPressedAt  = 0;
static uint32_t btnCount      = 0;
#endif

static void ledFlash();

// True exactly once per complete press-and-release.
static bool buttonReleased() {
  bool raw = digitalRead(BUTTON_PIN);
  if (raw != btnLastRaw) {             // still bouncing -> restart the timer
    btnLastRaw = raw;
    btnLastChange = millis();
    if (raw == LOW && btnStable == HIGH) ledFlash();  // rule 3.4: LED on at the press
#if CAL_MODE
    btnRawEdges++;
#endif
  }
  if (millis() - btnLastChange >= DEBOUNCE_MS && raw != btnStable) {
    btnStable = raw;
#if CAL_MODE
    if (btnStable == LOW) {
      btnPressedAt = btnLastChange;
      Serial.printf("BTN  DOWN  (raw edges so far: %lu)\r\n", (unsigned long)btnRawEdges);
    } else {
      btnCount++;
      Serial.printf("BTN  UP    #%lu  held %lu ms  raw edges %lu (2 = clean, more = bounce filtered)\r\n",
                    (unsigned long)btnCount, (unsigned long)(btnLastChange - btnPressedAt),
                    (unsigned long)btnRawEdges);
      btnRawEdges = 0;
      if (btnTestOnly) return false;   // log only, no measurement
    }
#endif
    return btnStable == HIGH;          // only the release counts
  }
  return false;
}

// Forget anything pressed while a measurement was running.
static void buttonResync() {
  btnStable = btnLastRaw = digitalRead(BUTTON_PIN);
  btnLastChange = millis();
}

// ============================================================
//  CAPACITANCE MEASUREMENT
// ============================================================

// One charge cycle. Returns ticks from CHARGE high to COMP low, 0 on failure.
static uint32_t IRAM_ATTR chargeOnce() {
  REG_WRITE(GPIO_OUT_W1TC_REG, CHARGE_MASK);   // discharge Cx through R1
  delayMicroseconds(DISCHARGE_US);

#ifdef FAKE_TICKS
  return FAKE_TICKS;                            // bench test without a circuit
#endif

  if (!(REG_READ(GPIO_IN_REG) & COMP_MASK)) {
    failNotReady++;
    return 0;                                   // OUT already low: not discharged
  }

  const uint32_t timeoutTicks = TIMEOUT_US * ticksPerUs;
  uint32_t elapsed;

  // Each GPIO read takes ~30 CPU cycles, so the edge is only seen on a coarse
  // grid. Shifting where polling starts by 0..DITHER_TICKS-1 cycles spreads the
  // edge across that grid, letting the average resolve below one grid step.
  static uint32_t dither = 0;
  dither = (dither + 1) % DITHER_TICKS;

  portENTER_CRITICAL(&timingMux);              // no interrupts while timing
  const uint32_t start = ESP.getCycleCount();
  REG_WRITE(GPIO_OUT_W1TS_REG, CHARGE_MASK);
  while (ESP.getCycleCount() - start < dither) {
  }
  do {
    elapsed = ESP.getCycleCount() - start;
  } while ((REG_READ(GPIO_IN_REG) & COMP_MASK) && elapsed < timeoutTicks);
  portEXIT_CRITICAL(&timingMux);

  REG_WRITE(GPIO_OUT_W1TC_REG, CHARGE_MASK);
  if (elapsed >= timeoutTicks) {
    failTimeout++;
    return 0;
  }
  return elapsed;
}

static Measurement measureCapacitance() {
  Measurement m = {};

  for (int i = 0; i < WARMUP_COUNT; i++) chargeOnce();
  failNotReady = failTimeout = 0;

  int n = 0;
  for (int i = 0; i < SAMPLE_COUNT; i++) {
    uint32_t t = chargeOnce();
    if (t) samples[n++] = t;
  }
  m.valid = n;
  if (n < SAMPLE_COUNT / 2) return m;          // circuit not responding

  std::sort(samples, samples + n);
  const int drop = (int)(n * TRIM_FRACTION);
  const int lo = drop, hi = n - drop;

  double sum = 0;
  for (int i = lo; i < hi; i++) sum += samples[i];
  const double mean = sum / (hi - lo);

  double var = 0;
  for (int i = lo; i < hi; i++) {
    const double d = samples[i] - mean;
    var += d * d;
  }

  m.meanUs = mean / ticksPerUs;
  m.sdUs   = sqrt(var / (hi - lo)) / ticksPerUs;
  m.minUs  = (float)samples[0] / ticksPerUs;
  m.maxUs  = (float)samples[n - 1] / ticksPerUs;
  m.pF     = CAL_SLOPE_PF_PER_US * m.meanUs + CAL_OFFSET_PF;
  m.ok     = true;
  return m;
}

static float moisturePercent(float pF) {
  return MOIST_SLOPE_PCT_PER_PF * pF + MOIST_OFFSET_PCT;
}

// ============================================================
//  OUTPUT
// ============================================================

static void ledFlash() {
  neopixelWrite(LED_PIN, 64, 64, 64);
  delay(LED_ON_MS);
  neopixelWrite(LED_PIN, 0, 0, 0);
}

// Exact competition format (rule 3.5).
static void printResult(const Measurement &m) {
  Serial.println("\"" TEAM_CODE "\"");
#if OUTPUT_MOISTURE
  Serial.printf("Moisture Level = %.2f %%\r\n", moisturePercent(m.pF));
#else
  Serial.printf("Capacitance = %.2f pF\r\n", m.pF);
#endif
  Serial.println();
  Serial.flush();
}

#if CAL_MODE
static void printCalibration(const Measurement &m, uint32_t elapsedMs) {
  Serial.printf("valid=%d/%d  ", m.valid, SAMPLE_COUNT);
  if (!m.ok) {
    Serial.printf("FAIL: not-ready=%d timeout=%d\r\n", failNotReady, failTimeout);
    return;
  }
  Serial.printf("t=%.4f us  sd=%.4f us  min=%.3f  max=%.3f  C=%.3f pF  (%lu ms)\r\n",
                m.meanUs, m.sdUs, m.minUs, m.maxUs, m.pF, (unsigned long)elapsedMs);
}

// Static wiring check: hold CHARGE low, then high, and read COMP each time.
// Expected: COMP=1 while discharged (Vc < Vref), COMP=0 once charged (Vc > Vref).
static void printDiagnostics() {
  digitalWrite(CHARGE_PIN, LOW);
  delay(20);
  const int compLow = digitalRead(COMP_PIN);
  digitalWrite(CHARGE_PIN, HIGH);
  delay(20);
  const int compHigh = digitalRead(COMP_PIN);
  digitalWrite(CHARGE_PIN, LOW);
  Serial.printf("DIAG  CHARGE=LOW -> COMP=%d (expect 1)   CHARGE=HIGH -> COMP=%d (expect 0)\r\n",
                compLow, compHigh);

  // With the internal pull-up swapped for a pull-down, COMP only reads 1 if the
  // external 10k pull-up (or a driven LM393 output) is really on this pin.
  pinMode(COMP_PIN, INPUT_PULLDOWN);
  delay(5);
  const int compPulldown = digitalRead(COMP_PIN);
  pinMode(COMP_PIN, INPUT_PULLUP);
  Serial.printf("DIAG  COMP with internal pull-down = %d (expect 1 = external 10k present)\r\n",
                compPulldown);
}

// Hold CHARGE at a fixed level for probing with a multimeter.
static void holdCharge(bool high) {
  Serial.printf("HOLD  CHARGE (GPIO%d) = %s for 15 s - probe now\r\n", CHARGE_PIN,
                high ? "HIGH" : "LOW");
  digitalWrite(CHARGE_PIN, high ? HIGH : LOW);
  for (int i = 0; i < 15; i++) {
    delay(1000);
    Serial.printf("      %2d s  COMP=%d\r\n", i + 1, digitalRead(COMP_PIN));
  }
  digitalWrite(CHARGE_PIN, LOW);
  Serial.println("HOLD  done, CHARGE back to LOW");
}
#endif

// ============================================================
//  MAIN
// ============================================================

static void runMeasurement() {
  delay(SETTLE_MS);
  const uint32_t t0 = millis();
  Measurement m = measureCapacitance();
  const uint32_t elapsedMs = millis() - t0;

#if CAL_MODE
  printCalibration(m, elapsedMs);
#else
  (void)elapsedMs;
  if (m.ok) {                                   // never print a made-up value
    printResult(m);
  }
#endif
}

void setup() {
  Serial.begin(SERIAL_BAUD);

  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(COMP_PIN, INPUT_PULLUP);              // in parallel with the 10k
  pinMode(CHARGE_PIN, OUTPUT);
  digitalWrite(CHARGE_PIN, LOW);
  neopixelWrite(LED_PIN, 0, 0, 0);

  ticksPerUs = getCpuFrequencyMhz();
  buttonResync();

#if CAL_MODE
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) delay(10);
  Serial.println();
  Serial.println("=== CALIBRATION MODE ===");
  Serial.printf("team=%s  cpu=%lu MHz  samples=%d  trim=%.0f%%\r\n", TEAM_CODE,
                (unsigned long)ticksPerUs, SAMPLE_COUNT, TRIM_FRACTION * 100);
  Serial.printf("fit: C = %.5f * t + %.5f\r\n", CAL_SLOPE_PF_PER_US, CAL_OFFSET_PF);
  Serial.println("press the button or send any key to measure");
  Serial.println("  d = wiring check   h = hold CHARGE high 15 s   l = hold CHARGE low 15 s");
  Serial.println("  b = toggle button test-only mode (starts ON: button just logs)");
  Serial.printf("BTN  GPIO%d idle level = %d (expect 1)\r\n", BUTTON_PIN, digitalRead(BUTTON_PIN));
#endif
}

void loop() {
  bool trigger = buttonReleased();

#if CAL_MODE || defined(SERIAL_TRIGGER)
  if (Serial.available()) {
    char cmd = 0;
    while (Serial.available()) cmd = (char)Serial.read();
#if CAL_MODE
    if (cmd == 'd') { printDiagnostics(); return; }
    if (cmd == 'h') { holdCharge(true);   return; }
    if (cmd == 'l') { holdCharge(false);  return; }
    if (cmd == 'b') {
      btnTestOnly = !btnTestOnly;
      Serial.printf("BTN  mode: %s\r\n", btnTestOnly ? "test only (no measurement)"
                                                     : "release triggers measurement");
      return;
    }
#endif
    (void)cmd;
    trigger = true;
  }
#endif

  if (trigger) {
    runMeasurement();
    buttonResync();
  }
}
