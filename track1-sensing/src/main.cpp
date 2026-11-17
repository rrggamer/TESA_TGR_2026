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
static const float CAL_SLOPE_PF_PER_US = 1.0f;
static const float CAL_OFFSET_PF       = 0.0f;

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
static const uint32_t TIMEOUT_US    = 500;   // give up on a single charge
static const uint32_t SETTLE_MS     = 200;   // let the hand move away first

static const uint32_t DEBOUNCE_MS = 30;
static const uint32_t LED_ON_MS   = 200;

static const uint32_t CHARGE_MASK = 1UL << CHARGE_PIN;
static const uint32_t COMP_MASK   = 1UL << COMP_PIN;

static portMUX_TYPE timingMux = portMUX_INITIALIZER_UNLOCKED;
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

// True exactly once per complete press-and-release.
static bool buttonReleased() {
  bool raw = digitalRead(BUTTON_PIN);
  if (raw != btnLastRaw) {             // still bouncing -> restart the timer
    btnLastRaw = raw;
    btnLastChange = millis();
  }
  if (millis() - btnLastChange >= DEBOUNCE_MS && raw != btnStable) {
    btnStable = raw;
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
    return 0;                                   // OUT already low: not discharged
  }

  const uint32_t timeoutTicks = TIMEOUT_US * ticksPerUs;
  uint32_t elapsed;

  portENTER_CRITICAL(&timingMux);              // no interrupts while timing
  const uint32_t start = ESP.getCycleCount();
  REG_WRITE(GPIO_OUT_W1TS_REG, CHARGE_MASK);
  do {
    elapsed = ESP.getCycleCount() - start;
  } while ((REG_READ(GPIO_IN_REG) & COMP_MASK) && elapsed < timeoutTicks);
  portEXIT_CRITICAL(&timingMux);

  REG_WRITE(GPIO_OUT_W1TC_REG, CHARGE_MASK);
  return elapsed < timeoutTicks ? elapsed : 0;
}

static Measurement measureCapacitance() {
  Measurement m = {};

  for (int i = 0; i < WARMUP_COUNT; i++) chargeOnce();

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
    Serial.println("FAIL: comparator never switched (check wiring / Vref)");
    return;
  }
  Serial.printf("t=%.4f us  sd=%.4f us  min=%.3f  max=%.3f  C=%.3f pF  (%lu ms)\r\n",
                m.meanUs, m.sdUs, m.minUs, m.maxUs, m.pF, (unsigned long)elapsedMs);
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
  ledFlash();
#else
  (void)elapsedMs;
  if (m.ok) {                                   // never print a made-up value
    printResult(m);
    ledFlash();
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
#endif
}

void loop() {
  bool trigger = buttonReleased();

#if CAL_MODE || defined(SERIAL_TRIGGER)
  if (Serial.available()) {
    while (Serial.available()) Serial.read();
    trigger = true;
  }
#endif

  if (trigger) {
    runMeasurement();
    buttonResync();
  }
}
