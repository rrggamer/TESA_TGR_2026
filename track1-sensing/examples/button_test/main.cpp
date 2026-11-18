#include <Arduino.h>

const int BUTTON_PIN = 4;

void setup() {
  Serial.begin(9600);
  pinMode(BUTTON_PIN, INPUT_PULLUP);  // not pressed = HIGH, pressed = LOW
  Serial.println("Button test ready");
}

void loop() {
  static bool lastState = HIGH;
  static int count = 0;

  bool state = digitalRead(BUTTON_PIN);
  if (state != lastState) {
    delay(30);                          // wait out contact bounce
    state = digitalRead(BUTTON_PIN);
    if (state != lastState) {
      lastState = state;
      if (state == LOW) {
        Serial.printf("Pressed #%d\n", ++count);
      } else {
        Serial.println("Released");
      }
    }
  }
}
