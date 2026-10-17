// led_light.ino  -  tunable white light for the coffee box
// Library: "Adafruit NeoPixel" (Arduino IDE > Library Manager)

#include <Adafruit_NeoPixel.h>
#include <Preferences.h>

constexpr uint8_t  LED_PIN   = 14;   // data pin of the LED ring
constexpr uint16_t LED_COUNT = 16;   // number of LEDs

// Factory values (used on first boot, or after the 'd' command)
constexpr uint8_t DEFAULT_R = 230;
constexpr uint8_t DEFAULT_G = 235;
constexpr uint8_t DEFAULT_B = 100;
constexpr uint8_t DEFAULT_L = 230;

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);
Preferences prefs;

uint8_t red = DEFAULT_R, green = DEFAULT_G, blue = DEFAULT_B, brightness = DEFAULT_L;

void apply() {
  strip.setBrightness(brightness);
  for (uint16_t i = 0; i < LED_COUNT; i++) {
    strip.setPixelColor(i, strip.Color(red, green, blue));
  }
  strip.show();
}

void save() {
  prefs.begin("light", false);
  prefs.putUChar("r", red);
  prefs.putUChar("g", green);
  prefs.putUChar("b", blue);
  prefs.putUChar("l", brightness);
  prefs.end();
}

void load() {
  prefs.begin("light", true);
  red        = prefs.getUChar("r", red);
  green      = prefs.getUChar("g", green);
  blue       = prefs.getUChar("b", blue);
  brightness = prefs.getUChar("l", brightness);
  prefs.end();
}

void resetToDefaults() {
  prefs.begin("light", false);
  prefs.clear();                       // forget saved values
  prefs.end();
  red = DEFAULT_R; green = DEFAULT_G; blue = DEFAULT_B; brightness = DEFAULT_L;
}

void printState() {
  Serial.printf("R=%u G=%u B=%u L=%u\n", red, green, blue, brightness);
}

void setup() {
  Serial.begin(115200);
  strip.begin();
  load();
  apply();
  Serial.println("Commands: c R G B | l 0-255 | s (save) | d (defaults) | p (print)");
  printState();
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  char cmd = line.charAt(0);
  if (cmd == 'c') {                       // c 230 235 100
    int r, g, b;
    if (sscanf(line.c_str(), "c %d %d %d", &r, &g, &b) == 3) {
      red = constrain(r, 0, 255);
      green = constrain(g, 0, 255);
      blue = constrain(b, 0, 255);
    }
  } else if (cmd == 'l') {                // l 200
    brightness = constrain(line.substring(1).toInt(), 0, 255);
  } else if (cmd == 's') {                // keep values after power off
    save();
    Serial.println("Saved");
  } else if (cmd == 'd') {                // back to factory values
    resetToDefaults();
    Serial.println("Defaults restored");
  }
  apply();
  printState();
}