#define BOARD_HAS_PSRAM
#define CONFIG_SPIRAM_SUPPORT

#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiMulti.h>
#include <ESPmDNS.h>

WiFiMulti wifiMulti;
const char* espID = "esp_1"; 
unsigned long prethodnoVreme = 0;
const long interval = 5000; 

// Pinovi za XIAO ESP32S3 Sense
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     10
#define SIOD_GPIO_NUM     40
#define SIOC_GPIO_NUM     39
#define Y9_GPIO_NUM       48
#define Y8_GPIO_NUM       11
#define Y7_GPIO_NUM       12
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       16
#define Y4_GPIO_NUM       18
#define Y3_GPIO_NUM       17
#define Y2_GPIO_NUM       15
#define VSYNC_GPIO_NUM    38
#define HREF_GPIO_NUM     47
#define PCLK_GPIO_NUM     13

void slikajIPosalji();

void setup() {
  Serial.begin(115200);
  delay(2000); 
  Serial.println("\n=================================");
  Serial.println("=== KORAK 1: PROVERA MEMORIJE ===");
  Serial.println("=================================");
  
  if (psramInit()) {
    Serial.println("[USPEH] PSRAM čip je uspešno aktiviran!");
    Serial.printf("Ukupna veličina PSRAM-a: %d bajtova\n", ESP.getPsramSize());
  } else {
    Serial.println("[CRTIČNA GREŠKA] PSRAM nije inicijalizovan! Pogledaj Tools meni.");
  }

  Serial.println("\n=== KORAK 2: PALJENJE WI-FI VEZE ===");
  WiFi.mode(WIFI_STA);
  wifiMulti.addAP("ZTE_E42B39", "6ML5U2WH86");
  wifiMulti.addAP("RedmiV", "12345678"); 
  
  Serial.print("Povezujem se na Wi-Fi...");
  int pokusaji = 0;
  while (wifiMulti.run() != WL_CONNECTED && pokusaji < 60) {
    delay(500);
    Serial.print(".");
    pokusaji++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[GRESKA] Veza nije uspjela. Resetujem...");
    ESP.restart();
  } 
  
  Serial.println("\n[OK] Wi-Fi uspješno povezan!");
  Serial.print("IP adresa: "); Serial.println(WiFi.localIP());

  if (MDNS.begin("esp-klijent")) {
    Serial.println("[OK] mDNS spreman.");
  }

  Serial.println("\n=== KORAK 3: INICIJALIZACIJA KAMERE ===");
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  
  config.xclk_freq_hz = 2000000;
  config.frame_size = FRAMESIZE_SVGA; 
  
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 8;        
  config.fb_count = 2;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[GRESKA] Kamera pala: 0x%x\n", err);
  } else {
    Serial.println("[OK] Kamera uspešno pokrenuta u PSRAM-u!");
    delay(1000);
    slikajIPosalji(); 
  }
}

void loop() {
  unsigned long trenutnoVreme = millis();
  if (trenutnoVreme - prethodnoVreme >= interval) {
    prethodnoVreme = trenutnoVreme;
    if (wifiMulti.run() == WL_CONNECTED) {
      slikajIPosalji();
    }
  }
}

// NOVA FUNKCIJA KOJA ZAOBILAZI BAG UNUTAR HTTPCLIENT-A
void slikajIPosalji() {
  Serial.println("\n--- SLIKANJE ---");
  
  IPAddress rpiIP = MDNS.queryHost("rpi"); 
  if (rpiIP.toString() == "0.0.0.0") {
    rpiIP.fromString("10.103.28.54"); 
  }

  camera_fb_t * fb_stari = esp_camera_fb_get();
  if (fb_stari) esp_camera_fb_return(fb_stari);
  
  camera_fb_t * fb = esp_camera_fb_get();
  if(!fb) {
    Serial.println("[GRESKA] Nemoguće uslikati!");
    return;
  }

  WiFiClient client;
  // Otvaramo direktnu sirovu TCP vezu na port 5000 nase Maline
  if (!client.connect(rpiIP, 5000)) {
    Serial.println("[GRESKA] Povezivanje na Raspberry Pi server propalo!");
    esp_camera_fb_return(fb);
    return;
  }

  String boundary = "----ESP32Boundary";
  String head = "--" + boundary + "\r\nContent-Disposition: form-data; name=\"file\"; filename=\"strujomer_" + String(espID) + ".jpg\"\r\nContent-Type: image/jpeg\r\n\r\n";
  String tail = "\r\n--" + boundary + "--\r\n";
  size_t totalLen = head.length() + fb->len + tail.length();

  // Ručno pišemo HTTP POST zaglavlje direktno u soket
  client.println("POST /upload HTTP/1.1");
  client.print("Host: "); client.println(rpiIP.toString() + ":5000");
  client.println("User-Agent: ESP32S3-Xiao");
  client.println("Accept: */*");
  client.print("Content-Type: multipart/form-data; boundary="); client.println(boundary);
  client.print("Content-Length: "); client.println(totalLen);
  client.println("Connection: close");
  client.println(); // Prazan red označava kraj HTTP zaglavlja

  // Šaljemo podatke u delovima (izbegavamo zauzeće duple memorije)
  client.print(head);
  client.write(fb->buf, fb->len);
  client.print(tail);
  client.flush();

  Serial.println("Podaci poslati! Čekam odgovor servera...");
  
  // Čitamo samo prvi red odgovora da vidimo HTTP status
  while (client.connected() && !client.available()) delay(10);
  if (client.available()) {
    String responseLine = client.readStringUntil('\n');
    Serial.print("Odgovor sa Raspberry Pi-ja: "); Serial.println(responseLine);
  }

  client.stop();
  esp_camera_fb_return(fb);
}
