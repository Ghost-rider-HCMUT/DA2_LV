#include <Wire.h>
#include <Servo.h>
#include <SPI.h>
#include <MFRC522.h> // thu vien "RFID"
Servo servov;
Servo servor;
#define CBV A0
#define CBR A1
#define CB1 A2
#define CB2 A3
#define CB3 A4
#define COI A5
#define D1  4
#define D2  3
#define D3  2
#define SAI 0
#define DUNG  1
#define BAT HIGH
#define TAT LOW
#define SS_PIN_V  10
#define RST_PIN_V 8
#define SS_PIN_R  9
#define RST_PIN_R 7
#define minr  10
#define maxr  100
#define minv  10
#define maxv  100
int find_rfid = 0;
int RFID = 0;
MFRC522 mfrc522V(SS_PIN_V, RST_PIN_V);
MFRC522 mfrc522R(SS_PIN_R, RST_PIN_R);
unsigned long uidDec, uidDecTemp; // hien thi so UID dang thap phan
byte bCounter, readBit;
unsigned long ticketNumber;
int vr = 0, index = 0, doc_bien = 0;
int i = 0, tt = 0, tt2 = 0;
int lan = 0, dong_mo = 0, cp = 0, so_xe = 0, vtri = 0;
String kqv = "", kqr = "";
int mode = 0, dk_vao = 0, dk_ra = 0, lan2 = 0;
int tt_vao = 0, tt_ra = 0, dc_ra = 0;
String dataa = "";
char *token1, *token2, *token3, *token4;
char tama[30];
String mang_bien[10];
unsigned long mang_id[10];
unsigned long int time_vao = 0, time_ra = 0, time_gui = 0, time_cho = 0, time_docrf = 0;
//****************************************************************************************//
void setup() {
  Serial.begin(9600);
  pinMode(CBV, INPUT_PULLUP);
  pinMode(CBR, INPUT_PULLUP);
  pinMode(CB1, INPUT_PULLUP);
  pinMode(CB2, INPUT_PULLUP);
  pinMode(CB3, INPUT_PULLUP);
  pinMode(D1, OUTPUT);
  pinMode(D2, OUTPUT);
  pinMode(D3, OUTPUT);
  pinMode(COI, OUTPUT);
  digitalWrite(D1, TAT);
  digitalWrite(D2, TAT);
  digitalWrite(D3, TAT);
  digitalWrite(COI, TAT);
  servov.attach(6);
  servor.attach(5);
  servov.write(minv);
  servor.write(minr);
  SPI.begin();
  delay(1000);
}
//****************************************************************************************//
void loop() {
  //  Serial.println(digitalRead(CB1));
  if (millis() - time_docrf <= 500) {
    doc_ra();
  }
  else if (millis() - time_docrf <= 1000) {
    doc_vao();
  }
  else {
    time_docrf = millis();
  }
  if (mode == 0) {
    Auto();
  }
  else {
    Man();
  }
  vr = 0;
  if (digitalRead(CB1) == 0) {
    digitalWrite(D1, BAT);
  }
  else {
    digitalWrite(D1, TAT);
  }

  if (digitalRead(CB2) == 0) {
    digitalWrite(D2, BAT);
  }
  else {
    digitalWrite(D2, TAT);
  }

  if (digitalRead(CB3) == 0) {
    digitalWrite(D3, BAT);
  }
  else {
    digitalWrite(D3, TAT);
  }
  guidulieu();
}
//****************************************************************************************//
void Auto() {
  if (digitalRead(CBV) == 0) {
    if (vr == 1) {
      cp = 1;
      guidulieu2();
      cp = 0;
      if (so_xe < 3) {
        mang_id[index] = uidDec;
        //      Serial.print("RFID = ");
        //      Serial.println(mang_id[index]);
        while (doc_bien == 0) {
          serialEvent();
          guidulieu();
        }
        mang_bien[index] = kqv;
        //      Serial.print("bien = ");
        //      Serial.println(mang_bien[index]);
        doc_bien = 0;
        vr = 0;
        index++;
        if (index >= 3)  index = 0;
        mov();
        while (digitalRead(CBV) == 0) {
          serialEvent();
          guidulieu();
        }
        so_xe++;
        guidulieu2();
        delay(500);
        dongv();
      }
    }
  }
  if (digitalRead(CBR) == 0) {
    if (vr == 2) {
      cp = 1;
      guidulieu2();
      cp = 0;
      while (doc_bien == 0) {
        serialEvent();
        guidulieu();
      }
      doc_bien = 0;
      vr = 0;
      for (int i = 0; i <= 2; i++) {
        if (uidDec == mang_id[i])  vtri = i;
      }
      if (kqr == mang_bien[vtri]) {
        mor();
        so_xe--;
        dc_ra = 1;
        guidulieu2();
        dc_ra = 0;
        while (digitalRead(CBR) == 0) {
          serialEvent();
          guidulieu();
        }
        delay(500);
        dongr();
      }
      else{
        dc_ra = 2;
        guidulieu2();
        dc_ra = 0;
      }
    }
  }
}
//****************************************************************************************//

//****************************************************************************************//
void Man() {
  if (dk_vao == 0) {
    dongv();
  }
  else            {
    mov();
  }
  if (dk_ra == 0)  dongr();
  else            mor();
}
//****************************************************************************************//
void serialEvent() {
  while (Serial.available()) {
    dataa = Serial.readStringUntil('\n');
    dataa.toCharArray(tama, 30);
    token1 = strtok(tama, "|");
    token2 = strtok(NULL, "|");

    String st1(token1);
    String st2(token2);

    if (st1 == "a")  mode = st2.toInt();
    if (st1 == "b")  dk_vao = st2.toInt();
    if (st1 == "c")  dk_ra = st2.toInt();
    if (st1 == "d") {
      if (vr == 1) kqv = st2;
      if (vr == 2) kqr = st2;
      doc_bien = 1;
    }
  }
}
//****************************************************************************************//
void mov() {
  if (tt == 0) {
    for (i = minv; i <= maxv; i++) {
      servov.write(i);
      delay(3);
    }
  }
  tt_vao = 1;
  tt = 1;
}
//---------------------------------------------------------------
void dongv() {
  if (tt == 1) {
    for (i = maxv; i >= minv; i--) {
      while (digitalRead(CBV) == 0) {
      }
      servov.write(i);
      delay(3);
    }
    tt_vao = 0;
    tt = 0;
  }
}
//---------------------------------------------------------------
//****************************************************************************************//
void mor() {
  if (tt2 == 0) {
    for (i = minr; i <= maxr; i++) {
      servor.write(i);
      delay(3);
    }
  }
  tt_ra = 1;
  tt2 = 1;
}
//---------------------------------------------------------------
void dongr() {
  if (tt2 == 1) {
    for (i = maxr; i >= minr; i--) {
      while (digitalRead(CBR) == 0) {
      }
      servor.write(i);
      delay(3);
    }
    tt_ra = 0;
    tt2 = 0;
  }
}
//****************************************************************************************//
void guidulieu() {
  if (millis() - time_gui >= 200) {
    Serial.print(mode);
    Serial.print("|");
    Serial.print(tt_vao);
    Serial.print("|");
    Serial.print(tt_ra);
    Serial.print("|");
    Serial.print(cp);
    Serial.print("|");
    Serial.print(vr);
    Serial.print("|");
    Serial.print(dc_ra);
    Serial.print("|");
    Serial.println(so_xe);
    time_gui = millis();
  }
}
//****************************************************************************************//
void guidulieu2() {
  Serial.print(mode);
  Serial.print("|");
  Serial.print(tt_vao);
  Serial.print("|");
  Serial.print(tt_ra);
  Serial.print("|");
  Serial.print(cp);
  Serial.print("|");
  Serial.print(vr);
  Serial.print("|");
  Serial.print(dc_ra);
  Serial.print("|");
  Serial.println(so_xe);
  time_gui = millis();
}
//****************************************************************************************//
void doc_vao() {
  mfrc522V.PCD_Init();
  // Tim the moi
  if ( ! mfrc522V.PICC_IsNewCardPresent()) {
    //return;
    find_rfid = 1;
    lan = 0;
  }
  else  find_rfid = 0;
  // Doc the
  if ( ! mfrc522V.PICC_ReadCardSerial()) {
    //return;
    find_rfid = 1;
    lan = 0;
  }
  else  find_rfid = 0;

  if (find_rfid == 0) {
    vr = 1;
    uidDec = 0;
    // Hien thi so UID cua the
    //    Serial.println("Serijnyj nomer karty / Card UID: ");
    for (byte i = 0; i < mfrc522V.uid.size; i++) {
      uidDecTemp = mfrc522V.uid.uidByte[i];
      uidDec = uidDec * 256 + uidDecTemp;
    }
    //    Serial.println("-----The Vao------");
    //    Serial.print("UID:");
    //    Serial.println(uidDec);
    digitalWrite(COI, HIGH);
    delay(200);
    digitalWrite(COI, LOW);
    delay(200);
    digitalWrite(COI, HIGH);
    delay(200);
    digitalWrite(COI, LOW);
    if (uidDec == 1127969705 || uidDec == 1664753574) {

    }
  }
}
//****************************************************************************************//
void doc_ra() {
  mfrc522R.PCD_Init();
  // Tim the moi
  if ( ! mfrc522R.PICC_IsNewCardPresent()) {
    //return;
    find_rfid = 1;
    lan = 0;
  }
  else  find_rfid = 0;
  // Doc the
  if ( ! mfrc522R.PICC_ReadCardSerial()) {
    //return;
    find_rfid = 1;
    lan = 0;
  }
  else  find_rfid = 0;

  if (find_rfid == 0) {
    vr = 2;
    uidDec = 0;
    // Hien thi so UID cua the
    //    Serial.println("Serijnyj nomer karty / Card UID: ");
    for (byte i = 0; i < mfrc522R.uid.size; i++) {
      uidDecTemp = mfrc522R.uid.uidByte[i];
      uidDec = uidDec * 256 + uidDecTemp;
    }
    //    Serial.println("-----The Ra------");
    //    Serial.print("UID:");
    //    Serial.println(uidDec);
    digitalWrite(COI, HIGH);
    delay(200);
    digitalWrite(COI, LOW);
    delay(200);
    digitalWrite(COI, HIGH);
    delay(200);
    digitalWrite(COI, LOW);
    if (uidDec == 1127969705 || uidDec == 1664753574) {

    }
  }
}
