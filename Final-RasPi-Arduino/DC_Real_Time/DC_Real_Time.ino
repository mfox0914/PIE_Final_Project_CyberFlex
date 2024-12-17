#include <Wire.h>
#include <Adafruit_MotorShield.h>

Adafruit_MotorShield AFMS1 = Adafruit_MotorShield(0x60);
Adafruit_MotorShield AFMS2 = Adafruit_MotorShield(0x61);
Adafruit_MotorShield AFMS3 = Adafruit_MotorShield(0x63);

Adafruit_DCMotor *indexMotor = AFMS1.getMotor(1);
Adafruit_DCMotor *middleMotor = AFMS1.getMotor(2);
Adafruit_DCMotor *ringMotor = AFMS1.getMotor(3);
Adafruit_DCMotor *pinkyMotor = AFMS1.getMotor(4);

// Figure out thumb motors on second shield later
Adafruit_DCMotor *thumbMotor1 = AFMS2.getMotor(1);
Adafruit_DCMotor *thumbMotor2 = AFMS2.getMotor(2);

//// Wrist motors
//Adafruit_DCMotor *WristMotorLeft = AFMS2.getMotor(3);
//Adafruit_DCMotor *WristMotorRight = AFMS2.getMotor(4);
//Adafruit_DCMotor *WristMotorRotate = AFMS3.getMotor(1);

//const int emgPin1 = A0;
//const int emgPin2 = A1;
//const int emgPin3 = A2;
//const int emgPin4 = A3;
//const int emgPin5 = A4;

const int MOTOR_SPEED = 200;
#define OPEN_DIR BACKWARD
#define CLOSE_DIR FORWARD

unsigned long sampleRate = 100;
unsigned long previousMillis = 0;

void setup() {
  Serial.begin(115200);
  AFMS1.begin();
  AFMS2.begin();
  AFMS3.begin();
  stopAllMotors();
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= (1000 / sampleRate)) {
    previousMillis = currentMillis;
    
    // Read EMG values
    int emgValue1 = analogRead(A0);
    int emgValue2 = analogRead(A1);
    int emgValue3 = analogRead(A2);
    int emgValue4 = analogRead(A3);
    int emgValue5 = analogRead(A4);
  
    Serial.print(emgValue1); Serial.print(",");
    Serial.print(emgValue2); Serial.print(",");
    Serial.print(emgValue3); Serial.print(",");
    Serial.print(emgValue4); Serial.print(",");
    Serial.println(emgValue5);
  }
  
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim(); 

    if (input.length() > 0) {
      int predictedGesture = input.toInt();
      performGesture(predictedGesture);
    }
  }
}

void performGesture(int gesture) {
  switch (gesture) {
    case 0: showRock(); break;
    case 1: showPaper(); break;
    case 2: showScissors(); break;
    case 3: showPhone(); break;
    case 4: showPoint(); break;
    default: resetPositions(); break;
  }
}

void setMotorState(Adafruit_DCMotor *motor, bool open) {
  if (open) {
    motor->setSpeed(MOTOR_SPEED);
    motor->run(OPEN_DIR);
  } else {
    motor->setSpeed(MOTOR_SPEED);
    motor->run(CLOSE_DIR);
  }
}

void stopAllMotors() {
  indexMotor->setSpeed(0);
  indexMotor->run(RELEASE);

  middleMotor->setSpeed(0);
  middleMotor->run(RELEASE);

  ringMotor->setSpeed(0);
  ringMotor->run(RELEASE);

  pinkyMotor->setSpeed(0);
  pinkyMotor->run(RELEASE);

  thumbMotor1->setSpeed(0);
  thumbMotor1->run(RELEASE);

  thumbMotor2->setSpeed(0);
  thumbMotor2->run(RELEASE);
}


// Code to to show each gesture
void showRock() {
  setMotorState(thumbMotor1, false);
  setMotorState(thumbMotor2, false);
  setMotorState(indexMotor, false);
  setMotorState(middleMotor, false);
  setMotorState(ringMotor, false);
  setMotorState(pinkyMotor, false);
  delay(300); stopAllMotors();
}

void showPaper() {
  setMotorState(thumbMotor1, true);
  setMotorState(thumbMotor2, true);
  setMotorState(indexMotor, true);
  setMotorState(middleMotor, true);
  setMotorState(ringMotor, true);
  setMotorState(pinkyMotor, true);
  delay(300); stopAllMotors();
}

void showScissors() {
  setMotorState(thumbMotor1, false);
  setMotorState(thumbMotor2, false);
  setMotorState(indexMotor, true);
  setMotorState(middleMotor, true);
  setMotorState(ringMotor, false);
  setMotorState(pinkyMotor, false);
  delay(300); stopAllMotors();
}

void showPhone() {
  setMotorState(thumbMotor1, true);
  setMotorState(thumbMotor2, true);
  setMotorState(indexMotor, false);
  setMotorState(middleMotor, false);
  setMotorState(ringMotor, false);
  setMotorState(pinkyMotor, true);
  delay(300); stopAllMotors();
}

void showPoint() {
  setMotorState(thumbMotor1, false);
  setMotorState(thumbMotor2, false);
  setMotorState(indexMotor, true);
  setMotorState(middleMotor, false);
  setMotorState(ringMotor, false);
  setMotorState(pinkyMotor, false);
  delay(300); stopAllMotors();
}

void resetPositions() {
  // Default to all open and then stop
  showPaper(); 
  stopAllMotors();
}
