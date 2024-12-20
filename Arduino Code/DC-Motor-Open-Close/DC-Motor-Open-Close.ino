// Basic script used for opening and closing hand with DC motors
// Implemented in 3rd Sprint review after we switched from Servo
// to DC motors

#include <Wire.h>
#include <Adafruit_MotorShield.h>

// Initialize motor shield and get DC motors from the shield
Adafruit_MotorShield AFMS1 = Adafruit_MotorShield(0x60);
Adafruit_DCMotor *indexMotor  = AFMS1.getMotor(1);
Adafruit_DCMotor *middleMotor = AFMS1.getMotor(2);
Adafruit_DCMotor *ringMotor   = AFMS1.getMotor(3);
Adafruit_DCMotor *pinkyMotor  = AFMS1.getMotor(4);

const int MOTOR_SPEED = 220;  // Speed up to 255
#define OPEN_DIR FORWARD
#define CLOSE_DIR BACKWARD

bool isClosed = false; // Track whether the hand is currently closed. Start with open = false.

// Set motor to open or closed
void setMotorState(Adafruit_DCMotor *motor, bool open) {
  motor->setSpeed(MOTOR_SPEED);
  motor->run(open ? OPEN_DIR : CLOSE_DIR);
}

// Stop all motors
void stopAllMotors() {
  indexMotor->setSpeed(0); indexMotor->run(RELEASE);
  middleMotor->setSpeed(0); middleMotor->run(RELEASE);
  ringMotor->setSpeed(0);  ringMotor->run(RELEASE);
  pinkyMotor->setSpeed(0); pinkyMotor->run(RELEASE);
}

// Open hand function
void openHand() {
  setMotorState(indexMotor, true);
  setMotorState(middleMotor, true);
  setMotorState(ringMotor, true);
  setMotorState(pinkyMotor, true);

  delay(500);
  stopAllMotors();
  isClosed = false;
}

// Close hand function 
void closeHand() {
  setMotorState(indexMotor, false);
  setMotorState(middleMotor, false);
  setMotorState(ringMotor, false);
  setMotorState(pinkyMotor, false);

  delay(500);
  stopAllMotors();
  isClosed = true;
}

void setup() {
  AFMS1.begin();
  Serial.begin(115200);
  
  // Start with the hand open
  openHand();
}

void loop() {
  // Read EMG value
  int emgValue = analogRead(A0);
  Serial.print("EMG Value: ");
  Serial.println(emgValue);

  // If emg value is above 200 AND the hand is open, close hand
  if (emgValue < 200 && !isClosed) {
    closeHand();
  }
  // If emg value is below 200 AND the hand is closed, open hand
  else if (emgValue >= 200 && isClosed) {
    openHand();
  }

  delay(100); 
}
