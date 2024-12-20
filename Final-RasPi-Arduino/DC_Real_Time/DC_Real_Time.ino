#include <Wire.h>
#include <Adafruit_MotorShield.h>

// Initialize Motorshields
Adafruit_MotorShield AFMS1 = Adafruit_MotorShield(0x60);
Adafruit_MotorShield AFMS2 = Adafruit_MotorShield(0x61);
Adafruit_MotorShield AFMS3 = Adafruit_MotorShield(0x62);

// Index, Middle, Ring, and Pinky Fingers on first Shield
Adafruit_DCMotor *indexMotor = AFMS1.getMotor(1);
Adafruit_DCMotor *middleMotor = AFMS1.getMotor(2);
Adafruit_DCMotor *ringMotor = AFMS1.getMotor(3);
Adafruit_DCMotor *pinkyMotor = AFMS1.getMotor(4);

// Inner and Outer Thumb Motors
Adafruit_DCMotor *thumbMotor1 = AFMS2.getMotor(1);
Adafruit_DCMotor *thumbMotor2 = AFMS2.getMotor(2);

// Wrist motors
Adafruit_DCMotor *WristMotorLeft = AFMS3.getMotor(1);
Adafruit_DCMotor *WristMotorRight = AFMS3.getMotor(2);

// Account for last wrist rotation (unfortunately could not end up implementing)
// Adafruit_DCMotor *WristMotorRotate = AFMS3.getMotor(1);

//const int emgPin1 = A0;
//const int emgPin2 = A1;
//const int emgPin3 = A2;
//const int emgPin4 = A3;
//const int emgPin5 = A4;

// Set motor speed
const int MOTOR_SPEED = 200;

// Define the open and close direction of the fingers
#define OPEN_DIR FORWARD
#define CLOSE_DIR BACKWARD

// Define sample rate of signals (In this case 100hz)
unsigned long sampleRate = 100;
unsigned long previousMillis = 0;

//// Make it only work every 2 seconds to prevent overloading
//unsigned long lastGestureTime = 0;
//const unsigned long gestureInterval = 2000; 

// Store previous gesture to prevent it from being done twice in a row
String lastGesture = "";

// Make sure you can't take an input while a gesture is in progress
bool gestureInProgress = false;  

void setup() {
  Serial.begin(115200);
  AFMS1.begin();
  AFMS2.begin();
  AFMS3.begin();
  stopAllMotors();

//  // Test to see if you can find motorshields
//  if (!AFMS1.begin(100)) {         // create with the default frequency 1.6KHz
//  // if (!AFMS.begin(100)) {  // OR with a different frequency, say 1KHz
//    Serial.println("Could not find Motor Shield. Check wiring.");
//    while (1);
//  }
//  Serial.println("Motor Shield 1 found.");
//  
//  if (!AFMS2.begin(100)) {         // create with the default frequency 1.6KHz
//  // if (!AFMS.begin(100)) {  // OR with a different frequency, say 1KHz
//    Serial.println("Could not find Motor Shield. Check wiring.");
//    while (1);
//  }
//  Serial.println("Motor Shield 2 found.");
//  
//  if (!AFMS3.begin(100)) {         // create with the default frequency 1.6KHz
//  // if (!AFMS.begin(100)) {  // OR with a different frequency, say 1KHz
//    Serial.println("Could not find Motor Shield. Check wiring.");
//    while (1);
//  }
//  Serial.println("Motor Shield 3 found.");
}

void loop() {

//  Test if motors work by hard coding them
//  setMotorState(thumbMotor1, true);
//  setMotorState(thumbMotor2, true);
//  setMotorState(indexMotor, true);
//  setMotorState(middleMotor, true);
//  setMotorState(ringMotor, true);
//  setMotorState(pinkyMotor, true);
//  delay(100); stopAllMotors();

  // Set up millis for accurate time reading
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= (1000 / sampleRate)) {
    previousMillis = currentMillis;
    
    // Read EMG values
    // Note: Originally used 5 emg values but for some undetermined reason, attaching
    // more than 4 prevented all of our code from working, so we switched back to
    // using only 4 sensors 

    int emgValue1 = analogRead(A0);
    int emgValue2 = analogRead(A1);
    int emgValue3 = analogRead(A2);
    int emgValue4 = analogRead(A3);
    // int emgValue5 = analogRead(A4);
  
    Serial.print(emgValue1); Serial.print(",");
    Serial.print(emgValue2); Serial.print(",");
    Serial.print(emgValue3); Serial.print(",");
    Serial.println(emgValue4); // Serial.print(",");
    // Serial.println(emgValue5);
  }

  // Recieve data from python script 
  // Only run if gesture is not in progress and serial is available
  if (!gestureInProgress && Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim(); 
    
    Serial.print(input);

    // Perform gesture based on what the python script says
    if (input.length() > 0 && input != lastGesture) {
      performGesture(input);
      lastGesture = input;

      // Check if enough time has passed since last gesture
//      unsigned long currentTime = millis();
//      if (currentTime - lastGestureTime >= gestureInterval) {
//        performGesture(input);
//        lastGestureTime = currentTime; // Update the last gesture time
//      } else {
//        // Not enough time has passed, ignore or you could print a message
//        Serial.println("Gesture ignored, too soon.");
//      }
  
    }
  }
}

// Perform gesture on hand based on input from python script
void performGesture(String gestureName) {
  if (gestureName == "Rock") {
    showRock();
  } else if (gestureName == "Paper") {
    showRelaxed();
  } else if (gestureName == "Scissors") {
    showScissors();
  } else if (gestureName == "Phone") {
    showPhone();
  } else if (gestureName == "Thumb") {
    showThumb();
  } else if (gestureName == "Middle") {
    showMiddle();
  } else if (gestureName == "Ring") {
    showRing();
  } else if (gestureName == "Pinky") {
    showPinky();
  } else if (gestureName == "Three") {
    showThree();
  } else if (gestureName == "Four") {
    showFour();
  } else if (gestureName == "FourThumb") {
    showFourThumb();
  } else if (gestureName == "ThreePinky") {
    showThreePinky();
  } else if (gestureName == "ThreeThumb") {
    showThreeThumb();
  } else if (gestureName == "TwoPinky") {
    showTwoPinky();
  } else if (gestureName == "TwoThumb") {
    showTwoThumb();
  } else if (gestureName == "Spiderman") {
    showSpiderman();
  } else if (gestureName == "Rockstar") {
    showRockstar();
  } else if (gestureName == "WristForward") {
    showWristForward();
  } else if (gestureName == "WristBack") {
    showWristBack();
// Did not end up implementing wrist turning gesture
//  } else if (gestureName == "WristTurn") {
//    showWristTurn();   
  } else if (gestureName == "Relaxed") {
    showRelaxed(); // Paper and relaxed gesture are the same      
  } else {
    // If the gesture is unrecognized, default to reset position
    resetPositions();
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

// Stop motors to prevent them from breaking the mechanical 
// control system of the hand
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
  // Set a gesture to be in progress so that we don't perform multiple at a time
  gestureInProgress = true;
  // Set which motors to move
  setMotorState(thumbMotor1, true); 
  setMotorState(thumbMotor2, true); 
  setMotorState(indexMotor, true); 
  setMotorState(middleMotor, true); 
  setMotorState(ringMotor, true); 
  setMotorState(pinkyMotor, true);
  // Wait a second for all the motors to move to positions 
  delay(1000); stopAllMotors();
  // Hold the gesture for 1 second
  delay(1000);
  // Reset all the motors back to their original position
  setMotorState(thumbMotor1, false); 
  setMotorState(thumbMotor2, false); 
  setMotorState(indexMotor, false); 
  setMotorState(middleMotor, false); 
  setMotorState(ringMotor, false); 
  setMotorState(pinkyMotor, false); 
  // Wait a second for all the motors to move back
  delay(1000); stopAllMotors();
  // Set gesture in progress as false 
  gestureInProgress = false;
}

void showPaper() {
  // Paper is just an open hand
  // Can simply just set all motors to stop
  stopAllMotors();
}

void showScissors() {
  gestureInProgress = true;
  setMotorState(thumbMotor1, true);
  setMotorState(thumbMotor2, true);
  setMotorState(ringMotor, true);
  setMotorState(pinkyMotor, true);
  delay(1000); stopAllMotors();
  delay(1000);
  setMotorState(thumbMotor1, false); 
  setMotorState(thumbMotor2, false); 
  setMotorState(ringMotor, false); 
  setMotorState(pinkyMotor, false); 
  delay(1000); stopAllMotors();
  gestureInProgress = false;
}

void showPhone() {
  gestureInProgress = true;
  setMotorState(indexMotor, true); 
  setMotorState(middleMotor, true); 
  setMotorState(ringMotor, true); 
  delay(1000); stopAllMotors();
  delay(1000);
  setMotorState(indexMotor, false); 
  setMotorState(middleMotor, false); 
  setMotorState(ringMotor, false); 
  delay(1000); stopAllMotors();
  gestureInProgress = false;
}

void showPoint() {
  gestureInProgress = true;
  setMotorState(thumbMotor1, true); 
  setMotorState(thumbMotor2, true);  
  setMotorState(middleMotor, true); 
  setMotorState(ringMotor, true); 
  setMotorState(pinkyMotor, true); 
  delay(1000); stopAllMotors();
  delay(1000);
  setMotorState(thumbMotor1, false); 
  setMotorState(thumbMotor2, false);  
  setMotorState(middleMotor, false); 
  setMotorState(ringMotor, false); 
  setMotorState(pinkyMotor, false); 
  delay(1000); stopAllMotors();
  gestureInProgress = false;
}

void showThumb() {
  gestureInProgress = true;
  setMotorState(indexMotor, true); 
  setMotorState(middleMotor, true); 
  setMotorState(ringMotor, true); 
  setMotorState(pinkyMotor, true);  
  delay(1000); stopAllMotors();
  delay(1000);
  setMotorState(indexMotor, false); 
  setMotorState(middleMotor, false); 
  setMotorState(ringMotor, false); 
  setMotorState(pinkyMotor, false);
  delay(1000); stopAllMotors();
  gestureInProgress = false;
}

void showMiddle() {
  gestureInProgress = true;
  setMotorState(thumbMotor1, true);
  setMotorState(thumbMotor2, true);
  setMotorState(indexMotor, true);
  setMotorState(ringMotor, true);
  setMotorState(pinkyMotor, true);
  delay(1000); stopAllMotors();
  delay(1000);
  setMotorState(thumbMotor1, false);
  setMotorState(thumbMotor2, false);
  setMotorState(indexMotor, false);
  setMotorState(ringMotor, false);
  setMotorState(pinkyMotor, false);
  delay(1000); stopAllMotors();
  gestureInProgress = false;
}

void showRing() {
  gestureInProgress = true;
  setMotorState(thumbMotor1, true);
  setMotorState(thumbMotor2, true);
  setMotorState(indexMotor, true);
  setMotorState(middleMotor, true);
  setMotorState(pinkyMotor, true);
  delay(1000); stopAllMotors();
  delay(1000);
  setMotorState(thumbMotor1, false);
  setMotorState(thumbMotor2, false);
  setMotorState(indexMotor, false);
  setMotorState(middleMotor, false);
  setMotorState(pinkyMotor, false);
  delay(1000); stopAllMotors();
  gestureInProgress = false;
}

void showPinky() {
  gestureInProgress = true;
  setMotorState(thumbMotor1, true);
  setMotorState(thumbMotor2, true);
  setMotorState(indexMotor, true);
  setMotorState(middleMotor, true);
  setMotorState(ringMotor, true);
  delay(1000); stopAllMotors();
  delay(1000);
  setMotorState(thumbMotor1, false);
  setMotorState(thumbMotor2, false);
  setMotorState(indexMotor, false);
  setMotorState(middleMotor, false);
  setMotorState(ringMotor, false);
  delay(1000); stopAllMotors();
  gestureInProgress = false;
}

void showThree() {
  gestureInProgress = true;
  setMotorState(thumbMotor1, true);
  setMotorState(thumbMotor2, true);
  setMotorState(pinkyMotor, true);
  delay(1000); stopAllMotors();
  delay(1000);
  setMotorState(thumbMotor1, false);
  setMotorState(thumbMotor2, false);
  setMotorState(pinkyMotor, false);
  delay(1000); stopAllMotors();
  gestureInProgress = false;
}

void showFour() {
  gestureInProgress = true;
  setMotorState(thumbMotor1, true);
  setMotorState(thumbMotor2, true);
  delay(1000); stopAllMotors();
  delay(1000);
  setMotorState(thumbMotor1, false);
  setMotorState(thumbMotor2, false);
  delay(1000); stopAllMotors();
  gestureInProgress = false;
}

void showFourThumb() {
  gestureInProgress = true;
  setMotorState(pinkyMotor, true);
  delay(1000); stopAllMotors();
  delay(1000);
  setMotorState(pinkyMotor, false);
  delay(1000); stopAllMotors();
  gestureInProgress = false;
}

void showThreePinky() {
  gestureInProgress = true;
  setMotorState(thumbMotor1, true);
  setMotorState(thumbMotor2, true);
  setMotorState(indexMotor, true);
  delay(1000); stopAllMotors();
  delay(1000);
  setMotorState(thumbMotor1, false);
  setMotorState(thumbMotor2, false);
  setMotorState(indexMotor, false);
  delay(1000); stopAllMotors();
  gestureInProgress = false;
}

void showThreeThumb() {
  gestureInProgress = true;
  setMotorState(ringMotor, true);
  setMotorState(pinkyMotor, true);
  delay(1000); stopAllMotors();
  delay(1000);
  setMotorState(ringMotor, false);
  setMotorState(pinkyMotor, false);
  delay(1000); stopAllMotors();
  gestureInProgress = false;
}

void showTwoPinky() {
  gestureInProgress = true;
  setMotorState(thumbMotor1, true);
  setMotorState(thumbMotor2, true);
  setMotorState(indexMotor, true);
  setMotorState(middleMotor, true);
  delay(1000); stopAllMotors();
  delay(1000);
  setMotorState(thumbMotor1, false);
  setMotorState(thumbMotor2, false);
  setMotorState(indexMotor, false);
  setMotorState(middleMotor, false);
  delay(1000); stopAllMotors();
  gestureInProgress = false;
}

void showTwoThumb() {
  gestureInProgress = true;
  setMotorState(middleMotor, true);
  setMotorState(ringMotor, true);
  setMotorState(pinkyMotor, true);
  delay(1000); stopAllMotors();
  delay(1000);
  setMotorState(middleMotor, false);
  setMotorState(ringMotor, false);
  setMotorState(pinkyMotor, false);
  delay(1000); stopAllMotors();
  gestureInProgress = false;
}

void showSpiderman() {
  gestureInProgress = true;
  setMotorState(middleMotor, true);
  setMotorState(ringMotor, true);
  delay(1000); stopAllMotors();
  delay(1000);
  setMotorState(middleMotor, false);
  setMotorState(ringMotor, false);
  delay(1000); stopAllMotors();
  gestureInProgress = false;
}

void showRockstar() {
  gestureInProgress = true;
  setMotorState(thumbMotor1, true);
  setMotorState(thumbMotor2, true);
  setMotorState(middleMotor, true);
  setMotorState(ringMotor, true);
  delay(1000); stopAllMotors();
  delay(1000);
  setMotorState(thumbMotor1, false);
  setMotorState(thumbMotor2, false);
  setMotorState(middleMotor, false);
  setMotorState(ringMotor, false);
  delay(1000); stopAllMotors();
  gestureInProgress = false;
}

void showWristForward() {
  gestureInProgress = true;
  // Wrist movement forward
  setMotorState(WristMotorLeft, true);
  setMotorState(WristMotorRight, false);
  delay(200); stopAllMotors();
  delay(1000);
  setMotorState(WristMotorLeft, true);
  setMotorState(WristMotorRight, false);
  delay(200); stopAllMotors();
  gestureInProgress = false;
}

void showWristBack() {
  gestureInProgress = true;
  // Wrist movement backward
  setMotorState(WristMotorLeft, false);
  setMotorState(WristMotorRight, true);
  delay(200); stopAllMotors();
  delay(1000);
  setMotorState(WristMotorLeft, true);
  setMotorState(WristMotorRight, false);
  delay(200); stopAllMotors();
  gestureInProgress = false;
}

//Unfortunately, could not get to a wrist turn motion due to
//unforseen mechanical issue
//void showWristTurn() {
//  // Wrist-specific movement logic
//  rotateWrist();
//  delay(500); stopAllMotors();
//}
//


void showRelaxed() {
  stopAllMotors();
}

void resetPositions() {
  stopAllMotors();  
}
