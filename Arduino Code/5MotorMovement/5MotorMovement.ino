#include <Servo.h>

// Code tested for second sprint review.
// Can move motors based on rock, paper, scissors movement

// Define servo objects
Servo thumbServo;
Servo indexServo;
Servo middleServo;
Servo ringServo;
Servo pinkyServo;

// Places where servos are connected
const int thumbPin = 3;
const int indexPin = 5;
const int middlePin = 6;
const int ringPin = 9;
const int pinkyPin = 10;

const int OPEN_POS = 0;     // Servo open position 
const int CLOSED_POS = 180;  // Servos closed position
 
void setup() {
  // Attach servos to their ports
  thumbServo.attach(thumbPin);
  indexServo.attach(indexPin);
  middleServo.attach(middlePin);
  ringServo.attach(ringPin);
  pinkyServo.attach(pinkyPin);

  // Set all servos to open
  thumbServo.write(OPEN_POS);
  indexServo.write(OPEN_POS);
  middleServo.write(OPEN_POS);
  ringServo.write(OPEN_POS);
  pinkyServo.write(OPEN_POS);

  Serial.begin(115200);
}

void loop() {
  // Read and print EMG data from A0 pin
  int emgValue = analogRead(A0);
  Serial.print("EMG Value: ");
  Serial.println(emgValue);

  // Determine gesture based on threshold EMG value
  if (emgValue >= 100 && emgValue <= 200) { // Show paper gesture 
    Serial.println("Gesture: Paper"); 
    showPaper();
  } else if (emgValue < 100) { // Show scissor gesture
    Serial.println("Gesture: Scissors");
    showScissors();
  } else if (emgValue > 190) { // Show rock gesture
    Serial.println("Gesture: Rock");
    showRock();
  } else {
    Serial.println("Gesture: None");
  }

  delay(500);
}

// Function to show Paper gesture
void showPaper() {
  thumbServo.write(OPEN_POS);
  indexServo.write(OPEN_POS);
  middleServo.write(OPEN_POS);
  ringServo.write(OPEN_POS);
  pinkyServo.write(OPEN_POS);
}

// Function to show Scissors gesture
void showScissors() {
  thumbServo.write(CLOSED_POS);
  indexServo.write(OPEN_POS);
  middleServo.write(OPEN_POS);
  ringServo.write(CLOSED_POS);
  pinkyServo.write(CLOSED_POS);
}

// Function to show Rock gesture
void showRock() {
  thumbServo.write(CLOSED_POS);
  indexServo.write(CLOSED_POS);
  middleServo.write(CLOSED_POS);
  ringServo.write(CLOSED_POS);
  pinkyServo.write(CLOSED_POS);
}
