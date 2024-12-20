#include <Servo.h>

// This was the code used for the first sprint review
// It takes threshold values from the sensors
// Opening the robotic hand if the signal is above that value
// Closing the hand if the signals is below that value

// Threshold values for controlling servo
#define OPEN_THRESHOLD 150  // sensor value when hand is open.
#define CLOSE_THRESHOLD 200 // sensor value when hand is closed.

#define EMG_PIN A0
#define SERVO_PIN 3
Servo SERVO_1;


void setup(){
  Serial.begin(115200);
  SERVO_1.attach(SERVO_PIN);
}


void loop(){
  // Read analog value from the sensor.
  int value = analogRead(EMG_PIN);

  // Check if open
  if(value > OPEN_THRESHOLD){
    SERVO_1.write(170); // Turn servo to 170 degrees for open hand
  }
  // Check if closed
  else if(value < CLOSE_THRESHOLD){
    SERVO_1.write(10); // Turn servo back to 10 degrees for closed hand
  }
  Serial.println(value);
  delay(2);
}
