#include <Servo.h>


// Threshold values for controlling the servo motor.
#define OPEN_THRESHOLD 150  // Set this based on the sensor value when hand is open.
#define CLOSE_THRESHOLD 200 // Set this based on the sensor value when hand is closed.


// Pin number where the sensor is connected. (Analog 0)
#define EMG_PIN A0


// Pin number where the servo motor is connected. (Digital PWM 3)
#define SERVO_PIN 3


// Define Servo motor
Servo SERVO_1;


void setup(){
  // Set Baud Rate to 115200. Remember to set Serial Monitor to match.
  Serial.begin(115200);
 
  // Attach the servo motor to digital pin 3
  SERVO_1.attach(SERVO_PIN);
}


void loop(){
  // Read the analog value from the sensor.
  int value = analogRead(EMG_PIN);


  // Check for "open" state
  if(value > OPEN_THRESHOLD){
    SERVO_1.write(170); // Turn servo to 170 degrees for open hand
  }
  // Check for "close" state
  else if(value < CLOSE_THRESHOLD){
    SERVO_1.write(10); // Turn servo back to 10 degrees for closed hand
  }
  Serial.println(value);
  delay(20);
}
