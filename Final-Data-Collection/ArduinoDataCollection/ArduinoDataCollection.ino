// Arduino Script to Collect data from sensors

// Set places where sEMG sensors are connected 
const int emgPin1 = A0;
const int emgPin2 = A1;
const int emgPin3 = A2;
const int emgPin4 = A3;
// const int emgPin5 = A4;

// In an ideal world, you would use a sampling rate of around 1000 for a 
// cleaner signal, but we found it sufficient to use a rate of 100 for our 
// purposes
// If you do want to increase the sampling rate to 1000, simply replace every
// instance of millis with micros
const unsigned long sampleRate = 100; 
unsigned long previousMillis = 0;

void setup() {
    Serial.begin(115200); 
    // Setup pins we are connecting to sensors
    // Originally used 5 sensors but had to switch to 4 due to issues 
    // with our hardware
    pinMode(emgPin1, INPUT);
    pinMode(emgPin2, INPUT);
    pinMode(emgPin3, INPUT);
    pinMode(emgPin4, INPUT);
    // pinMode(emgPin5, INPUT);
}

void loop() {
    unsigned long currentMillis = millis();
    if (currentMillis - previousMillis >= (1000 / sampleRate)) {
        previousMillis = currentMillis;

        // Read pins to get signals from sensors 
        int emgValue1 = analogRead(emgPin1);
        int emgValue2 = analogRead(emgPin2);
        int emgValue3 = analogRead(emgPin3);
        int emgValue4 = analogRead(emgPin4);
        // int emgValue5 = analogRead(emgPin5);

        // Send all EMG values separated by a comma
        Serial.print(emgValue1);
        Serial.print(",");
        Serial.print(emgValue2);
        Serial.print(",");
        Serial.print(emgValue3);
        Serial.print(",");
        Serial.println(emgValue4);
        // Serial.print(",");
        // Serial.println(emgValue5);        
    }
}
