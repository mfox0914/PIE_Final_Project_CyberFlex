# File used to collect preproccesed data from sEMG sensors 

# To collect data connect to the arduino with all sensors connected to analog 
# pins, push the 'ArduinoDataCollection' Arduino script seen in the same folder 
# as this code, create the folder where you want to save your data and run this 
# python script, follow the instructions which appear in your terminal

import serial
import time
import pandas as pd
import os

SERIAL_PORT = '/dev/ttyACM0' # Set serial port where arduino is connected
BAUD_RATE = 115200 # Match Baud Rate to Arduino
DATA_DIR = './emg_final_data_collection/'

# Dataset collection configuration
# Set which gestures you want to collect data for
GESTURES = ['rock', 
            'paper', 
            'scissors',
            'phone',
            'point',
            'thumb',
            'pinky',
            'three',
            'four',
            'four-thumb',
            'three-pinky',
            'three-thumb',
            'two-pinky',
            'two-thumb',
            'spiderman',
            'relaxed'] # Added relaxed gesture to account for no gesture being done 
DURATION = 0.5 # Set duration of collecting data (in seconds)
REPETITIONS = 50 # Set amount of repetitions you want to collect data for

# Define function to collect emg data
def collect_emg_data(gesture_label, repetition, duration=DURATION):
    '''
    Connects to Arduino and collects sEMG data for a given gesture. 

    Args:
        gesture_label: String containing name of the gesture
        repetition: Integer representing number 
        Duration: Float determining how long to reco
    '''
    print(f"Connecting to Arduino. Perform Gesture Now")
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            time.sleep(1)
            print(f"Recording '{gesture_label}' (Rep: {repetition})...")

            start_time = time.time()
            data = []

            while time.time() - start_time < duration:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if line:
                        values = line.split(',')
                        if len(values) == 4: # Collect data if there are 4 emg values
                            emg_value1 = int(values[0])
                            emg_value2 = int(values[1])
                            emg_value3 = int(values[2])
                            emg_value4 = int(values[3])
                            # emg_value5 = int(values[4])
                            timestamp = time.time() # Add timestamp
                            data.append([timestamp, emg_value1, emg_value2, emg_value3, emg_value4]) # Append values to CSV
                        else:
                            print(f"Malformed data: {line}")
                    else:
                        print("Skipping empty line...")
                except ValueError:
                    print(f"Invalid data received: {line}")
                except Exception as e:
                    print(f"Error reading data: {e}")
                    break

            if data:
                # Create dataframe to store data
                df = pd.DataFrame(data, columns=['timestamp', 'emg_value_sensor1', 'emg_value_sensor2', 'emg_value_sensor3', 'emg_value_sensor4'])
                # Create file name
                file_path = os.path.join(DATA_DIR, f"{gesture_label}_rep{repetition}_{int(start_time)}.csv")
                # Save dataframe to csv file
                df.to_csv(file_path, index=False)
                print(f"Data for '{gesture_label}' (Rep: {repetition}) saved to {file_path}.")
            else:
                # Add print statement if no data is collected for debugging purposes
                print(f"No data collected for '{gesture_label}' (Rep: {repetition}).")

    # Add print statements if no data is collected for debugging purposes
    except serial.SerialException as e:
        print(f"Error connecting to Arduino: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Loop to collect data
if __name__ == "__main__":
    # Loop over gestures and repetitions, guiding the use along the process in the terminal
    print("Starting dataset collection...")
    for gesture in GESTURES:
        for rep in range(1, REPETITIONS + 1):
            # Wait for the user to press enter
            input(f"(Rep {rep}/{REPETITIONS}): Prepare to perform '{gesture}' gesture and press Enter to start...")
            # Collect data for gesture
            collect_emg_data(gesture_label=gesture, repetition=rep, duration=DURATION)
    print("Data collection complete.")
