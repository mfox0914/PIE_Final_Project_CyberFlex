# File used to collect proccesed data from sEMG sensors 

# As seen in 'Sensor-Collection.py', the sEMG sensors already have an output
# with an envelope filter, but we thought it would be a great learning 
# oppourtunity to filter the signals ourself by taking the RAW signal output
# from the sensors.

# To collect data connect to the arduino with all sensors connected to analog 
# pins, push the 'ArduinoDataCollection' Arduino script seen in the same folder 
# as this code, run this python script and follow the instructions which appear 
# in your terminal

# Import necessary modules and libraries
import serial
import time
import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt

SERIAL_PORT = '/dev/ttyACM0'  # Set serial port where arduino is connected
BAUD_RATE = 115200
DATA_DIR = './emg_data_5_sensor/'

# Dataset collection configuration
GESTURES = ['rock', 
            'paper', 
            'scissors',
            'phone',
            '4fingers',
            'spiderman']
DURATION = 2 # seconds
REPETITIONS = 30

# Bandpass filter settings
fs = 100  # Sampling frequency
lowcut = 30.0
highcut = 300.0
n_bits = 10
Vcc = 5.0
Gain = 1000

# Create output directory for data values
os.makedirs(DATA_DIR, exist_ok=True)

# Bandpass filter to process emg data
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

b, a = butter_bandpass(lowcut, highcut, fs)

def process_emg_data(emg_values):
    """
    Convert, filter, rectify, and calculate envelope.
    """
    # Convert raw values to voltage 
    emg_converted = ((np.array(emg_values) / (2 ** n_bits)) * Vcc / Gain) * 1000 
    # Apply bandpass filter
    emg_filtered = filtfilt(b, a, emg_converted)
    # Full wave rectification
    emg_rectified = np.abs(emg_filtered)
    # Calculate RMS envelope
    window_size = int(0.1 * fs) 
    emg_envelope = np.sqrt(
        np.convolve(emg_rectified ** 2, np.ones(window_size) / window_size, mode='same')
    )

    return emg_filtered, emg_rectified, emg_envelope

def collect_emg_data(gesture_label, repetition, duration=DURATION):
    '''
    Function used to collect data
    '''
    print(f"Connecting to Arduino on {SERIAL_PORT}...")
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            time.sleep(1)
            print(f"Recording gesture '{gesture_label}' (Rep {repetition})...")
            
            start_time = time.time()
            raw_data = []
            processed_data = []
            timestamps = []

            while time.time() - start_time < duration:
                try:
                    # Read data from serial
                    line = ser.readline().decode('utf-8').strip()
                    if line:
                        # Parse incoming data
                        values = [int(val) for val in line.split(',') if val]
                        if len(values) == 4:  # Collect data if there are 4 emg values
                            # Timestamp
                            timestamp = time.time()
                            timestamps.append(timestamp)
                            # Save raw values
                            raw_data.append(values)
                            # Process EMG data
                            filtered, rectified, envelope = process_emg_data(values)
                            # Save processed data
                            processed_data.append([filtered.tolist(), rectified.tolist(), envelope.tolist()])
                        else:
                            print(f"Malformed data: {line}")
                except ValueError:
                    print(f"Invalid data received: {line}")
                except Exception as e:
                    print(f"Error reading data: {e}")
                    break

            # Save raw data to CSV
            if raw_data:
                raw_df = pd.DataFrame(raw_data, columns=['sensor1', 'sensor2', 'sensor3', 'sensor4'])
                raw_file_path = os.path.join(DATA_DIR, f"{gesture_label}_rep{repetition}_raw.csv")
                raw_df.to_csv(raw_file_path, index=False)
                print(f"Raw data for '{gesture_label}' saved to {raw_file_path}.")
            else:
                # Add print statement if no data is collected for debugging purposes
                print(f"No data collected")

            # Save processed data to CSV
            if processed_data:
                processed_df = pd.DataFrame(
                    {'timestamp': timestamps,
                    'envelope': [x[2] for x in processed_data]}  # Save only RMS envelope
                )
                processed_file_path = os.path.join(DATA_DIR, f"{gesture_label}_rep{repetition}_processed.csv")
                processed_df.to_csv(processed_file_path, index=False)
                print(f"Processed data for '{gesture_label}' saved to {processed_file_path}.")

            else:
                print(f"No processed data collected for '{gesture_label}'.")

    # Add print statements if no data is collected for debugging purposes
    except serial.SerialException as e:
        print(f"Error connecting to Arduino: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    print("Starting dataset collection...")
    for gesture in GESTURES:
        for rep in range(1, REPETITIONS + 1):
            input(f"(Rep {rep}/{REPETITIONS}): Prepare to perform the '{gesture}' gesture and press Enter to start...")
            collect_emg_data(gesture_label=gesture, repetition=rep, duration=DURATION)
    print("Dataset collection complete.")
