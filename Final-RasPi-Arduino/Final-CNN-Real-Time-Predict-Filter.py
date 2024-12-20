# This is the final script which is used to predict which gesture is being made
# based on emg values given by the arduino. This is then communicated to 
# the Arduino which moves motors base don this prediction

import torch
import numpy as np
import serial
import torch.nn as nn
import torch.nn.functional as F
import joblib
import time
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, detrend

# Serial communication setup
serial_port = '/dev/ttyACM0'
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate)

# Parameters
window_size = 150  # Number of samples per input window
fs = 100          # Sampling frequency (Hz)
lowcut = 30.0      # Low cutoff frequency for bandpass filter
highcut = 300.0    # High cutoff frequency for bandpass filter

# Bandpass filter 
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

b, a = butter_bandpass(lowcut, highcut, fs)

def process_emg_data(raw_emg_data):
    """
    Processes raw EMG data (detrending, bandpass filtering, and normalization)
    """
    emg_detrended = detrend(raw_emg_data) # Detrend the data to remove drift
    emg_filtered = filtfilt(b, a, emg_detrended) # Apply bandpass filter
    # Normalize the filtered data
    emg_filtered = np.array(emg_filtered).reshape(1, -1)
    emg_normalized = scaler.transform(emg_filtered)

    return emg_normalized

# Define CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes, window_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * (window_size // 4), 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# Load trained CNN model
model_path = 'test_cnn_emg_model.pth'
model_cnn = CNN(num_classes=3, window_size=window_size)
model_cnn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model_cnn.eval()

# Load scaler used to scale data when training model
scaler = joblib.load('scaler.save')

# Map gestures 
gesture_mapping = {0: 'Rock', 
                   1: 'Paper', 
                   2: 'Scissors'}

def update_scaler(scaler, new_data, alpha=0.1):
    """
    Updates the scaler using exponential moving average (EMA) with new data.
    """
    new_mean = alpha * np.mean(new_data) + (1 - alpha) * scaler.mean_
    new_var = alpha * np.var(new_data) + (1 - alpha) * scaler.var_
    scaler.mean_ = new_mean
    scaler.var_ = new_var
    scaler.scale_ = np.sqrt(new_var)

def get_emg_data(window_size=150):
    """
    Collects live EMG data from the Arduino over the serial port.
    """
    emg_data = []
    while len(emg_data) < window_size:
        try:
            line = ser.readline().decode('utf-8').strip()
            parts = line.split(',')  # Parse CSV format: "timestamp,emg_value"
            if len(parts) == 2:
                timestamp, emg_value = int(parts[0]), int(parts[1])
                emg_data.append(emg_value)
        except (ValueError, IndexError, UnicodeDecodeError):
            continue
    return emg_data

def predict_gesture(emg_data):
    """
    Predicts the gesture based on the processed EMG data using the CNN model.
    """
    emg_processed = process_emg_data(emg_data)
    emg_tensor = torch.FloatTensor(emg_processed).unsqueeze(1)  # Add channel dimension

    with torch.no_grad():
        output = model_cnn(emg_tensor)
        _, predicted = torch.max(output, 1)  # Get the class with the highest probability
    return predicted.item()

def send_to_arduino(gesture_name):
    """
    Sends the predicted gesture to the Arduino over the serial port.
    """
    ser.write(f"{gesture_name}\n".encode('utf-8'))
    time.sleep(0.1)
    while ser.in_waiting > 0:
        response = ser.readline().decode('utf-8').strip()
        print(f"Arduino response: {response}")

# Main loop
try:
    while True:
        # Collect EMG data
        raw_emg_data = get_emg_data(window_size=window_size)
        # Update scaler to avoid drift in signals
        update_scaler(scaler, raw_emg_data)
        # Predict gesture using CNN model
        gesture = predict_gesture(raw_emg_data)
        gesture_name = gesture_mapping.get(gesture, 'Unknown')
        # Output predicted gesture
        print(f"Predicted Gesture: {gesture_name}")
        # Send gesture to Arduino
        send_to_arduino(gesture_name)
except KeyboardInterrupt:
    print("Exiting...")
    ser.close()
