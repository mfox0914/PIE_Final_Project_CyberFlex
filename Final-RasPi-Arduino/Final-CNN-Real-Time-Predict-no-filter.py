# This is the final script which is used to predict which gesture is being made
# based on emg values given by the arduino. This is then communicated to 
# the Arduino which moves motors base don this prediction

import torch
import numpy as np
import serial
import torch.nn as nn
import torch.nn.functional as F
import time

# Set serial_port for arduino and match baud rate to arduino
serial_port = '/dev/ttyACM0'
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate)

window_size = 150

# Define CNN architecture (see ipynb file for more on how this functions)
class CNN(nn.Module):
    def __init__(self, num_classes, window_size=window_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(x)
        return x


# Load Trained Model which we saved from the ipynb script
model_path = '9gesture_model.pth' # Update depending on path
model_cnn = CNN(num_classes=9, window_size=window_size)
model_cnn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
model_cnn.eval()

# Below are absolutely all the gestures we collected data for
# While training the model 'final_model.pth' we were able to achieve just 
# under 90% accuracy for each of the gestures (This is with 50 points of data
# for each gesture)
# gesture_mapping = {0:'Rock', 
#                    1:'Paper', 
#                    2:'Scissors', 
#                    3:'Phone', 
#                    4:'Point',
#                    5:'Thumb',
#                    6:'Middle',
#                    7:'Ring',
#                    8:'Pinky',
#                    9:'Three',
#                    10:'Four',
#                    11:'FourThumb',
#                    12:'ThreePinky',
#                    13:'ThreeThumb',
#                    14:'TwoPinky',
#                    15:'TwoThumb',
#                    16:'Spiderman',
#                    17:'Rockstar',
#                    18:'WristForward',
#                    19:'WristBack',
#                    20:'WristTurn',
#                    21:'Relaxed'}

# In order to improve this and simplify our system, we scaled back to only
# using the 16 gestures found below
# We also collected more data, now using 100 data points to identify each gesture
# gesture_mapping = {0:'Rock', 
#                    1:'Paper', 
#                    2:'Scissors', 
#                    3:'Phone', 
#                    4:'Point',
#                    5:'Thumb',
#                    6:'Pinky',
#                    7:'Three',
#                    8:'Four',
#                    9:'FourThumb',
#                    10:'ThreePinky',
#                    11:'ThreeThumb',
#                    12:'TwoPinky',
#                    13:'TwoThumb',
#                    14:'Spiderman',
#                    15:'Relaxed'} # Add relaxed gesture so that the hand can reset if no gesture is being made

# Scaling back even further to 9 gestures
gesture_mapping = {0:'Rock', 
                   1:'Paper', 
                   2:'Scissors', 
                   3:'Phone', 
                   4:'Thumb',
                   5:'Pinky',
                   6:'Three',
                   7:'Four',
                   8:'Spiderman',
                   9:'Relaxed'} # Add relaxed gesture so that the hand can reset if no gesture is being made

def get_emg_data(window_size=window_size):
    '''
    Function to read 4 emg sensor data points from arduino script
    '''
    emg_data = []
    while len(emg_data) < window_size:
        try:
            line = ser.readline().decode('utf-8').strip()
            values = [int(x) for x in line.split(',')]
            if len(values) == 4:
                emg_data.append(values)
        except (ValueError, UnicodeDecodeError):
            continue
    return np.array(emg_data).T

def predict_gesture(emg_data):
    '''
    Function to predict which gesture is being made using the CNN
    '''
    emg_tensor = torch.FloatTensor(emg_data).unsqueeze(0) 

    with torch.no_grad():
        output = model_cnn(emg_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def send_to_arduino(gesture_name):
    '''
    Function to send gesture to arduino so that it can move the motors
    '''
    ser.write(f"{gesture_name}\n".encode('utf-8'))
    time.sleep(0.2)
    while ser.in_waiting > 0:
        response = ser.readline().decode('utf-8').strip()
        print(f"Arduino response: {response}")

# Main Loop to Loop through script
try:
    while True:
        emg_data = get_emg_data(window_size=window_size)
        gesture = predict_gesture(emg_data)
        gesture_name = gesture_mapping.get(gesture)
        print(f"Predicted Gesture: {gesture_name}")
        send_to_arduino(gesture_name)
except KeyboardInterrupt:
    print("Exiting...")
    ser.close()