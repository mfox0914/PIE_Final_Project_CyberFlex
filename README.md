# PIE_Final_Project_CyberFlex
Welcome to our repo for CyberFlex!

This repo hold all the final material needed to create your own version of the robotic hand that we (Darian Jimenez, Maya Adelman, Esther Aduamah, Michaela Fox, and Suwanee Li) created for our Principles of Integrated Engineering Final Project.

## Setup
First open the `Final-Data-Collection` folder. You must collect data using `Final-Sensor-Data-Collection-no-filter.py` if you are using the regular signal pin on the myoware sensors. Also be sure to connect your Arduino Uno to your sensors using the analog pins (as well as to power) and run `ArduinoDataCollection.ino` to begin collecting data.

After your data has been collected, you can train a model by using the first demo model in this Google COLAB notebook: 

https://colab.research.google.com/drive/1LpTZBAKLXA85sSInc_OY33snMsKJWvS4?usp=sharing

Lastly, save the path to your model and load it in this directory. Next open the `Final-RasPi-Arduino` folder. Then simply run `Final-CNN-Real-Time-Predict-no-filer.py` while wearing the sensors and ensure `DC_Real_Time.ino` is running on your Arduino. If you make a gesture, you should see what you labeled it as in your terminal!


## External Software Dependencies
`import torch
import numpy as np
import serial
import torch.nn as nn
import torch.nn.functional as F
import time
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, detrend
import pandas as pd
import os`

Note: Each folder 'Final-RasPi-Arduino' and 'Final-Data-Collection' has two files, one for filtered data and for non filtered emg data. Choose teh one most suited for your scenario.
