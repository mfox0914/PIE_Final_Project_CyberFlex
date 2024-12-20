# Script made to address signal drift issue in second sprint
# To prevent signals from drifting, we simply use this script stop and start 
# the python script which predicts the gesture instead of letting it run 
# continuously

# Did not end up using in final interation, however we thought it would be 
# interesting to include

import subprocess

while True:
    try:
        subprocess.run(["python", "Final-CNN-Real-Time-Predict-no-filter.py"])
    except KeyboardInterrupt:
        print("Exiting...")
        break