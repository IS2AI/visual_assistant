import time, subprocess, signal
from subprocess import Popen, PIPE
import sys, os
import psutil

import inquirer
print('Welcome to the Artificial Vision System For Blinds')

while True:
    questions = [
                inquirer.List('Choice',
                            message="Choose a program ?",
                            choices=['Scan environment using Object detection', 'Text to speech', 'Spatial audio', 'OCR', 'Exit'],)]
                
    answers = inquirer.prompt(questions)               

    if answers['Choice'] == 'Scan environment using Object detection':
        cmd = 'gnome-terminal -x sh -c "export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.8/pyrealsense2;python3 object_detection_realsense.py"'
        p1 = subprocess.Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    if answers['Choice'] == 'Text to speech':
        p1.kill()    
    if answers['Choice'] == 'Spatial':
        p1.kill()


    if answers['Choice'] == 'Exit':
        sys.exit()

   # message = input("Press enter to quit\n\n")











