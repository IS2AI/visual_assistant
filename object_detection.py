import sys
sys.path.append('../')
from utils.utils import *
import cv2
import time
import os

if __name__ == '__main__':
    pred = Predictor(engine_path='./models/yolov7_fp16.engine')
    pred.detect_video(4) # set 0 use a webcam
    pred.get_fps()
