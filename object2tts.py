import sys
sys.path.append('../')
from utils.utils_spatial import preproc, vis
from utils.utils_spatial import Predictor
from utils.realsense_depth import *
import cv2
import inquirer

if __name__ == '__main__':
    pred = Predictor(engine_path='./models/yolov7_fp16.engine')
    dc = DepthCamera()
    try:
        while True:
            ret, depth_image, color_image = dc.get_frame()
            questions = [
             inquirer.List('Choice',
                        message="Do you want continue ?",
                        choices=['Scan envinvorment', 'NO'],)]
            answers = inquirer.prompt(questions)              
            if answers['Choice'] == 'NO':
                origin_img = pred.inference(color_image, depth_image, conf = .4)
                cv2.imshow('detections',origin_img)
                cv2.imshow('depth', depth_image)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        dc.release()



    