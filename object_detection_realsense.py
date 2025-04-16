import sys
sys.path.append('../')
from utils.utils_color import preproc, vis
from utils.utils_color import Predictor
from utils.realsense_depth import *
import cv2


if __name__ == '__main__':
    pred = Predictor(engine_path='./models/yolov6_fp16.engine')
    dc = DepthCamera()
    try:
        while True:
            ret, depth_image, color_image = dc.get_frame()
            origin_img = pred.inference(color_image, depth_image, conf = .4)
            cv2.imshow('detections',origin_img)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        dc.release()



    