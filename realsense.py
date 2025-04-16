import sys
sys.path.append('../')
from utils.realsense_depth import *
import cv2
import time
import os


if __name__ == '__main__':
    dc = DepthCamera()
    try:
        print("Intel Realsense D435 started successfully.")
        while True:
            ret, depth_image, color_image = dc.get_frame()
            cv2.imshow("Color frame", color_image)
            cv2.imshow("Depth frame", depth_image)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        dc.release()                                                                                                                                                                                            