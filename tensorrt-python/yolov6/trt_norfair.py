import sys
sys.path.append('../')
from utils.utils_norfair import preproc, vis
from utils.utils_norfair import BaseEngine
import numpy as np
import cv2
import time
import os
import pyrealsense2 as rs
import time


class Predictor(BaseEngine):
    def __init__(self, engine_path , imgsz=(640,640)):
        super(Predictor, self).__init__(engine_path)
        self.imgsz = imgsz
        self.show_color_flag = True ## CHANGE TO DISABLE COLOR DETECTION
        self.n_classes = 81
        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush', 'face' ]


if __name__ == '__main__':
    # Configure depth and color streams
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60) # last parameter is fps
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

    # Configure and start the pipeline
    pipe.start(config)

    # Create alignment primitive with color as its target stream
    align = rs.align(rs.stream.color)

    pred = Predictor(engine_path='yolov6-nano-nms-fp32.trt')

    fps_array = []
    timer = 0
    try:
        while timer < 300:
            frameset = pipe.wait_for_frames()
            frameset = align.process(frameset)
            depth_frame = frameset.get_depth_frame()
            color_frame = frameset.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            #boxes, scores, class_ids = yolov6_detector(color_image)
            #print(boxes)
            
            #dectshow(color_image, boxes, class_ids, depth_image)
            t1 = time.time()
            origin_img = pred.inference(color_image, depth_data = depth_image, conf = .4)
            t2 = time.time()
            inference_time = 1/(t2-t1)
            # fps_array.append(round(inference_time,1))
            #origin_img = cv2.putText(origin_img, str(round(inference_time,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2, cv2.LINE_AA)
            cv2.imshow('detections',origin_img)
            # Apply colormap on depth image (must be converted to 8-bit image first)
            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            #images = np.hstack((color_image, depth_colormap))
            #cv2.namedWindow('RealSense Output', cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('RealSense Output', images)
            # print(timer)
            # timer +=1
            key = cv2.waitKey(1)
            # Press esc or 'q' to close image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # print(f'Average FPS: {np.mean(fps_array)}')
        pipe.stop()




    # pred = Predictor(engine_path='yolov6-nano-nms-fp32.trt')
    # img_path = '../src/office_env_people.jpg'
    # origin_img = pred.inference(img_path)
    # cv2.imshow('detections',origin_img)
    # pred.get_fps()
    # key = cv2.waitKey(1000)
    # # Press esc or 'q' to close image window
    # if key & 0xFF == ord('q') or key == 27:
    #     cv2.destroyAllWindows()
    #cv2.imwrite("%s_yolov6.jpg" % os.path.splitext(
    #    os.path.split(img_path)[-1])[0], origin_img)
    #pred.detect_video('../src/video1.mp4') # set 0 use a webcam
    
