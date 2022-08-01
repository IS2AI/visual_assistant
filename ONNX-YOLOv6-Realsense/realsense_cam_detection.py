import pyrealsense2 as rs
import numpy as np
import cv2
import random
import torch
import time
import onnxruntime

from YOLOv6 import YOLOv6

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush', 'face'] 
# Initialize YOLOv6 object detector
model_path = "models/best_ckpt.onnx"
yolov6_detector = YOLOv6(model_path, conf_thres=0.45, iou_thres=0.5)

def get_mid_pos(frame, box, depth_data, randnum):
    distance_list = []
    # returned bounding box format: [xmin, ymin, xmax, ymax, confidence, class, name]
    mid_pos = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2] # determine the center pixel positions of the index depth
    min_val = min(abs(box[2] - box[0]), abs(box[3]-box[1])) # determine depth search range 

    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)] # INDICES ARE SWITCHED IN ORIGINAL
        cv2.circle(frame, (int(mid_pos[0]+ bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)

        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
     # Median Filter in case any points will give irrelevant values (too small, too big)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4]

    return np.mean(distance_list)

def dectshow(org_img, boxes, cl_ids, depth_data):
    img = org_img.copy()
    for box, cl_id in zip(boxes,cl_ids):
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
        dist = get_mid_pos(org_img, box, depth_data, 24)
        # LOOK INTO PROPER CONVERSION OF DISTANCE
        cv2.putText(img, names[cl_id] + ' - ' + str(dist / 1000)[:4] + 'm',(int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),1)
    cv2.imshow('detecetion_image', img)

if __name__ == "__main__":
    # Configure depth and color streams
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # last parameter is fps
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Configure and start the pipeline
    pipe.start(config)

    # Create alignment primitive with color as its target stream
    align = rs.align(rs.stream.color)
    try:
        while True:
            frameset = pipe.wait_for_frames()
            frameset = align.process(frameset)
            depth_frame = frameset.get_depth_frame()
            color_frame = frameset.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            boxes, scores, class_ids = yolov6_detector(color_image)
            #print(boxes)
            
            dectshow(color_image, boxes, class_ids, depth_image)

            # Apply colormap on depth image (must be converted to 8-bit image first)
            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            #images = np.hstack((color_image, depth_colormap))
            #cv2.namedWindow('RealSense Output', cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('RealSense Output', images)
            
            key = cv2.waitKey(1)
            # Press esc or 'q' to close image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipe.stop()
   