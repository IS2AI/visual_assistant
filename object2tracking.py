import sys
sys.path.append('../')
from utils.utils_norfair import preproc, vis
from utils.utils_norfair import Predictor
from utils.realsense_depth import *
import cv2
import time
from PyInquirer import style_from_dict, Token, prompt


Classes = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush', 'face']

if __name__ == '__main__':
    pred = Predictor(engine_path='./models/yolov6_fp16.engine')
    dc = DepthCamera()
    style = style_from_dict({ Token.QuestionMark: '#E91E63 bold', Token.Selected: '#00FFFF', Token.Instruction: '', Token.Answer: '#2196f3 bold', Token.Question: '#7FFF00 bold',})
    time.sleep(0.2)
    class_option=[ 
    {
        'type':'list',
        'name':'class',
        'message':'Class for tracking:',
        'choices': Classes,
    }
]
    class_answer=prompt(class_option,style=style)
    class_to_track=class_answer['class']
    try:
        while True:
            ret, depth_image, color_image, depth_color_image = dc.get_frame()
            origin_img = pred.inference(color_image, depth_image, class_to_track, conf = .4)
            #rotate_180 = cv2.rotate(depth_color_image, cv2.ROTATE_180) 
            cv2.imshow('detections',origin_img)
            cv2.imshow('depth', depth_color_image)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        dc.release()



    