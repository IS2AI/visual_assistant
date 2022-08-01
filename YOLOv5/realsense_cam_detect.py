# NEED to install library in conda environment first
import pyrealsense2 as rs
import numpy as np
import cv2
import random
import torch
import time
import schedule
from gtts import gTTS
import keyboard
import os
from playsound import playsound

# Import your model from local machine
#model = torch.hub.load('ultralytics/yolov5', 'yolov5n') 
# MODIFY THIS PATH TO POINT TO YOUR MODEL
model = torch.hub.load('/Users/mukal/yolov5', 'custom', path='/Users/mukal/yolov5/runs/train/cocoface_nano/weights/best.pt', source='local')  # local repo
model.conf = 0.3

rus_translations = {
    'person':'человек',
    'bicycle':'велосипед', 
    'car': 'машина', 
    'motorcycle': 'мотоцикл', 
    'airplane': 'самолет', 
    'bus' : 'автобус',
    'train': 'поезд', 
    'truck': 'грузовик', 
    'boat': 'лодка', 
    'traffic light': 'светофор',
    'fire hydrant': 'пожарный гидрант', 
    'stop sign': 'знак стоп', 
    'parking meter': 'паркометр', 
    'bench': 'лавочка',
    'bird': 'птица',
    'cat': 'кошка',
    'dog': 'собака',
    'horse': 'лошадь',
    'sheep': 'овца', 
    'cow': 'корова',
    'elephant': 'слон', 
    'bear': 'медведь', 
    'zebra': 'зебра', 
    'giraffe': 'жираф', 
    'backpack': 'рюкзак', 
    'umbrella': 'зонтик', 
    'handbag': 'сумка',
    'tie': 'галстук',
    'suitcase': 'чемодан', 
    'frisbee': 'фрисби',
    'skis': 'лыжи', 
    'snowboard': 'сноуборд', 
    'sports ball': 'мяч',
    'kite': 'летающий змей', 
    'baseball bat': 'бейсбольная бита', 
    'baseball glove': 'бейсбольная перчатка', 
    'skateboard': 'скейтборд', 
    'surfboard': 'серфборд',
    'tennis racket': 'теннисная ракетка', 
    'bottle': 'бутылка', 
    'wine glass': 'бокал', 
    'cup': 'кружка', 
    'fork': 'вилка', 
    'knife': 'нож', 
    'spoon': 'ложка', 
    'bowl': 'миска', 
    'banana': 'банан', 
    'apple': 'яблоко',
    'sandwich': 'сэндвич', 
    'orange': 'апельсин', 
    'broccoli': 'брокколи', 
    'carrot': 'морковь', 
    'hot dog': 'хот дог', 
    'pizza': 'пицца', 
    'donut': 'пончик', 
    'cake': 'торт', 
    'chair': 'стул', 
    'couch': 'диван',
    'potted plant': 'домашнее растение', 
    'bed': 'кровать', 
    'dining table': 'стол', 
    'toilet': 'унитаз', 
    'tv': 'телевизор', 
    'laptop': 'ноутбук', 
    'mouse': 'мышка', 
    'remote': 'пульт', 
    'keyboard': 'клавиатура', 
    'cell phone': 'мобильный телефон',
    'microwave': 'микроволновка', 
    'oven': 'печь', 
    'toaster': 'тостер', 
    'sink': 'раковина', 
    'refrigerator': 'холодильник', 
    'book': 'книга', 
    'clock': 'часы', 
    'vase': 'ваза', 
    'scissors': 'ножница', 
    'teddy bear': 'плюшевая игрушка',
    'hair drier': 'фен', 
    'toothbrush': 'зубная щетка', 
    'face': 'лицо'
}

def get_mid_pos(frame, box, depth_data, randnum):
    distance_list = []
    # returned bounding box format: [xmin, ymin, xmax, ymax, confidence, class, name]
    mid_pos = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2] # determine the center pixel positions of the index depth
    min_val = min(abs(box[2] - box[0]), abs(box[3]-box[1])) # determine depth search range 

    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        dist = depth_data[int(mid_pos[1] + bias//6), int(mid_pos[0] + bias)]
        cv2.circle(frame, (int(mid_pos[0]+ bias), int(mid_pos[1] + bias//6)), 4, (255,0,0), -1)

        if dist:
            distance_list.append(dist)
    
    distance_list = np.array(distance_list)
     # Median Filter in case any points will give irrelevant values (too small, too big)
    distance_list = np.sort(distance_list)[randnum//2-randnum//3:randnum//2+randnum//3]
    
    mean_out = 0
    if len(distance_list) > 0:
        mean_out = np.mean(distance_list)
    return mean_out

def dectshow(org_img, boxes, depth_data, profile):
    global print_bbxs
    print_bbxs = []
    print_bbxs.append("Следующие объекты были обнаружены возле вас: ")
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    img = org_img.copy()
    for box in boxes:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)
        dist = get_mid_pos(org_img, box, depth_data, 36)
        # LOOK INTO PROPER CONVERSION OF DISTANCE
        if dist * depth_scale:
            if(not rus_translations[box[-1]]=="лицо"):
                sentence_object_det = rus_translations[box[-1]] + ' в ' + str(dist * depth_scale).split(".")[0] + ' метрах ' + str(dist * depth_scale)[:4].split(".")[1] + ' сантиметрах.'
                if  str(dist * depth_scale).split(".")[0] == "0": sentence_object_det = sentence_object_det.replace(' 0 метрах','')
                if str(dist * depth_scale)[:4].split(".")[1] == "00": sentence_object_det = sentence_object_det.replace(' 00 сантиметров','')
                if str(dist * depth_scale)[:4].split(".")[1][0] == "0": 
                    sentence_object_det = sentence_object_det[:sentence_object_det.rfind("0")] + str(dist * depth_scale)[:4].split(".")[1][1] + ' сантиметрах.'
                print_bbxs.append(sentence_object_det)
        cv2.putText(img, box[-1] + str(dist * depth_scale)[:4] + 'm',(int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),1)
    cv2.imshow('detecetion_image', img)
    
    

def print_preds():
    global print_bbxs
    print_bbxs.append('\n')
    text = "\n".join(print_bbxs)
    tts = gTTS(text = text, lang = 'ru', slow = False)
    if os.path.exists(os.path.join(os.getcwd(),'det_vocalize.mp3')):
        os.remove(os.path.join(os.getcwd(),'det_vocalize.mp3'))
    tts.save('det_vocalize.mp3')
    playsound('./det_vocalize.mp3')
    
    print("Vocalizing Predictions")
    #print("\n".join(print_bbxs))
    

#schedule.every(30).seconds.do(print_preds)
keyboard.on_press_key("i", lambda _:print_preds())

if __name__ == "__main__":
    # Configure depth and color streams
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # last parameter is fps
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Configure and start the pipeline
    profile = pipe.start(config)

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

            results = model(color_image)
            boxes = results.pandas().xyxy[0].values
            dectshow(color_image, boxes, depth_image, profile)

            #schedule.run_pending()

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





