from cmath import cos
from email import header
from statistics import mean
from tkinter.tix import DirTree
from turtle import color
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import random
from time import sleep
from pygame import mixer
import pygame
import pandas as pd
import math
from scipy.stats import circmean
import norfair
from norfair import Detection, Tracker, Paths
from typing import List, Optional, Sequence, Tuple
import math 

DISTANCE_THRESHOLD_BBOX: float = 0.5
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000
track_points = 'bbox'
near =0


class Rect:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def width(self):
        return self.x1 - self.x0

    def height(self):
        return self.y1 - self.y0

    def area(self):
        if self.empty():
            return 0 
        else:
            return self.width() * self.height()

    def intersect(self, other):
        return Rect(
            max(self.x0, other.x0), max(self.y0, other.y0),
            min(self.x1, other.x1), min(self.y1, other.y1)
        )

    def intersects(self, other):
        return not self.intersect(other).empty()

    def empty(self):
        return self.width() <= 0 or self.height() <= 0

    def split_width(self, percentages):
        x = self.x0 
        rects = []
        for p in percentages:
            x1 = x + self.width() * p
            rects.append(Rect(x, self.y0, x1, self.y1))
            x = x1 
        return rects

    def __repr__(self):
        return (f"Rect(x0: {self.x0}, y0: {self.y0}, "
                f"x1: {self.x1}, y1: {self.y1})")

class Predictor:
    def __init__(self, engine_path, imgsz=(640,640)):
        distance_function = iou if track_points == 'bbox' else euclidean_distance
        distance_threshold = (
            DISTANCE_THRESHOLD_BBOX
            if track_points == 'bbox'
            else DISTANCE_THRESHOLD_BBOX
        )
        self.tracker = Tracker(distance_function=distance_function,
                                distance_threshold= distance_threshold)
        self.imgsz = imgsz
        self.show_color_flag = False
        self.mean = None
        self.std = None
        self.n_classes = 81
        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush', 'face']

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})



    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        return data
    
    def detect_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            blob, ratio = preproc(frame, self.imgsz, self.mean, self.std)
            data = self.infer(blob)
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            dets = self.postprocess(predictions,ratio)
            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:,
                                                                :4], dets[:, 4], dets[:, 5]
                frame = vis(frame, final_boxes, final_scores, final_cls_inds,
                                conf=0.5, class_names=self.class_names)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    
   
    def inference(self, img_path, depth_data, class_to_track, conf=0.5):
        init_sound()
        global near
        W_pos=''
        H_pos=''
       
        threat_volume = [0,0]
        origin_img = img_path #cv2.imread(img_path)
        origin_img = cv2.rotate(origin_img, cv2.ROTATE_180) 
        (H, W) = origin_img.shape[:2]
        #print(H,W)
        #H//6:5*H//6,W//6:5*W//8,
        dh = 6
        dw = 6
        origin_img =  origin_img[H//dh:H-H//dh,W//dw:W-W//dw:]
        origin_img = cv2.resize(origin_img, (W,H))
        depth_data = depth_data[H//dh:H-H//dh,W//dw:W-W//dw]
        depth_data = cv2.resize(depth_data, (W,H))
        (H, W) = origin_img.shape[:2]
        image_rect = Rect(0, 0, W, H)
        parts = ["Left", "Center", "Right"]
        part_rects = image_rect.split_width([0.25, 0.5, 0.25])
        img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std)
        data = self.infer(img)
        predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
        dets = self.postprocess(predictions,ratio)
        #print(origin_img.shape)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,
                                                             :4], dets[:, 4], dets[:, 5]
            obj_dist = []
            texts = []
            for box in final_boxes:
                mid_pos = get_mid_pos(box, depth_data, 24)
                obj_dist.append(mid_pos)
            boxes_updated, scores_updated, cls_inds_updated,obj_distances_updated, colors_updated = [],[], [], [],[]
            for i in range(len(final_boxes)):
                
                #dist = obj_dist[i]
                box = final_boxes[i]
                dist = obj_dist[i]
                cls_id = int(final_cls_inds[i])
                score = final_scores[i]
                if str(self.class_names[cls_id]) == class_to_track :
                    if score < conf:
                        continue
                    x0 = int(box[0])
                    y0 = int(box[1])
                    x1 = int(box[2])
                    y1 = int(box[3])
                    boxes_updated.append([x0,y0,x1,y1])
                    scores_updated.append(score)
                    cls_inds_updated.append(cls_id)
                    obj_distances_updated.append(dist)
                    (best_region, intersection_area) = find_best_region(parts, 
                                                    part_rects, 
                                                    Rect(x0,y0,x1,y1))
                    print(f"Best region '{best_region}' with intersection area {intersection_area}")
                    #centerX, centerY = (x0, y0)
                    centerX = (x0 + x1)/2
                    centerY = (y0 + y1)/2
                    print(dist)
                    volume = getVolumeAtDistance(dist)
                    if dist <= 5500 and dist >= 4000 and getChannelBusy(0) == False:                           
                        if  centerX <= W/3:
                            W_pos = " left "
                            threat_volume[0] = volume
                            threat_volume[1] = 0.001
                            print(threat_volume[0], threat_volume[1])
                            setChannelVolume(0, threat_volume[0], threat_volume[1])
                            #left_objects.append(class_id)       
                        elif centerX <= (W/3 * 2):
                            W_pos = " center "
                            threat_volume[0] = volume
                            threat_volume[1] = volume
                            print(threat_volume[0], threat_volume[1])
                            setChannelVolume(0, threat_volume[0], threat_volume[1])
                            #midlle_objects.append(class_id)
                        else:
                            W_pos = " right "
                            threat_volume[1] = volume
                            threat_volume[0] = 0.001
                            print(threat_volume[0], threat_volume[1])
                            setChannelVolume(0, threat_volume[0], threat_volume[1])
                        if centerY <= H/3:
                            H_pos = "top"
                            playV(1500)
                        elif centerY <= (H/3 * 2):
                            H_pos = "middle"
                            playV(1000)
                        else:
                            H_pos = "bottom"
                            playV(1500)                                 
                    elif dist < 4000 and dist >= 2500 and getChannelBusy(0) == False:

                        if  centerX <= W/3:
                            W_pos = " left "
                            threat_volume[0] = volume
                            threat_volume[1] = 0.001
                            print(threat_volume[0], threat_volume[1])
                            setChannelVolume(0, threat_volume[0], threat_volume[1])
                            #left_objects.append(class_id)       
                        elif centerX <= (W/3 * 2):
                            W_pos = " center "
                            threat_volume[0] = volume
                            threat_volume[1] = volume
                            print(threat_volume[0], threat_volume[1])
                            setChannelVolume(0, threat_volume[0], threat_volume[1])
                            #midlle_objects.append(class_id)
                        else:
                            W_pos = " right "
                            threat_volume[1] = volume
                            threat_volume[0] = 0.001
                            print(threat_volume[0], threat_volume[1])
                            setChannelVolume(0, threat_volume[0], threat_volume[1])
                        if centerY <= H/3:
                            H_pos = "top"
                            playV(1000)
                        elif centerY <= (H/3 * 2):
                            H_pos = "middle"
                            playV(500)
                        else:
                            H_pos = "bottom"
                            playV(1000)
                    elif dist < 2500 and dist >= 1000 and getChannelBusy(0) == False:
                        #playSfx("2", 2)
                        near=0
                        if  centerX <= W/4:
                            W_pos = " left "
                            threat_volume[0] = volume
                            threat_volume[1] = 0.001
                            print(threat_volume[0], threat_volume[1])
                            setChannelVolume(0, threat_volume[0], threat_volume[1])
                            #left_objects.append(class_id)       
                        elif centerX <= (W/4 * 3 ) :
                            W_pos = " center "
                            threat_volume[0] = volume
                            threat_volume[1] = volume
                            print(threat_volume[0], threat_volume[1])
                            setChannelVolume(0, threat_volume[0], threat_volume[1])
                            #midlle_objects.append(class_id)
                        else:
                            W_pos = " right "
                            threat_volume[1] = volume
                            threat_volume[0] = 0.001
                            print(threat_volume[0], threat_volume[1])
                            setChannelVolume(0, threat_volume[0], threat_volume[1]) 
                        if centerY <= H/4:
                            H_pos = "top"
                            playV(500)
                        elif centerY <= (H/4 * 3):
                            H_pos = "middle"
                            playV(250)
                        else:
                            H_pos = "bottom"
                            playV(500)
                    elif dist < 1000 and getChannelBusy(0) == False:
                        if  centerX <= W/4:
                            W_pos = " left "
                            threat_volume[0] = volume
                            threat_volume[1] = 0.001
                            print(threat_volume[0], threat_volume[1])
                            setChannelVolume(0, threat_volume[0], threat_volume[1])
                            #left_objects.append(class_id)       
                        elif centerX <= (W/4 * 3):
                            W_pos = " center "
                            threat_volume[0] = volume
                            threat_volume[1] = volume
                            print(threat_volume[0], threat_volume[1])
                            # if getChannelBusy(2) == False:
                            #     if near == 0:
                            #         playBGM('aldynyzda')
                            #         near= 1
                            setChannelVolume(0, threat_volume[0], threat_volume[1])                    
                        else:
                            W_pos = " right "
                            threat_volume[1] = volume
                            threat_volume[0] = 0.001
                            print(threat_volume[0], threat_volume[1])
                            setChannelVolume(0, threat_volume[0], threat_volume[1])
                        if centerY <= H/4:
                            H_pos = "top"
                            playV(250)
                        elif centerY <= (H/4 * 3):
                            H_pos = "middle"
                            playV(100)
                        else:
                            H_pos = "bottom"
                            playV(250)                    
                        # if centerX <= (W/5*2):
                        #     W_pos = " left "
                        #     threat_volume[0] = volume
                        #     threat_volume[1] = 0.001
                        #     print(threat_volume[0], threat_volume[1])
                        #     setChannelVolume(0, threat_volume[0], threat_volume[1])
                        #     #left_objects.append(class_id)       
                        # elif centerX >= (W/5 * 3):
                        #     W_pos = " right "
                        #     threat_volume[1] = volume
                        #     threat_volume[0] = 0.00
                        #     print(threat_volume[0], threat_volume[1])
                        #     setChannelVolume(0, threat_volume[0], threat_volume[1])
                        #     #midlle_objects.append(class_id)
                        # else:
                        #     W_pos = " center "
                        #     threat_volume[0] = volume
                        #     threat_volume[1] = volume
                        #     print(threat_volume[0], threat_volume[1])
                        #     setChannelVolume(0, threat_volume[0], threat_volume[1])
                       
                            #right_objets.append(class_id)
                        # if centerY <= H/3:
                        #     H_pos = "top"
                        #     playSfx("1", 2)
                        #     print(threat_volume[0], threat_volume[1])
                        #     setChannelVolume(2, threat_volume[0], threat_volume[1])
                        # elif centerY <= (H/3 * 2):
                        #     H_pos = "middle"
                        #     playSfx("2", 2)
                        #     print(threat_volume[0], threat_volume[1])
                        #     setChannelVolume(2, threat_volume[0], threat_volume[1])
                        # else:
                        #     H_pos = "bottom"
                        #     playSfx("3", 2)
                        #     print(threat_volume[0], threat_volume[1])
                        #     setChannelVolume(2, threat_volume[0], threat_volume[1])
                    #print(H_pos + W_pos + self.class_names[cls_id] + str(cls_id))
                
                    texts.append(H_pos + W_pos + str(self.class_names[cls_id]))
                    #texts.append(W_pos + str(self.class_names[cls_id]))
                if texts:
                    description = ', '.join(texts)
                    print(description)
        #if dets is not None:
            #if self.show_color_flag:
                #colors_updated = det_cropped_bboxes_colors(hsv_image,boxes_updated,cls_inds_updated)
                #detections = convert_detections_to_norfair_detections(boxes_updated, scores_updated, cls_inds_updated, obj_distances_updated, colors_updated)
            #else:
            detections = convert_detections_to_norfair_detections(boxes_updated, scores_updated, cls_inds_updated, obj_distances_updated, [])
            tracked_objects = self.tracker.update(detections)
            
            if track_points == "bbox":
                draw_boxes(origin_img, detections, color_by_label= True, draw_labels= True)
                draw_tracked_objects(origin_img, tracked_objects, color_by_label = True)


            # origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds, obj_dist,
            #                  conf=conf, class_names=self.class_names)
        return origin_img
        #cv2.imshow('detections',frame)
        #return frame

    @staticmethod
    def postprocess(predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        return dets
    
    def get_fps(self):
        # warmup
        import time
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(20):
            _ = self.infer(img)
        t1 = time.perf_counter()
        _ = self.infer(img)
        print(1/(time.perf_counter() - t1), 'FPS')

def find_best_region(regions, region_rects, bounding_box):
    max_intersection = 0
    max_region = ""
    for (region, region_rect) in zip(regions, region_rects):
        intersection = region_rect.intersect(bounding_box).area()
        if max_intersection < intersection:
            max_intersection = intersection 
            max_region = region
    return (max_region, max_intersection)


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

    
colorsList = {'Red': [255, 0, 0],
              'Green': [0, 128, 0],
              'Blue': [0, 0, 255],
              'Yellow': [255, 255, 0],
              'Violet': [238, 130, 238],
              'Orange': [255, 165, 0],
              'Black': [0, 0, 0],
              'White': [255, 255, 255],
              'Pink': [255, 192, 203],
              'Brown': [165, 42, 42]}
color_thresholds_hue = {
    'Red': [0,30],
    'Orange': [30,44],
    'Yellow': [45,90],
    'Green': [91,149],
    'Cyan': [150,224],
    'Blue': [225, 269],
    'Violet': [270,300],
    'Magenta': [301,360]
}
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

def det_cropped_bboxes_colors(image, bboxes, cl_inds):
    #print(image.shape)
    colors_array = []
    for bbox,cl_ind in zip(bboxes, cl_inds):
        if cl_ind == 0 or cl_ind == 80: # do not define color for people and faces
            cname = ' '
        else: 
            bbox = [0 if single__point < 0 else single__point for single__point in bbox ] # for some reason some bounding boxes have slightlu negative values (eg -1,-2)
            #print(bbox)
            crop_image = image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
            #print(crop_image.shape)
            h,s,v = cv2.split(crop_image)
            
            x = np.cos(np.array(h) * np.pi/180.)
            y = np.sin(np.array(h) * np.pi/180.)
            mean_h = (np.arctan2(np.mean(y), np.mean(x)))*180/np.pi + 180
            mean_s = np.mean(s)
            mean_v = np.mean(v)
            cname = 'undefined'
            if mean_s < 0.2 and mean_v < 0.6:
                cname = 'Grey'
            if mean_s < 0.2 and mean_v < .4:
                cname = 'Black'
            else:
                for color in color_thresholds_hue.keys():
                
                    if color_thresholds_hue[color][0]<mean_h<color_thresholds_hue[color][1]:
                        cname = color
            
            #print(color_thresholds_hue.values())
            print(f'Label {class_names[cl_ind]}')
            print(f'Color name is {cname}')
            print(f'Mean hue {mean_h}')
            
            print(f'S channel mean  {np.mean(s)}')
            print(f'V channel mean {np.mean(v)}')
            print('NEXT IMAGE')
            # cv2.imshow('Cropped Image', crop_image)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()
        colors_array.append(cname)
    return colors_array

def get_middle_points(image, box, input_size):
    # if len(image.shape) == 3:
    #     padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    # else:
    #     padded_img = np.ones(input_size) * 114.0
    # img = np.array(image)
    # r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    # resized_img = cv2.resize(
    #     img,
    #     (int(img.shape[1] * r), int(img.shape[0] * r)),
    #     interpolation=cv2.INTER_LINEAR,
    # ).astype(np.float32)
    # padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # img = padded_img
    crop_image = image[int(box[0]):int(box[3]),int(box[1]):int(box[2])]
    print(f'Cropped image shape {crop_image.shape}')
    #img = image.transpose((1,2,0))
    cv2.imshow('cropped', crop_image)
    #print(img)
    mid_pos = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2] # determine the center pixel positions of the index depth
    min_val = min(abs(box[2] - box[0]), abs(box[3]-box[1])) # determine depth search range
    

    minimum_dist = 1000
    b,g,r = 0,0,0
    for _ in range(4):
        bias = random.randint(-min_val//6, min_val//6)
        #print(f'Bias is {bias}')
        biased_mid_pos = [int(mid_pos[0] + bias), int(mid_pos[1] + bias)]
        #print(f'Biased mid point {biased_mid_pos}')
        # mid_points.append([biased_mid_pos])
        #print(f'B G R is {img[biased_mid_pos[1],biased_mid_pos[0]]}')
    
    # for mid_point in mid_points:
        #print(mid_point)
        # bgr.append([img[mid_point[1],mid_point[0]]])
        bgr_arr = image[biased_mid_pos[1],biased_mid_pos[0]]
        b+= bgr_arr[0]
        g+= bgr_arr[1]
        r+= bgr_arr[2]
    #print(bgr)
    b_mean = int(b//5)
    g_mean = int(g//5)
    r_mean = int(r//5)
    # print(f'B mean is {b_mean}')
    # print(f'G mean is {g_mean}')
    # print(f'R mean is {r_mean}')
    # print('NEXT BBOX')
    # for i in range(len(colors_csv)):
    #     d = abs(r_mean- int(colors_csv.loc[i,"R"])) + abs(g_mean- int(colors_csv.loc[i,"G"]))+ abs(b_mean- int(colors_csv.loc[i,"B"]))
    #     if(d <= minimum_dist):
    #         minimum_dist = d
    #         cname = colors_csv.loc[i,"color_name"]
    # return cname
    rgb = [r_mean, g_mean, b_mean]
    for color in colorsList:
        dist = np.linalg.norm(np.array(rgb) - np.array(colorsList[color]))
        if dist <= minimum_dist:
            name = color
            minimum_dist = dist
    return name


        

def get_mid_pos(box, depth_data, randnum):
        mid_points = []
        distance_list = []
        # returned bounding box format: [xmin, ymin, xmax, ymax, confidence, class, name]
        mid_pos = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2] # determine the center pixel positions of the index depth
        min_val = min(abs(box[2] - box[0]), abs(box[3]-box[1])) # determine depth search range 
        #print(f'Mid point {mid_pos}')
        mid_points.append(mid_pos)
        for i in range(randnum):
            bias = random.randint(-min_val//4, min_val//4)
            dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)] # INDICES ARE SWITCHED IN ORIGINAL
            #cv2.circle(frame, (int(mid_pos[0]+ bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)

            if dist:
                distance_list.append(dist)
        distance_list = np.array(distance_list)
            # Median Filter in case any points will give irrelevant values (too small, too big)
        distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4]

        return np.mean(distance_list)

def vis(img, boxes, scores, cls_ids, obj_dist, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        dist = obj_dist[i]
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}% - {:.2f}'.format(class_names[cls_id], score * 100, dist/1000)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img
def convert_detections_to_norfair_detections(bboxes, conf_scores, cls_inds, obj_distances, colors):
    norfair_detections: List[Detection] = []
    if track_points == 'bbox':
        if colors:
            for box,score, cl_ind, dist,col in zip(bboxes,conf_scores, cls_inds, obj_distances,colors):
                bbox =  np.array(
                    [
                        [box[0], box[1]],
                        [box[2], box[3]]
                    ]
                )
                scores = np.array([score, score])
                norfair_detections.append(
                    Detection(points = bbox, scores = scores, label = cl_ind, data = [dist,col])
                )
        else: 
            for box,score, cl_ind, dist in zip(bboxes,conf_scores, cls_inds, obj_distances):
                bbox =  np.array(
                    [
                        [box[0], box[1]],
                        [box[2], box[3]]
                    ]
                )
                scores = np.array([score, score])
                norfair_detections.append(
                    Detection(points = bbox, scores = scores, label = cl_ind, data = dist)
                )
    return norfair_detections

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def center(points):
    return [np.mean(np.array(points), axis=0)]

def iou(detection, tracked_object):
    # Detection points will be box A
    # Tracked objects point will be box B.

    box_a = np.concatenate([detection.points[0], detection.points[1]])
    box_b = np.concatenate([tracked_object.estimate[0], tracked_object.estimate[1]])

    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # Compute the area of both the prediction and tracker
    # rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + tracker
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    # Since 0 <= IoU <= 1, we define 1/IoU as a distance.
    # Distance values will be in [1, inf)
    return 1 / iou if iou else (MAX_DISTANCE)
class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush', 'face']
def draw_boxes(
    frame,
    detections,
    line_color=None,
    line_width=1,
    random_color=False,
    color_by_label=True,
    draw_labels=False,
    label_size=0.5,
):
    if detections is None:
        return frame
    if line_color is None:
        line_color = (0, 0, 0)
    for d in detections:
        txt_color = (255, 255, 255) #(0, 0, 0) if np.mean(_COLORS[d.label]) > 0.5 else (255, 255, 255)
        if color_by_label:
            #line_color = Color.random(abs(hash(d.label)))
            line_color = (_COLORS[d.label] * 255).astype(np.uint8).tolist()
        elif random_color:
            line_color = (0, 0, 0) if np.mean(_COLORS[d.label]) > 0.5 else (255, 255, 255)
        points = d.points
        points = validate_points(points)
        points = points.astype(int)
        cv2.rectangle(
            frame,
            tuple(points[0, :]),
            tuple(points[1, :]),
            color=line_color,
            thickness=line_width,
        )

        if draw_labels:
            if(hasattr(d.data, "__len__")):
                text_string = f"{class_names[d.label]} : {(np.around(d.scores[0]*100, decimals = 1))}% - {np.around((d.data[0] / 1000), decimals= 2)} m - {d.data[1]}"
            else:
                text_string =  f"{class_names[d.label]} : {(np.around(d.scores[0]*100, decimals = 1))}% - {np.around((d.data / 1000), decimals= 2)} m "
            label_draw_position = np.array(points[0, :])
            cv2.putText(
                frame,
                text_string,
                #f"{class_names[d.label]} : {(np.around(d.scores[0]*100, decimals = 1))}% - {np.around((d.data / 1000), decimals= 2)} m",
                tuple(label_draw_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                label_size,
                txt_color,
                line_width,
                #cv2.LINE_AA,
            )

    return frame

def draw_tracked_objects(
    frame: np.array,
    objects: Sequence["TrackedObject"],
    radius: Optional[int] = None,
    color: Optional[Tuple[int, int, int]] = None,
    id_size: Optional[float] = .5,
    id_thickness: Optional[int] = 1,
    draw_points: bool = True,
    color_by_label: bool = True,
    draw_labels: bool = False,
    label_size: Optional[int] = 2,
):
    frame_scale = frame.shape[0] / 100
    if radius is None:
        radius = int(frame_scale * 0.5)


    for obj in objects:
        if not obj.live_points.any():
            continue
        if color_by_label:
            point_color = (0, 0, 0) if np.mean(_COLORS[obj.label]) > 0.5 else (255, 255, 255)
            id_color = point_color
        if draw_points:
            for point, live in zip(obj.estimate, obj.live_points):
                if live:
                    cv2.circle(
                        frame,
                        tuple(point.astype(int)),
                        radius=radius,
                        color=point_color,
                        thickness=-1,
                    )

            if draw_labels:
                points = obj.estimate[obj.live_points]
                points = points.astype(int)
                label_draw_position = np.array([min(points[:, 0]), min(points[:, 1])])
                label_draw_position -= radius
                cv2.putText(
                    frame,
                    f"L: {obj.label}",
                    tuple(label_draw_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    label_size,
                    point_color,
                    id_thickness,
                    cv2.LINE_AA,
                )

        if id_size > 0:
            points = obj.estimate[obj.live_points]
            points = points.astype(int)
            id_draw_position = np.array([min(points[:, 0]), min(points[:, 1]) + 15])
            cv2.putText(
                frame,
                f'ID: {str(obj.id)}',
                id_draw_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                id_size,
                id_color,
                id_thickness,
                cv2.LINE_AA,
            )

def validate_points(points: np.array) -> np.array:
    # If the user is tracking only a single point, reformat it slightly.
    if points.shape == (2,):
        points = points[np.newaxis, ...]
    elif len(points.shape) == 1:
        print_detection_error_message_and_exit(points)
    else:
        if points.shape[1] != 2 or len(points.shape) > 2:
            print_detection_error_message_and_exit(points)
    return points


def print_detection_error_message_and_exit(points):
    print("\n[red]INPUT ERROR:[/red]")
    print(
        f"Each `Detection` object should have a property `points` of shape (num_of_points_to_track, 2), not {points.shape}. Check your `Detection` list creation code."
    )
    print("You can read the documentation for the `Detection` class here:")
    print("https://github.com/tryolabs/norfair/tree/master/docs#detection\n")
    exit()


def getVolumeAtDistance(dist):
    return max(0, 1-1/5500*dist)

def show_distance(event, x, y, args, params):
    global point
    point = (x, y)

def init_sound():
    #mixer.init(channels=2)
    pygame.mixer.init(frequency=44100, size=32, channels=2, buffer=512)
    mixer.set_num_channels(11)

def playSfx(track, channel=1, volume=1, loops=0):

    # failsafe
    if channel == 0:
        print("WARNING: Channel 0 is reserved for BGM!")

        # don't just quit the program due to a sound playing fail
        # try to find another available channel instead
        for i in range(1, 10):
            if not getChannelBusy(i):
                channel = i

        # no available channel to fallback?
        # well, we will have to interrupt one then...
        if channel == 0:
            channel = 1
    
    chn = mixer.Channel(channel)
    track_path = "data/askat/" + str(track) + ".wav"
    snd = mixer.Sound(track_path)
    chn.set_volume(volume)
    chn.play(snd, loops)

def playBGM(track):
    chn = mixer.Channel(2)
    track_path = "data/" + str(track) + ".wav"
    snd = mixer.Sound(track_path)
    #chn.set_volume(volume)
    chn.play(snd, 0)

# def playV(mytime):
#     sampleRate = 44100
#     frequency = 440
#     length = 5

#     t = np.linspace(0, length, sampleRate * length)  #  Produces a 5 second Audio-File
#     buffer = np.sin(frequency * 2 * np.pi * t).astype(np.float32)

#     chn = mixer.Channel(0)
#     #buffer = np.sin(2 * np.pi * np.arange(44100) * 300 / 44100).astype(np.float32)
#     snd = mixer.Sound(buffer)
#     chn.play(snd, loops=0, maxtime=mytime)
#     #pygame.time.wait(int(snd.get_length() * 1000))


def playV(mytime):
    chn = mixer.Channel(0)
    track_path = "data/askat/2.wav"
    snd = mixer.Sound(track_path)
    #chn.set_volume(volume)
    chn.play(snd, loops=0, maxtime=mytime)

def getChannelBusy(channel):
    try:
        chn = mixer.Channel(channel)
        return chn.get_busy()
    except IndexError:
        return False

def setChannelVolume(channel, volume_left, volume_right = None):
    chn = mixer.Channel(channel)

    if not volume_right:
        chn.set_volume(volume_left)
    else:
        # yeah, can also do panning!
        chn.set_volume(volume_left, volume_right)

def stopChannel(channel):
    try:
        chn = mixer.Channel(channel)
        chn.stop()
    except IndexError:
        pass