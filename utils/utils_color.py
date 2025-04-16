import imghdr
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import random


class Predictor:
    def __init__(self, engine_path, imgsz=(640,640)):
        self.imgsz = imgsz
        self.show_color_flag = True
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
    

    def inference(self, img_path, depth_data, conf=0.5):
        f = open("detection.txt", "w")
        origin_img = img_path 
        (H, W) = origin_img.shape[:2]
        img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std)
        hsv_image = cv2.cvtColor(img.transpose((1,2,0)), cv2.COLOR_BGR2HSV)
        data = self.infer(img)
        predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
        dets = self.postprocess(predictions,ratio)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,
                                                             :4], dets[:, 4], dets[:, 5]
            obj_dist = []
            texts = []
            for box in final_boxes:
                mid_pos = get_mid_pos(box, depth_data, 24)
                obj_dist.append(mid_pos)
            boxes_updated, scores_updated, cls_inds_updated, obj_distances_updated, colors_updated = [],[], [], [], []
            for i in range(len(final_boxes)):
                box = final_boxes[i]
                dist = obj_dist[i]
                cls_id = int(final_cls_inds[i])
                score = final_scores[i]
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
                #colors_updated = det_cropped_bboxes_colors(hsv_image,boxes_updated,cls_inds_updated)
                centerX, centerY = (x0, y0)
                if centerX <= W/3:
                    W_pos = "left "
                elif centerX <= (W/3 * 2):
                    W_pos = "center "
                else:
                    W_pos = "right "
                if centerY <= H/3:
                    H_pos = "top "
                elif centerY <= (H/3 * 2):
                    H_pos = "mid "
                else:
                    H_pos = "bottom "
                texts.append(H_pos + W_pos + str(self.class_names[cls_id]))
            if texts:
                description = ', '.join(texts)
            f.write(description)
            f.close()
            colors_updated = det_cropped_bboxes_colors(hsv_image,boxes_updated,cls_inds_updated)

           
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds, obj_dist,colors_updated, texts,
                             conf=conf, class_names=self.class_names) 
        return origin_img


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


color_thresholds_hue = {
    'Red': [0,30],
    'Orange': [30,44],
    'Yellow': [45,90],
    'Green': [91,149],
    'Cyan': [150,224],
    'Blue': [225, 269],
    'Violet': [270,300],
    'Magenta': [301,36] }

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

        colors_array.append(cname)
    return colors_array

def get_mid_pos(box, depth_data, randnum):
        distance_list = []
        # returned bounding box format: [xmin, ymin, xmax, ymax, confidence, class, name]
        mid_pos = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2] # determine the center pixel positions of the index depth
        min_val = min(abs(box[2] - box[0]), abs(box[3]-box[1])) # determine depth search range 

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



def vis(img, boxes, scores, cls_ids, obj_dist, colors, locations ,conf=0.5, class_names=None):
    #print(colors)
    for i in range(len(boxes)):
        if i < len(colors) and i < len(locations):
            dist = obj_dist[i]
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            #print(locations[i])
            object_color = list(filter(None, colors))
            #print(colors)
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            if cls_id == 0 or cls_id == 80: # do not define color for people and faces
                text = '{}:{:.1f}% - {:.2f}, {}'.format(class_names[cls_id], score * 100, dist/1000, locations[i])
            else:
                text = '{}:{:.1f}% - {:.2f}, {}, {}'.format(class_names[cls_id], score * 100, dist/1000, str(object_color[1]), locations[i])
                print(text)
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
