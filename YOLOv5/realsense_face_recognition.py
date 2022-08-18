# NEED to install library in conda environment first
import pyrealsense2 as rs
import face_recognition
import numpy as np
import pickle
import cv2
import random
import torch
import time
import os

# Import your model from local machine
#model = torch.hub.load('ultralytics/yolov5', 'yolov5n') 
# MODIFY THIS PATH TO POINT TO YOUR MODEL
model = torch.hub.load('/home/askat/Research/visual_assistant/yolov5', 'custom', path='/home/askat/Research/visual_assistant/yolov5/best-yolov5nano.pt', source='local')  # local repo
model.conf = 0.3


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

def dectshow(org_img, boxes, depth_data, profile, embeddings):
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    img = org_img.copy()
    for box in boxes:
        dist = get_mid_pos(org_img, box, depth_data, 36)
        if box[-1] == 'face':

            # pass the face trough face recognition model
            encoding = face_recognition.face_encodings(img, [(int(box[1]), int(box[2]), int(box[3]), int(box[0]))], num_jitters=3)[0]

            # attempt to match each face in the input image to our 
            # known encodings
            matches = face_recognition.compare_faces(embeddings["encodings"],
                        encoding, tolerance=0.45)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a 
                # dictionary to count the total number of times each face
                # was matched 
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = embeddings["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of 
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)

                # include the probability in the label
                name = str(name)

            cv2.putText(img, name + ' ' + str(dist * depth_scale)[:4] + 'm',(int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),1)
        else:
            cv2.putText(img, box[-1] + ' ' + str(dist * depth_scale)[:4] + 'm',(int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),1)

        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 2)

    cv2.imshow('Output', img)
    

if __name__ == "__main__":
    # load the known faces and encodings
    print("[INFO] loading facial encodings...")
    embeddings = pickle.loads(open("embeddings/speakingfaces_embeddings.pickle", "rb").read())

    print("[INFO] configuring the RealSense camera...")
    pipeline = rs.pipeline()
    cam_config = rs.config()
    cam_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cam_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(cam_config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)


    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align = rs.align(rs.stream.color)

    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() 
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            results = model(color_image)
            boxes = results.pandas().xyxy[0].values
            dectshow(color_image, boxes, depth_image, profile, embeddings)


            key = cv2.waitKey(1)
            # Press esc or 'q' to close image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()





