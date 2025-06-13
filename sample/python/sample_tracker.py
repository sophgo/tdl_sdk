import sys
import os
from tdl import nn,image,tracker
import re
import cv2
import numpy as np

def draw_bboxes(img, trackers):
    color_map = {}
    for tracker_info in trackers:
        box_info = tracker_info["box_info"]
        status = tracker_info["status"]
        obj_idx = tracker_info["obj_idx"]
        track_id = tracker_info["track_id"]
        velocity_x = tracker_info["velocity_x"]
        velocity_y = tracker_info["velocity_y"]
        x1 = int(box_info["x1"])
        y1 = int(box_info["y1"])
        x2 = int(box_info["x2"])
        y2 = int(box_info["y2"])
        class_id = box_info["class_id"]
        #class_name = box_info["object_type"]
        if track_id not in color_map:
            np.random.seed(track_id)
            color = tuple(np.random.randint(0, 256, 3).tolist())
        color_map[track_id] = color
        color = color_map[track_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"ID:{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        #print(status, obj_idx, track_id)
    return img

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python sample_tracker.py <obj_model_path> <image_dir_path>")
        #print("Usage: python sample_tracker.py <obj_model_path> <face_model_path> <image_dir_path>")
        sys.exit(1)
    obj_model_path = sys.argv[1]
    #face_model_path = sys.argv[2]
    image_dir_path = sys.argv[2]
    files = os.listdir(image_dir_path)
    files = sorted(files)
    num_frames = len(files)
    obj_detector = nn.get_model(nn.ModelType.YOLOV8N_DET_PERSON_VEHICLE, obj_model_path)
    #face_detector = nn.FaceDetector(nn.ModelType.SCRFD_DET_FACE, face_model_path)
    tracker_type = tracker.TrackerType.TDL_MOT_SORT
    tracker = tracker.Tracker(tracker_type)
    #tracker.setImgSize(width, height)
    #track_config = tracker.get_track_config()
    #track_config.max_unmatched_times = 10  
    #track_config.track_confirmed_frames = 2  
    #track_config.high_score_thresh = 0.5  
    #track_config.high_score_iou_dist_thresh = 0.7
    #track_config.low_score_iou_dist_thresh = 0.5  
    #tracker.set_track_config(track_config)
    for i in range(num_frames):
        img_path = os.path.join(image_dir_path, files[i])
        img = image.read(img_path)
        obj_bboxes = obj_detector.inference(img)
        #face_bboxes = face_detector.inference(img)
        #bboxes = []
        #for bbox_type in [obj_bboxes, face_bboxes]:
            #for bbox in bbox_type:
                #merged_info = {
                    #'class_id':bbox['class_id'],
                    #'class_name': bbox['class_name'],
                    #'x1': bbox['x1'],
                    #'y1': bbox['y1'],
                    #'x2': bbox['x2'],
                    #'y2': bbox['y2'],
                    #'score':bbox['score']
                #}
                #bboxes.append(merged_info)
        #print(len(obj_bboxes))    
        print(f"Frame ID: {i}")
        trackers = tracker.track(obj_bboxes, i)
        img_arr = cv2.imread(img_path)
        result_arr = draw_bboxes(img_arr, trackers)
        result_img = image.Image()
        output = result_img.from_numpy(result_arr)
        image.write(output, f"{image_dir_path}/output_{i}.jpg")
