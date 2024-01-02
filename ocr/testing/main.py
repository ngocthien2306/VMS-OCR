from ultralytics import YOLO
import cv2
import os
import torch
print(os.getcwd())
import numpy as np
from util import *


results = {}

mot_tracker = Sort()
device = torch.device("cuda:0")
# load models
coco_model = YOLO('data/models/yolov8n.pt', task='detect')
license_plate_detector = YOLO('data/models/license_plate_detector.pt', task='detect')

# load video
cap = cv2.VideoCapture('data/external/videos/sample.mp4')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    
    if ret:
        frame = cv2.resize(frame, (1280, 720))
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model.predict(frame, verbose=False)[0]
        # print(detections)
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector.predict(frame, verbose=False)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
                    print("--------------------------------------------")
                    print(results[frame_nmr][car_id])
                    
        frame_output_vehile = detections.plot()
        frame_output_lp = license_plates.plot()
        imgs = (1280, 720)
        frame_output_vehile = cv2.resize(frame_output_vehile, imgs)
        frame_output_lp = cv2.resize(frame_output_lp, imgs)
        
        cv2.imshow('Vehicle Detection', frame_output_vehile)
        cv2.imshow('License Plate Recognition', frame_output_lp)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()
# write results
write_csv(results, './test.csv')