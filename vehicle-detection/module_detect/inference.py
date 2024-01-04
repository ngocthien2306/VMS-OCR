import traceback
import os
import sys
from system.utils import get_computer_name, get_ipv4_address, get_camera_data
from system.event_handler import EventHandlerConfig
from system.frame_reader import FrameReader
from system.plc_controller import PLCControllerConfig
from module_detect.base_model import LogicConfig
from module_detect.logic_handler import LogicHandler
import time
from system.utils import get_polygon_points
import cv2
from system.socket import sokect_server
import threading
import eventlet
from system.utils import SERVER_BE_IP, CLASSES
from ultralytics import YOLO


MODULE_ID = "vms"
VEHICLE_DETECTOR = YOLO("../models/yolov8m.pt", task='detect')
LP_DETECTOR =  YOLO("../models/lp_best_s.pt", task='detect')

def initialize_logic_handler(camera_ids, polygons):
    dict_points = get_polygon_points()
    server_name = get_computer_name()
    logic_handlers = {}
    for camera_id in camera_ids:
        print(f"http://{get_ipv4_address()}:8005/stream-manage/output/{MODULE_ID}-{camera_id}")
        event_handler_config = EventHandlerConfig(
            post_frame_url=f"http://{get_ipv4_address()}:8005/stream-manage/output/{MODULE_ID}-{camera_id}",
            post_event_url=f"http://{SERVER_BE_IP}:8080/event",
            camera_id=camera_id,
            module_id=MODULE_ID,
            msgType=2,
            frame_stream_size=None,
            frame_log_size=(1280, 720),
            frame_org_size=(1280, 720)
        )
        plc_controller_config = PLCControllerConfig(
            plc_ip_address="192.168.2.150",
            plc_port=502,
            plc_address=1,
            modbus_address=8196
        )
        logic_config = LogicConfig(
            event_handler_config=event_handler_config,
            # plc_controller_config=plc_controller_config
        )
        logic_handlers[camera_id] = LogicHandler(config=logic_config, points=dict_points[camera_id], camera_id=camera_id, lp_detector=LP_DETECTOR)
    
    return logic_handlers
        
def initialize_socket(logic_handlers):
    server_instance = sokect_server.SocketIOServer(logic_handlers)
 
    def run_server():
        server_instance.run()

    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    eventlet.sleep(1)
    
def object_detection(frames_dict):
    predictions_vehicle = VEHICLE_DETECTOR.predict(list(frames_dict.values()), verbose=False, classes=CLASSES, conf=0.3)
    predictions_lp = LP_DETECTOR.predict(list(frames_dict.values()), verbose=False, conf=0.3)
    results_vehicle = {}
    results_lp = {}
    
    if len(list(frames_dict.keys())) > 0:
        for predict, camera_id in zip(predictions_vehicle, list(frames_dict.keys())):
            results_vehicle[camera_id] = predict
            
        for predict, camera_id in zip(predictions_lp, list(frames_dict.keys())):
            results_lp[camera_id] = predict
                   
            
    return results_vehicle, results_lp



def inference_module(logic_handlers, frame_reader):
    while True:
        try:
            frames_dict = frame_reader.get_last_frames()
            results_vehicle, results_lp = object_detection(frames_dict.copy())
            
            for frame_dict in frames_dict.items():
                key, value = frame_dict

                logic_handlers[key].run(value, results_vehicle[key], results_lp[key])
                logic_handlers[key].fps(show=True)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        except Exception as e:
            traceback.print_exc()
            continue

def main():

    camera_ids, polygons = get_camera_data()
    frame_reader = FrameReader(camera_ids)
    logic_handlers = initialize_logic_handler(camera_ids, polygons)
    # initialize_socket(logic_handlers)
    inference_module(logic_handlers, frame_reader)

        
    cv2.destroyAllWindows()
    
        