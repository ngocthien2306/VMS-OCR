import time

from module_detect.base_model import LogicConfig
from system.utils import *
from system.event_handler import EventHandler, EventHandlerBase
from system.plc_controller import PLCController, PLCControllerBase
import cv2
import threading
from system.utils import SERVER_BE_IP, CLASSES
from system.sort import *

class LogicHandler:
    def __init__(self, config: LogicConfig, points, camera_id, lp_detector) -> None:
        self._config = config
        if self._config.event_handler_config is not None:
            self._event_handler = EventHandler(self._config.event_handler_config)
        else:
            self._event_handler = EventHandlerBase(self._config.event_handler_config)
        if self._config.plc_controller_config is not None:
            self._plc_controller = PLCController(self._config.plc_controller_config)
        else:
            self._plc_controller = PLCControllerBase(self._config.plc_controller_config)

        self.lp_detector = lp_detector
        self.mot_tracker = Sort()
        self._queue = []
        self._number_true_frame = 7
        self._last_event_timestamp = 0
        self._last_handle_wrong = True

        self._points = points
        self.colors = {}
        self._camera_id = camera_id
        self._start_time = time.time()
        self._end_time = time.time()
        self._count_frame = 0
        self._is_clicked_alarm = False
        self.is_start_record = False

        self._current_fps = 0
        
        # self._thread = threading.Thread(target=self.run_thread) 
        # self._thread.daemon = True
        # self._thread.start()

    def fps(self, f=1.0, show=False):
        self._end_time = time.time()
        if self._end_time - self._start_time >= f:
            if show:
                print(f"{self._camera_id}: {self._count_frame} fps")
            self._start_time = self._end_time
            self._current_fps = self._count_frame
            self._count_frame = 0
            
        self._count_frame += 1 
            
    def show_state_record(self):
        self._end_time = time.time()
        if self._end_time - self._start_time >= 1:
            self._start_time = self._end_time
            
        print(f"{self._camera_id}: {self.is_start_record}")

    def _process_frame(self, frame, results):
        inside_yn = False

        frame1 = cv2.resize(frame, (1280, 720))
        frame_plot = frame.copy()
        detections_ = []
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, score, _ = detection
            detections_.append([x1, y1, x2, y2, score])
            
        track_ids = mot_tracker.update(np.asarray(detections_))
        license_plates = self.lp_detector.predict(frame, verbose=False)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
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
                    data = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
                    
                    print(data)
                    print("--------------------------------------------")
                    
                    
        frame_plot = license_plates.plot()
        cv2.imshow("frame_plot", frame_plot)
        key = cv2.waitKey(1)
        if key == ord('q'):
            print()

        return inside_yn, frame_plot

    def update(self, frame, result):
        self._queue = [frame, result]
        
    def run_thread(self):
        while True:
            if len(self._queue) > 0:
                frame, result = self._queue
                self.run(frame, result)
                
            time.sleep(0.01)
        
    def run(self, frame, result):
        is_wrong, frame_plot = self._process_frame(frame, result)
        
        if is_wrong:
            self._event_handler.update(frame, frame_plot)
            
        self._event_handler.post_frame(frame_plot)

        return is_wrong, frame_plot

    def count_frame(self):
        self._count_frame += 1
        
    

    