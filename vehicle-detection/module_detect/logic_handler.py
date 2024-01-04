import time

from module_detect.base_model import LogicConfig
from system.utils import *
from system.event_handler import EventHandler, EventHandlerBase
from system.plc_controller import PLCController, PLCControllerBase
import cv2
import threading
from system.image_utils import *
from system.sort import *
from module_ocr.paddleocr import ocr_model_init

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
            
        self.ocr_model, self.args = ocr_model_init()

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
        
    def sorting_key(self, item):
        return item[0][0][1]  # Assuming the y-coordinate is at index 1

    def _process_frame(self, frame, results_vehicle, results_lp):

        frame_plot = frame.copy()
        detections_ = []
        if results_vehicle is not None:
            for detection in results_vehicle.boxes.data.tolist():
                x1, y1, x2, y2, score, _ = detection
                detections_.append([x1, y1, x2, y2, score])
               
        vehicles = []
        if len(results_lp.boxes.data) > 0 and len(results_vehicle.boxes.data) > 0:
            track_ids = self.mot_tracker.update(np.asarray(detections_))
   
            for license_plate in results_lp.boxes.data.tolist():
                x1, y1, x2, y2, score, _ = license_plate
                xcar1, ycar1, xcar2, ycar2, car_id, bbox_score_vehicle = get_car(license_plate, track_ids)
                if car_id != -1:
                    
                    margin = 0  # Adjust this value based on your requirement
                    alpha = 2  # Adjust this value for the resizing factor

                    # Calculate new coordinates with margin
                    x1_with_margin = max(0, int(x1) - margin)
                    x2_with_margin = min(frame.shape[1], int(x2) + margin)
                    y1_with_margin = max(0, int(y1) - margin)
                    y2_with_margin = min(frame.shape[0], int(y2) + margin)

                    # Perform the crop with margin
                    license_plate_crop_with_margin = frame[y1_with_margin:y2_with_margin, x1_with_margin:x2_with_margin, :]

                    # Resize the cropped image by a factor of alpha
                    resized_license_plate_crop = cv2.resize(license_plate_crop_with_margin, None, fx=alpha, fy=alpha)
                    
                    cv2.imshow(self._camera_id + "_license_plate_crop", resized_license_plate_crop)
                    
                    results = self.ocr_model.ocr(resized_license_plate_crop,
                                    det=self.args.det,
                                    rec=self.args.rec,
                                    cls=self.args.use_angle_cls,
                                    bin=self.args.binarize,
                                    inv=self.args.invert,
                                    alpha_color=self.args.alphacolor)
                    
                    if results[0] is not None:
                        results = sorted(results[0], key=self.sorting_key)
                        result_string = '-'.join(data[1][0] for data in results)
                        
                        confidence_scores = [box[1][1] for box in results]
                        license_plate_text_score = sum(confidence_scores) / len(confidence_scores)
                        data = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'bbox_score': bbox_score_vehicle},
                                                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': result_string,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
                        vehicles.append(data)
                        print(data)
                        print("--------------------------------------------")
        
        for vehicle in vehicles:
            car = vehicle['car']
            lp_data = vehicle['license_plate']
            plot_detection_result(box=car['bbox'], frame=frame_plot, label='vehicle', conf=car['bbox_score'], color=(233, 193, 133))
            plot_detection_result(lp_data['bbox'], frame=frame_plot, label=lp_data['text'], conf=lp_data['text_score'], color=(99, 112, 236))
            
        cv2.imshow(self._camera_id, frame_plot)
            
        key = cv2.waitKey(1)
        if key == ord('q'):
            print()

        return vehicles, frame_plot
        
    def update(self, frame, result):
        self._queue = [frame, result]
        
    def run_thread(self):
        while True:
            if len(self._queue) > 0:
                frame, result = self._queue
                self.run(frame, result)
                
            time.sleep(0.01)
        
    def run(self, frame, result_vehicle, results_lp):
        is_wrong, frame_plot = self._process_frame(frame, result_vehicle, results_lp)
        
        if is_wrong:
            self._event_handler.update(frame, frame_plot)
            
        self._event_handler.post_frame(frame_plot)

        return is_wrong, frame_plot

    def count_frame(self):
        self._count_frame += 1
        
    

    