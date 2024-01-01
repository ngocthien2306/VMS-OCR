from fastapi import APIRouter

from utils import model
from utils.socket import socket_connection
from services.event_service import write_log, write_video_log
from utils.plc_controller import *
from services.camera_service import camera_service
import time
import threading

router = APIRouter(prefix="/event")

@router.post("")
async def post_event(event: model.Event):
    
    camera = camera_service.get_by_id(event.camera_id)
    current_time =  event.dict()['timestamp']

    if current_time > camera['start_time'] and current_time < camera['end_time']:
        
        await socket_connection.send_data(
            channel="alert",
            data=event.dict()
        )
        
        def connect_plc():
            plc_controller_config = PLCControllerConfig(
                    plc_ip_address="192.168.1.250",
                    plc_port=502,
                    plc_address=1,
                    modbus_address=8212
            )
            
            _plc_controller = PLCController(plc_controller_config)
            
            time.sleep(0.02)
            _plc_controller.turn_on()

            if camera is not None:
                plc_ip = camera['plc']['ip']
                list_config = {}
                for i, device in enumerate(camera['plc']['device']): 
                    if "Den" in device['device_name']:
                        plc_controller_config = PLCControllerConfig(
                            plc_ip_address=plc_ip,
                            plc_port=502,
                            plc_address=1,
                            modbus_address=device['var']
                        )
                        _plc_controller = PLCController(plc_controller_config)
                        time.sleep(0.02)
                        _plc_controller.turn_on()
            
        background_thread = threading.Thread(target=connect_plc)
        background_thread.start()
        return "success"
    
    return "fail"

@router.post('/video')
async def save_log(event: model.EventVideo):
    print(event)
    event = write_video_log(event)
    return event


