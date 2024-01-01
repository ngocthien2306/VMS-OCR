from typing import Generic, TypeVar, List
from pydantic import BaseModel
from pydantic.generics import GenericModel
from datetime import datetime

M = TypeVar("M", bound=BaseModel)


class Event(BaseModel):
    camera_id: str
    module_id: str
    timestamp: int
    image_uri: str
    msgType: int


class EventVideo(BaseModel):
    camera_id: str
    module_id: str
    timestamp: int
    video_uri: str
    msgType: int
    dns: str
    
class EvenVideotLog(EventVideo):
    set_false: bool = False
    line: int = 2
    shift: int = 1

class EventLog(Event):
    set_false: bool = False
    line: int = 2
    shift: int = 1

class History(EventLog):
    id: str

class Filter(BaseModel):
    page: int = 0
    size: int = 10
    module_id: str = None
    camera_id: str = None
    shift: int = None
    line: int = None 
    start_timestamp: int = 0
    end_timestamp: int = 0

class ReportEvent(BaseModel):
    id: str
    


class RangeDate(BaseModel):
    start_time: float = datetime.timestamp(datetime.now())
    end_time: float = datetime.timestamp(datetime.now())
    camera_id: str = ""
    
    
class PLCDevice(BaseModel):
    device_name: str = ''
    address: str = ''
    var: int = 0
    slave: str = ''
    master: str = ''

class PLCModel(BaseModel):
    ip: str = ""
    device: List[PLCDevice] = []
    station: str = ""

class Camera(BaseModel):
    rtsp_link: str = ""
    camera_id: str = ""
    camera_name: str = ""
    camera_location: str = ""
    frame_rate: int = 6
    output_width: int = 1280
    output_height: int = 720
    server_id: str = "" 
    camera_status: str = "INIT"
    start_time: float = datetime.timestamp(datetime.now())
    end_time: float = datetime.timestamp(datetime.now())
    plc: PLCModel = None
    
class CameraInfo(BaseModel):
    camera_ip: str
    username: str
    password: str

class CameraResponse(Camera):
    state: str = "INIT"

class Reponse(GenericModel, Generic[M]):
    data: M
    status: str = "success"

class ListReponse(GenericModel, Generic[M]):
    data: List[M]
    status: str = "success"


