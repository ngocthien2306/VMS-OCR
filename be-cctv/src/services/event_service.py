import datetime

from db.mongo_db import db_connect
from utils import model

def get_shift_by_timestamp(timestamp: int):
    datetime_obj = datetime.datetime.fromtimestamp(timestamp)
    h = datetime_obj.hour
    if 6 <= h < 14:
        return 1
    if 14 <= h < 22:
        return 2
    return 3

def write_log(event: model.Event):
    shift = get_shift_by_timestamp(event.timestamp)
    event_log = model.EventLog(
        **event.dict(), shift = shift
    )
    db_connect.insert_one(event_log.dict())
    
def write_video_log(event: model.EventVideo):
    shift = get_shift_by_timestamp(event.timestamp)
    event_log = model.EvenVideotLog(
        **event.dict(), shift = shift
    )
    db_connect.insert_one(event_log.dict())
    return event_log