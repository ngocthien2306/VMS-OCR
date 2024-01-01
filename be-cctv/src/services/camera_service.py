from utils.project_config import project_config
import io
import datetime
import xlsxwriter
from utils.project_config import project_config

from bson.objectid import ObjectId
from db.mongo_db import MongoDB
from utils.model import *


class CameraService:
    def __init__(self) -> None:
        self._db_connect = MongoDB(project_config.DB_NAME, project_config.CAMERA_DOCUMENT)
    
    def convert_objectid(self, record):
        record_id = str(record["_id"])
        record["id"] = record_id
        del record["_id"]
        return record

    def get_camera_by_server(self, server_id: str):
        query = {}
        if server_id is not None:
            query['server_id'] = server_id
        
        cursor = self._db_connect._collection.find(query)
        converted_records = list(map(self.convert_objectid, list(cursor)))
        return converted_records
    
    def get_all_camera(self):
        cursor = self._db_connect._collection.find()
        converted_records = list(map(self.convert_objectid, list(cursor)))
        return converted_records
    
    def get_by_id(self, camera_id: str):
        query = {}
        if camera_id is not None:
            query['camera_id'] = camera_id
            
        return self._db_connect.find_one(query)
    
            
    def add_camera(self, camera: Camera):
        if self.check_camera_id_exist(camera.camera_id):
            raise Exception("Camera ID already exists")
        
        if self.check_rtsp_exist(camera.rtsp_link):
            raise Exception("RTSP Link already exists")
        
        self._db_connect.insert_one(camera.dict())
        return CameraResponse(
                **{
                    **camera.dict(), 
                })
        
    def update_camera(self, camera: Camera):
        if not self.check_camera_id_exist(camera.camera_id):
            raise Exception("Camera ID already exists")
        
        self._db_connect._collection.update_one({'camera_id': camera.camera_id}, {"$set": camera.dict()})

        return CameraResponse(
            **{
                **camera.dict(), 
            })
        
    def delete_camera(self, camera_id):
        if not self.check_camera_id_exist(camera_id):
            raise Exception("Camera ID not exists")
        
        self._db_connect._collection.delete_one({'camera_id': camera_id})
        return ""
        
            
    def check_camera_id_exist(self, camera_id: str):
        result = self._db_connect._collection.find_one({'camera_id': camera_id})
        return result is not None

    def check_rtsp_exist(self, rtsp: str):
        result = self._db_connect._collection.find_one({'rtsp_link': rtsp})
        return result is not None
    
    def check_server_exist(self, server_id: str):
        result = self._db_connect._collection.find_one({'server_id': server_id})
        return result is not None
    
camera_service = CameraService()
    
    