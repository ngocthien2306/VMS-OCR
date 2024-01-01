import io
import datetime
import xlsxwriter
from utils.project_config import project_config
from services.camera_service import CameraService
from bson.objectid import ObjectId
from db.mongo_db import MongoDB
from utils.models.server_model import *

class ServerSerivce:
    def __init__(self) -> None:
        self._db_connect = MongoDB(project_config.DB_NAME, project_config.SERVER_DOCUMENT)
        
    def convert_objectid(self, record):
        record_id = str(record["_id"])
        record["id"] = record_id
        del record["_id"]
        return record
    
    def get_all(self):
        cursor = self._db_connect._collection.find()
        converted_records = list(map(self.convert_objectid, list(cursor)))
        return converted_records
    
    def get_by_id(self, server_id: str):
        server = self._db_connect._collection.find_one({'server_id': server_id})
        return server
    
    def get_by_server_name(self, server_name: str):
        server = self._db_connect._collection.find_one({'server_name': server_name})
        return server
    
    def add_server(self, model: Server):
        if self.check_id_exist(model.server_id):
            raise Exception("Server ID already exists")
        
        if self.check_ip_exist(model.ip):
            raise Exception("IP address already exists")
        
        self._db_connect.insert_one(model.dict())
        return Server(
                **{
                    **model.dict(), 
                })
        
    def update_server(self,  model: Server):
        if not self.check_id_exist(model.server_id):
            raise Exception("Server ID not exists")
        
        self._db_connect._collection.update_one({'server_id': model.server_id}, {"$set": model.dict()})

        return Server(
                **{
                    **model.dict(), 
                })
        
    def delete_server(self, server_id):
        if not self.check_id_exist(server_id):
            raise Exception("Camera ID not exists")
        
        self._db_connect._collection.delete_one({'server_id': server_id})
        return ""
    
    def check_id_exist(self, server_id: str):
        result = self._db_connect._collection.find_one({'server_id': server_id})
        return result is not None
    
    def check_ip_exist(self, ip: str):
        result = self._db_connect._collection.find_one({'ip': ip})
        return result is not None
    
server_service = ServerSerivce()


