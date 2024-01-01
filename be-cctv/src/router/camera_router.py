from fastapi import APIRouter, HTTPException, Request
from utils.model import *
from services.camera_service import camera_service
from services.server_service import server_service
import requests
import threading

router = APIRouter(prefix="/camera")

@router.get("/{server_name}")
async def get_camera(server_name: str=None):
    server = server_service.get_by_server_name(server_name)
    records = camera_service.get_camera_by_server(server['server_id'])
    
    if server is not None:
        
        server['cameras'] = records if records is not None else []
        del server['_id']
        return {
            "data": server,
            "msg": "success",
        } 
    else:
        return {
            "data": {},
            "msg": "fail"
        } 

@router.get("/")
async def get_all_camera():
    servers = server_service.get_all()

    if servers is not None:
        for server in servers:
            
            cameras = camera_service.get_camera_by_server(server['server_id'])
            path = "http://{0}:8005/stream-manage/output/motion-detections-{1}"
            for camera in cameras:
                camera['stream_url'] = path.format(server['ip'], camera['camera_id'])
            
            server['cameras'] = cameras if cameras is not None else []
            del server['id']
            
        return {
            "data": list(servers),
            "msg": "success",
            
        } 
    else:
        return {
            "data": {},
            "msg": "fail"
        } 
        
        

@router.post("", response_model=Reponse[CameraResponse])
async def add_camera_api(camera: Camera): 
    try:
        
        result = camera_service.add_camera(camera)
        
        def add_to_streaming():
            server = server_service.get_by_id(camera.server_id)
            server_name = server['server_name']
            root_url = f'http://{server_name}:8005/stream-manage/camera'
            requests.post(root_url, json=camera.dict())
        
        background_thread = threading.Thread(target=add_to_streaming)
        background_thread.start()
        
        return {"data": result}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@router.delete("/{camera_id}", response_model=Reponse[object])
async def delete_camera(camera_id: str):
    try:
        camera = camera_service.get_by_id(camera_id)
        result = camera_service.delete_camera(camera_id)
        
        def delete_to_streaming():
            server = server_service.get_by_id(camera['server_id'])
            server_name = server['server_name']
            root_url = f'http://{server_name}:8005/stream-manage/camera/{camera_id}'
            requests.delete(root_url)
            
        background_thread = threading.Thread(target=delete_to_streaming)
        background_thread.start()
        
        return {"data": result}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
        
@router.put("", response_model=Reponse[CameraResponse])
async def update_camera(camera: Camera): 
    try:
        result = camera_service.update_camera(camera)
        
        def refesh_streaming():
            server = server_service.get_by_id(camera.server_id)
            server_name = server['server_name']
            root_url = f'http://{server_name}:8005/stream-manage/camera/refresh'
            requests.get(root_url)
            
        background_thread = threading.Thread(target=refesh_streaming)
        background_thread.start()
        
        return {"data": result}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
        
@router.put("/update-date", response_model=Reponse[CameraResponse])
async def update_camera_date(model: RangeDate):
    try:
        result = camera_service.get_by_id(model.camera_id)  
        if result is None: 
            raise HTTPException(
            status_code=400,
            detail='Camera ID does not existed'
        )
        print(result)
        result['start_time'] = model.start_time
        result['end_time'] = model.end_time
        result = camera_service.update_camera(Camera(**result))
        
        return {"data": result}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

   
@router.put("/update-status", response_model=Reponse[CameraResponse])
async def update_camera_status(camera: Camera): 
    try:
        result = camera_service.get_by_id(camera.camera_id)
        result['camera_status'] = camera.camera_status
        result = camera_service.update_camera(Camera(**result))
        
        return {"data": result}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


