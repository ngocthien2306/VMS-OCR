from fastapi import APIRouter, HTTPException, Request
from utils.models.server_model import *
from services.server_service import server_service
from services.camera_service import camera_service
from utils.socket import SocketIOClient
import threading
from utils.plc_controller import *
router = APIRouter(prefix="/server")

@router.get("/")
async def get_server():
    records = server_service.get_all()
    return {
        "data": list(records),
        "msg": "success"
    } 
    
@router.post("", response_model=Reponse[Server])
async def add_server(server: Server): 
    try:
        result = server_service.add_server(server)
        
        return {"data": result}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
        
        
@router.put("", response_model=Reponse[Server])
async def update_server(server: Server): 
    try:
        result = server_service.update_server(server)
        return {"data": result}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@router.put("/update-ip", response_model=Reponse[Server])
async def update_ip(model: Server):
    try:
        server = server_service.get_by_server_name(model.server_name)
        if server is not None:
            server['ip'] = model.ip
            del server['_id']
            
            server = server_service.update_server(Server(**server))
            return {"data": server}
        else:
            raise HTTPException(
                status_code=400,
                detail='Server ID not exist'
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
        
       
@router.delete("/{server_id}", response_model=Reponse[object])
async def delete_server(server_id: str):
    try:
        result = server_service.delete_server(server_id)
        return {"data": result}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )      

@router.post('/report')
async def report_camera(model: RequestReport):
    try:
        server = None
        camera = camera_service.get_by_id(model.camera_id)
        if camera is not None:
            server = server_service.get_by_id(camera['server_id'])
        else:
            raise HTTPException(
            status_code=400,
            detail='Camera ID is not existed'
        )  
        
        if server is not None:
            def call_client_by_ip():
                if model.status == 'alarm':
                    print('Starting connect Socket')
                    client_socket = SocketIOClient(f"http://{server['ip']}:5000")
                    client_socket.send_alarm(model.dict())
                    
            def connect_plc():
                print('Starting connect PLC')
                
                _plc_master_light = create_plc_instance()
                _plc_master_sound = create_plc_instance(var=8222)

                if model.status == 'alarm':
                    time.sleep(0.02)
                    _plc_master_light.turn_on()
                    time.sleep(0.02)
                    _plc_master_sound.turn_on()     
                else:
                    time.sleep(0.02)
                    _plc_master_light.turn_off()
                    time.sleep(0.02)
                    _plc_master_sound.turn_off()
                
                plc_ip = camera['plc']['ip']
                list_config = {}
                for i, device in enumerate(camera['plc']['device']): 
                    _plc_controller = create_plc_instance(ip=plc_ip, var=device['var'])
                    
                    if model.status == 'alarm':
                        time.sleep(0.02)
                        _plc_controller.turn_on()
                    else:
                        time.sleep(0.02)
                        _plc_controller.turn_off()
                            
            background_thread = threading.Thread(target=connect_plc)
            background_thread.start()
            background_thread = threading.Thread(target=call_client_by_ip)
            background_thread.start()
        else:
            raise HTTPException(
            status_code=400,
            detail='Server is not existed'
        )  
        return {"detail": 'Send alarm successfull', 'status_code': 200}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )  

def create_plc_instance(ip="192.168.1.250", port=502, address=1, var=8212):
    plc_controller_config = PLCControllerConfig(
        plc_ip_address=ip,
        plc_port=port,
        plc_address=address,
        modbus_address=var
    )
    _plc_controller = PLCController(plc_controller_config)
    return _plc_controller

@router.post('/off-all-plc')
def turn_off_all_plc():
    try:
        cameras = camera_service.get_all_camera()
        for cam in cameras:
            plc_ip = cam['plc']['ip']
            list_config = {}
            for i, device in enumerate(cam['plc']['device']): 
                plc_controller_config = PLCControllerConfig(
                    plc_ip_address=plc_ip,
                    plc_port=502,
                    plc_address=1,
                    modbus_address=device['var']
                )
                _plc_controller = PLCController(plc_controller_config)
                list_config[i] = _plc_controller
                
            for i in range(len(cam['plc']['device'])):
                time.sleep(0.02)
                res1 = list_config[i].turn_off()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )      
  
            

