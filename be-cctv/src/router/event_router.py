from fastapi import APIRouter, HTTPException
from utils import model
from services.event_service import write_log, write_video_log
from utils.socket import AsyncSocketIOClient
import socketio
import traceback

client_socket = AsyncSocketIOClient("http://dev.i-soft.com.vn:5000")
router = APIRouter(prefix="/event")

@router.on_event("startup")
async def startup_event():
    await client_socket.connect()
    await client_socket.healthy_check_event("healthy_check", {"msg": "Hello World!"})
    

@router.on_event("shutdown")
async def shutdown_event():
    await client_socket.disconnect()

@router.post("")
async def post_event(event: model.Event):
    try:
        await client_socket.emit_event("alert", event.dict())
        event = write_log(event)
        return {"message": "success"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post('/video')
async def save_log(event: model.EventVideo):
    try:
        print(event)
        event = write_video_log(event)
        return {"message": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
