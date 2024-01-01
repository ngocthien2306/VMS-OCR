from fastapi import APIRouter
from services.report_service import group_by_day_all_of_module, group_by_day_camera_of_module

router = APIRouter(prefix="/report")

@router.get("/group-by/day/all")
async def day_all(module_id: str):
    response = group_by_day_all_of_module(module_id)
    return {
        "data": response,
        "msg": "success"
    }

@router.get("/group-by/day/camera")
async def day_camera(module_id: str):
    response = group_by_day_camera_of_module(module_id)
    return {
        "data": response,
        "msg": "success"
    }