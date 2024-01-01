from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from utils import model
from services.history_service import get_all, report_false, export_xlsx, count_report

router = APIRouter(prefix="/history")

@router.get("")
async def get_history(page: int = 0, size: int = 10, module_id: str = None, camera_id: str = None, shift: int = None, line: int = None, start_timestamp: int = 0, end_timestamp = 0):
    filter = model.Filter(
        page=page,
        size=size,
        module_id=module_id,
        camera_id=camera_id,
        shift=shift,
        line=line,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp
        )
    records, count = get_all(filter)
    return {
        "data": list(records),
        "total_records": count,
        "msg": "success"
    }

@router.get("/violate-count")
async def get_violate_count():
    result, count = count_report()
    print(result)
    return {
        "data": list(result),
        "total_records": count,
        "msg": "success"
    }
    

@router.get("/export")
async def get_history(page: int = 0, size: int = 10, module_id: str = None, camera_id: str = None, shift: int = None, line: int = None, start_timestamp: int = 0, end_timestamp = 0):
    filter = model.Filter(
        page=page,
        size=size,
        module_id=module_id,
        camera_id=camera_id,
        shift=shift,
        line=line,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp
        )
    response = export_xlsx(filter)
    headers = {
        'Content-Disposition': 'attachment; filename="CCTV_AI_HISTORY.xlsx"'
    }
    return StreamingResponse(response, headers=headers)

@router.post("/report-false")
async def report_history_false(body: model.ReportEvent):
    report_false(body.id)
    return {
        "data": {},
        "msg": "success"
    }
