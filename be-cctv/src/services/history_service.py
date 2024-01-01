import io
import datetime
import xlsxwriter

from bson.objectid import ObjectId

from utils import model
from db.mongo_db import db_connect

def convert_objectid(record):
    record_id = str(record["_id"])
    record["id"] = record_id
    del record["_id"]
    return record

def get_all(filter: model.Filter):
    query = {}
    if filter.module_id is not None:
        query["module_id"] = filter.module_id
    if filter.camera_id is not None:
        query["camera_id"] = filter.camera_id
    if filter.shift is not None:
        query["shift"] = filter.shift
    if filter.line is not None:
        query["line"] = filter.line

    if filter.size > 50:
        filter.size = 50

    if filter.start_timestamp > 0:
        if not query.get("timestamp"):
            query["timestamp"] = {}
        query["timestamp"]["$gte"] = filter.start_timestamp
    if filter.end_timestamp > 0:
        if not query.get("timestamp"):
            query["timestamp"] = {}
        query["timestamp"]["$lte"] = filter.end_timestamp

    collection = db_connect.get_collection()
    cursor = collection.find(query).sort("timestamp", -1)
    count = collection.count_documents(query)
    skips = filter.size * filter.page
    records = cursor.skip(skips).limit(filter.size)
    return map(convert_objectid, records), count

def count_report():
    collection = db_connect.get_collection('cameras')
    distinct_camera_ids = collection.distinct("camera_id")
    
    collection = db_connect.get_collection('event_logs')
    count = collection.count_documents({})

    # Group by camera_id and count the occurrences
    pipeline = [
        {
            "$group": {
                "_id": "$camera_id",  # Group by the value of camera_id
                "count": {"$sum": 1}  # Count the occurrences for each camera_id
            }
        },
        {
            "$project": {
                "camera_id": "$_id",  # Include camera_id in the result
                "count": 1,           # Include the count in the result
                "_id": 0              # Exclude the _id field from the result
            }
        }
    ]

    result = list(collection.aggregate(pipeline))
    
    final_result = [{"count": next((item["count"] for item in result if item["camera_id"] == camera_id), 0),
                    "camera_id": camera_id} for camera_id in distinct_camera_ids]

    return final_result, count

def report_false(id: str):
    collection = db_connect.get_collection()
    collection.update_one({"_id": ObjectId(id)}, {"$set": { "set_false": True }})

def export_xlsx(filter: model.Filter):
    records, _ = get_all(filter)
    
    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet()

    worksheet.write(0, 0, "THOI GIAN")
    worksheet.write(0, 1, "CAMERA")
    worksheet.write(0, 2, "MODULE")
    worksheet.write(0, 3, "CA LAM VIEC")
    worksheet.write(0, 4, "LINE")


    for index, record in enumerate(records):
        row_index = index + 1
        timestamp = record["timestamp"]
        date_time = datetime.datetime.fromtimestamp(timestamp).strftime("%d/%m/%Y, %H:%M:%S")
        camera_id = record["camera_id"]
        module_id = record["module_id"]
        shift = record["shift"]
        line = record["line"]

        worksheet.write(row_index, 0, date_time)
        worksheet.write(row_index, 1, camera_id)
        worksheet.write(row_index, 2, module_id)
        worksheet.write(row_index, 3, shift)
        worksheet.write(row_index, 4, line)
    workbook.close()
    output.seek(0)
    return output