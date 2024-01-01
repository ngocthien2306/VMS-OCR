from db.mongo_db import db_connect
import datetime

def fill(records):
    start_date = datetime.date.today()
    response = []
    for i in range(0, 10, 1):
        end_date = start_date - datetime.timedelta(days=i)
        end_date_str = end_date.strftime("%Y-%m-%d")
        result_date_in_record = list(filter(lambda r: r['timestamp'] == end_date_str, records))
        if result_date_in_record:
            response.append({
                "timestamp": end_date_str,
                "value": result_date_in_record[0]["value"]
            })
        else: 
            response.append({
                "timestamp": end_date_str,
                "value": 0
            })
    response.reverse()
    return response

def convert_objectid(record):
    record_id = str(record["_id"])
    record["timestamp"] = record_id
    del record["_id"]
    return record

def get_all_camera_of_module(module_id):
    collection = db_connect.get_collection()
    camera_ids = collection.distinct("camera_id", {"module_id": module_id})
    return camera_ids


def group_by_day_all_of_module(module_id, start_timestamp=None, end_timestamp=None):
    collection = db_connect.get_collection()
    records = collection.aggregate(
        [
        {"$match": {
            "module_id": module_id,
            }
        },
        {
            "$group": {
                "_id": {
                    "$dateToString": {
                        "format": "%Y-%m-%d",
                        "date": "$_id",
                    },
                },
                "value": { "$sum": 1 },
            }
        },
        {
                "$sort": {
                    "_id": -1
                }
        }
        ]
    )
    records = list(map(convert_objectid, records))
    response = {
        "module_id": module_id,
        "timeseries": fill(records)
    }
    return response

def group_by_day_camera_of_module(module_id, start_timestamp=None, end_timestamp=None):
    collection = db_connect.get_collection()
    camera_ids = get_all_camera_of_module(module_id)
    response = []
    for camera_id in camera_ids:
        records = collection.aggregate(
            [
            {"$match": {
                "module_id": module_id,
                "camera_id": camera_id
                }
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$_id",
                        },
                    },
                    "value": { "$sum": 1 },
                }
            }, 
            {
                "$sort": {
                    "_id": -1
                }
            }
            ]
        )
        records = list(map(convert_objectid, records))
        response.append({
            "module_id": module_id,
            "camera_id": camera_id,
            "timeseries": fill(records)
        })
    return response

if __name__ == "__main__":
    print(group_by_day_camera_of_module("ppe"))
