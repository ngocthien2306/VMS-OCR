import datetime
from tqdm import tqdm
from db.mongo_db import db_connect
from bson.objectid import ObjectId

def get_shift_by_timestamp(timestamp: int):
    datetime_obj = datetime.datetime.fromtimestamp(timestamp)
    h = datetime_obj.hour
    if 6 <= h < 14:
        return 1
    if 14 <= h < 22:
        return 2
    return 3

collection = db_connect.get_collection()
cursor = collection.find({})
for record in tqdm(cursor):
    id_record = record["_id"]
    timestamp = record["timestamp"]
    collection.update_one({"_id": id_record}, {"$set": { "shift": get_shift_by_timestamp(timestamp), "line": 2 }})

    # cursor = collection.find({"_id": id_record})
    # print(cursor[0])