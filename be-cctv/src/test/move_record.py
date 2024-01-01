import datetime
from tqdm import tqdm
from db.mongo_db import db_connect
from bson.objectid import ObjectId

collection_event = db_connect.get_collection()
collection_event_false = db_connect.get_collection("event_false")

cursor = collection_event.find({"set_false": True})
records = list(cursor)

print(len(records))

for record in tqdm(records):
    try:
        collection_event_false.insert_one(record)
    except:
        pass
    
for record in tqdm(records):
    try:
        collection_event.delete_one({"_id": record["_id"]})
    except:
        pass