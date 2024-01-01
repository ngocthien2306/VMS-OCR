import datetime
import xlsxwriter
import io

from tqdm import tqdm
from db.mongo_db import db_connect

# output = io.BytesIO()
# workbook = Workbook(output, {'in_memory': True})
workbook = xlsxwriter.Workbook('Example2.xlsx')
worksheet = workbook.add_worksheet()

collection = db_connect.get_collection()
cursor = collection.find({})
index = 0
for record in cursor:
    if index > 20:
        break
    timestamp = record["timestamp"]
    camera_id = record["camera_id"]
    module_id = record["module_id"]
    worksheet.write(index, 0, camera_id)
    worksheet.write(index, 1, module_id)
    worksheet.write(index, 2, datetime.datetime.fromtimestamp(timestamp).strftime("%d/%m/%Y, %H:%M:%S"))

    index += 1

workbook.close()