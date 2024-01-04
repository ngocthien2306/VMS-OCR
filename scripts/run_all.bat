@echo off
call conda activate vms

cd C:\Users\Admin\source\repos\VMS-OCR\be-cctv
start cmd /k python src/app.py

cd ..

cd C:\Users\Admin\source\repos\VMS-OCR\streaming
start cmd /k python src/app.py

cd ..

cd C:\Users\Admin\source\repos\VMS-OCR\vehicle-detection
start cmd /k python main.py
