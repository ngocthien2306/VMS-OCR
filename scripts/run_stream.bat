@echo off
call conda activate vms

cd C:\Users\Admin\source\repos\VMS-OCR\streaming
start cmd /k python src/app.py