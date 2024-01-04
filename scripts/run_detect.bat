@echo off
call conda activate vms

cd C:\Users\Admin\source\repos\VMS-OCR\vehicle-detection
start cmd /k python main.py
