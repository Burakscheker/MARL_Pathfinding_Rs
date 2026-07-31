@echo off
cd /d "%~dp0"
if exist runs\.vdn_wall_medium_done  del /q runs\.vdn_wall_medium_done
if exist runs\.qmix_wall_medium_done del /q runs\.qmix_wall_medium_done
if exist runs\.vdn_wall_hard_done  del /q runs\.vdn_wall_hard_done
if exist runs\.qmix_wall_hard_done del /q runs\.qmix_wall_hard_done

start "VDN wall-medium RESUME"  cmd /k ".venv\Scripts\python.exe train.py --algo vdn  --episodes 1000 --tag vdn_wall_medium  --difficulty medium --init-from vdn_wall_medium  --resume --seed 0 & echo done > runs\.vdn_wall_medium_done"
start "QMIX wall-medium RESUME" cmd /k ".venv\Scripts\python.exe train.py --algo qmix --episodes 2000 --tag qmix_wall_medium --difficulty medium --init-from qmix_wall_medium --resume --seed 0 & echo done > runs\.qmix_wall_medium_done"
start "VDN wall-hard RESUME"  cmd /k ".venv\Scripts\python.exe train.py --algo vdn  --episodes 2000 --tag vdn_wall_hard  --difficulty hard --init-from vdn_wall_hard  --resume --seed 0 & echo done > runs\.vdn_wall_hard_done"
start "QMIX wall-hard RESUME" cmd /k ".venv\Scripts\python.exe train.py --algo qmix --episodes 3000 --tag qmix_wall_hard --difficulty hard --init-from qmix_wall_hard --resume --seed 0 & echo done > runs\.qmix_wall_hard_done"
