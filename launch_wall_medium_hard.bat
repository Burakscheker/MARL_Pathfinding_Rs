@echo off
cd /d "%~dp0"
if exist runs\.iql_wall_medium_done  del /q runs\.iql_wall_medium_done
if exist runs\.vdn_wall_medium_done  del /q runs\.vdn_wall_medium_done
if exist runs\.qmix_wall_medium_done del /q runs\.qmix_wall_medium_done
if exist runs\.iql_wall_hard_done  del /q runs\.iql_wall_hard_done
if exist runs\.vdn_wall_hard_done  del /q runs\.vdn_wall_hard_done
if exist runs\.qmix_wall_hard_done del /q runs\.qmix_wall_hard_done

start "IQL wall-medium"  cmd /k ".venv\Scripts\python.exe train.py --algo iql  --episodes 4000 --tag iql_wall_medium  --difficulty medium --init-from iql_nowall  --seed 0 & echo done > runs\.iql_wall_medium_done"
start "VDN wall-medium"  cmd /k ".venv\Scripts\python.exe train.py --algo vdn  --episodes 4000 --tag vdn_wall_medium  --difficulty medium --init-from vdn_nowall  --seed 0 & echo done > runs\.vdn_wall_medium_done"
start "QMIX wall-medium" cmd /k ".venv\Scripts\python.exe train.py --algo qmix --episodes 4000 --tag qmix_wall_medium --difficulty medium --init-from qmix_nowall --seed 0 & echo done > runs\.qmix_wall_medium_done"

start "IQL wall-hard"  cmd /k ".venv\Scripts\python.exe train.py --algo iql  --episodes 4000 --tag iql_wall_hard  --difficulty hard --init-from iql_nowall  --seed 0 & echo done > runs\.iql_wall_hard_done"
start "VDN wall-hard"  cmd /k ".venv\Scripts\python.exe train.py --algo vdn  --episodes 4000 --tag vdn_wall_hard  --difficulty hard --init-from vdn_nowall  --seed 0 & echo done > runs\.vdn_wall_hard_done"
start "QMIX wall-hard" cmd /k ".venv\Scripts\python.exe train.py --algo qmix --episodes 4000 --tag qmix_wall_hard --difficulty hard --init-from qmix_nowall --seed 0 & echo done > runs\.qmix_wall_hard_done"
