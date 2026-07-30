@echo off
cd /d "%~dp0"
if exist runs\.iql_wall_easy_done  del /q runs\.iql_wall_easy_done
if exist runs\.vdn_wall_easy_done  del /q runs\.vdn_wall_easy_done
if exist runs\.qmix_wall_easy_done del /q runs\.qmix_wall_easy_done

start "IQL wall-easy"  cmd /k ".venv\Scripts\python.exe train.py --algo iql  --episodes 4000 --tag iql_wall_easy  --difficulty easy --init-from iql_nowall  --seed 0 & echo done > runs\.iql_wall_easy_done"
start "VDN wall-easy"  cmd /k ".venv\Scripts\python.exe train.py --algo vdn  --episodes 4000 --tag vdn_wall_easy  --difficulty easy --init-from vdn_nowall  --seed 0 & echo done > runs\.vdn_wall_easy_done"
start "QMIX wall-easy" cmd /k ".venv\Scripts\python.exe train.py --algo qmix --episodes 4000 --tag qmix_wall_easy --difficulty easy --init-from qmix_nowall --seed 0 & echo done > runs\.qmix_wall_easy_done"
