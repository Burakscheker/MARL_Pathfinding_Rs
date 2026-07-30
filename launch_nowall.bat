@echo off
cd /d "%~dp0"
if exist runs\.iql_nowall_done  del /q runs\.iql_nowall_done
if exist runs\.vdn_nowall_done  del /q runs\.vdn_nowall_done
if exist runs\.qmix_nowall_done del /q runs\.qmix_nowall_done

start "IQL nowall"  cmd /k ".venv\Scripts\python.exe train.py --algo iql  --episodes 10000 --tag iql_nowall  --seed 0 & echo done > runs\.iql_nowall_done"
start "VDN nowall"  cmd /k ".venv\Scripts\python.exe train.py --algo vdn  --episodes 10000 --tag vdn_nowall  --seed 0 & echo done > runs\.vdn_nowall_done"
start "QMIX nowall" cmd /k ".venv\Scripts\python.exe train.py --algo qmix --episodes 10000 --tag qmix_nowall --seed 0 & echo done > runs\.qmix_nowall_done"
