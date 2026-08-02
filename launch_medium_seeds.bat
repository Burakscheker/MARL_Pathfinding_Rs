@echo off
REM Medium zorlukta seed varyansi olcumu. Seed 0 zaten mevcut
REM (*_wall_medium tag'leri), burada seed 1 ve 2 ekleniyor.
REM Hepsi AYNI *_nowall checkpoint'inden basliyor -- sadece INCE-AYAR
REM fazinin seed'i degisiyor (konfig ornekleme, epsilon-greedy kesif,
REM buffer ornekleme). Amac: "medium kosusu sanssiz miydi" sorusu.
cd /d "%~dp0"
if exist runs\.iql_wall_medium_s1_done  del /q runs\.iql_wall_medium_s1_done
if exist runs\.vdn_wall_medium_s1_done  del /q runs\.vdn_wall_medium_s1_done
if exist runs\.qmix_wall_medium_s1_done del /q runs\.qmix_wall_medium_s1_done
if exist runs\.iql_wall_medium_s2_done  del /q runs\.iql_wall_medium_s2_done
if exist runs\.vdn_wall_medium_s2_done  del /q runs\.vdn_wall_medium_s2_done
if exist runs\.qmix_wall_medium_s2_done del /q runs\.qmix_wall_medium_s2_done

start "IQL med s1"  cmd /k ".venv\Scripts\python.exe train.py --algo iql  --episodes 4000 --tag iql_wall_medium_s1  --difficulty medium --init-from iql_nowall  --seed 1 & echo done > runs\.iql_wall_medium_s1_done"
start "VDN med s1"  cmd /k ".venv\Scripts\python.exe train.py --algo vdn  --episodes 4000 --tag vdn_wall_medium_s1  --difficulty medium --init-from vdn_nowall  --seed 1 & echo done > runs\.vdn_wall_medium_s1_done"
start "QMIX med s1" cmd /k ".venv\Scripts\python.exe train.py --algo qmix --episodes 4000 --tag qmix_wall_medium_s1 --difficulty medium --init-from qmix_nowall --seed 1 & echo done > runs\.qmix_wall_medium_s1_done"

start "IQL med s2"  cmd /k ".venv\Scripts\python.exe train.py --algo iql  --episodes 4000 --tag iql_wall_medium_s2  --difficulty medium --init-from iql_nowall  --seed 2 & echo done > runs\.iql_wall_medium_s2_done"
start "VDN med s2"  cmd /k ".venv\Scripts\python.exe train.py --algo vdn  --episodes 4000 --tag vdn_wall_medium_s2  --difficulty medium --init-from vdn_nowall  --seed 2 & echo done > runs\.vdn_wall_medium_s2_done"
start "QMIX med s2" cmd /k ".venv\Scripts\python.exe train.py --algo qmix --episodes 4000 --tag qmix_wall_medium_s2 --difficulty medium --init-from qmix_nowall --seed 2 & echo done > runs\.qmix_wall_medium_s2_done"
