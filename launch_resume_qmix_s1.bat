@echo off
REM qmix_wall_medium_s1 ep2000'de kesildi (surec kapandi). Checkpoint saglam
REM (steps=449006, CSV ep2000 satiriyla birebir ayni). Kalan 2000 episode
REM --resume ile kaldigi epsilon/agirlik durumundan surduruluyor.
cd /d "%~dp0"
if exist runs\.qmix_wall_medium_s1_done del /q runs\.qmix_wall_medium_s1_done

start "QMIX med s1 RESUME" cmd /k ".venv\Scripts\python.exe train.py --algo qmix --episodes 2000 --tag qmix_wall_medium_s1 --difficulty medium --init-from qmix_wall_medium_s1 --resume --seed 1 & echo done > runs\.qmix_wall_medium_s1_done"
