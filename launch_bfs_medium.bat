@echo off
REM BFS-ozellikli ince ayar, EGITILMIS nowall uzerinden (--init-from).
REM Onceki dogrulama kosusu SIFIRDAN baslamisti ve cokmustu (%0) -- o test
REM iki degisikligi birden yapiyordu, BFS ozelliklerini olcemiyordu.
REM Bu kosu referansla birebir kiyaslanabilir: AYNI kurulum (IQL, medium,
REM 4000 ep, --init-from nowall), TEK fark BFS gozlem ozellikleri.
REM Referans: 3-seed ortalamasi %47.5 (%36.5 / %43.8 / %62.3)
cd /d "%~dp0"
if exist runs\.iql_bfs_medium_done del /q runs\.iql_bfs_medium_done

start "IQL BFS medium" cmd /k ".venv\Scripts\python.exe train.py --algo iql --episodes 4000 --tag iql_bfs_medium --difficulty medium --init-from iql_nowall_bfs --seed 0 & echo done > runs\.iql_bfs_medium_done"
