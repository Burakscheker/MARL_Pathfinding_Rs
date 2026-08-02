@echo off
REM Dogrulama kosusu: engel-farkinda BFS gozlem ozellikleri gercekten
REM basari oranini yukseltiyor mu? Referans: AYNI kombinasyonun (IQL,
REM medium duvar, 4000 episode) 3-seed ortalamasi %47.5 (%36.5/%43.8/%62.3).
REM Hedef: %70+.  Sifirdan egitiliyor (--init-from YOK) -- BFS ozellikleri
REM navigasyonu kolaylastirdiysa nowall on-egitimine artik gerek olmamali.
cd /d "%~dp0"
if exist runs\.iql_bfs_medium_done del /q runs\.iql_bfs_medium_done

start "IQL BFS-dogrulama" cmd /k ".venv\Scripts\python.exe train.py --algo iql --episodes 4000 --tag iql_bfs_medium --difficulty medium --seed 0 & echo done > runs\.iql_bfs_medium_done"
