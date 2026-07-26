@echo off
REM ============================================================
REM  MARL Pathfinding - IQL + VDN + QMIX otomatik egitim + rapor
REM
REM  Kullanim:
REM    run_all.bat            -> 30000 episode (varsayilan)
REM    run_all.bat 5000       -> 5000 episode
REM
REM  Ne yapar:
REM    1) IQL, VDN, QMIX egitimlerini AYNI ANDA (paralel) baslatir
REM    2) Ucu de bitene kadar bekler (saatler surebilir, ozellikle QMIX)
REM    3) eval.evaluate ile final karsilastirma tablosunu uretir
REM    4) viz.plot_iql_report --final ile grafikleri uretir
REM
REM  Cikti dosyalari:
REM    runs\eval_report.md
REM    runs\viz\final_algorithm_comparison.png
REM    runs\viz\final_success_rate_stacked.png
REM    runs\viz\{iql,vdn,qmix}_final_demo_grids.png
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

set VENV=.venv\Scripts\python.exe
set SEED=0

set EPISODES=%1
if "%EPISODES%"=="" set EPISODES=30000

if not exist runs mkdir runs

echo ============================================================
echo  MARL Pathfinding otomatik kosum
echo  Episodes : %EPISODES%
echo  Seed     : %SEED%
echo  Baslangic: %date% %time%
echo ============================================================
echo.

if exist runs\.iql_done  del /q runs\.iql_done
if exist runs\.vdn_done  del /q runs\.vdn_done
if exist runs\.qmix_done del /q runs\.qmix_done

echo [1/4] IQL, VDN, QMIX egitimleri PARALEL baslatiliyor...
echo       (her biri kendi log dosyasina yaziyor, bu pencere serbest kalir)

REM "&" (&& degil) kullaniliyor: egitim HATA ile de bitse marker dosyasi
REM yine de yazilsin, yoksa asagidaki bekleme dongusu sonsuza kadar takilir.

start "IQL egitimi"  /B cmd /c "%VENV% train.py --algo iql  --episodes %EPISODES% --tag iql_final  --curriculum --seed %SEED% > runs\iql_final_stdout.log  2>&1 & echo done > runs\.iql_done"
start "VDN egitimi"  /B cmd /c "%VENV% train.py --algo vdn  --episodes %EPISODES% --tag vdn_final  --curriculum --seed %SEED% > runs\vdn_final_stdout.log  2>&1 & echo done > runs\.vdn_done"
start "QMIX egitimi" /B cmd /c "%VENV% train.py --algo qmix --episodes %EPISODES% --tag qmix_final --curriculum --seed %SEED% > runs\qmix_final_stdout.log 2>&1 & echo done > runs\.qmix_done"

echo.
echo [2/4] Uc egitim de bitene kadar bekleniyor...
echo       Ilerlemeyi ayri bir pencerede izlemek icin:
echo         type runs\iql_final_stdout.log
echo         type runs\vdn_final_stdout.log
echo         type runs\qmix_final_stdout.log
echo.

set /a WAITED=0
:WAIT_LOOP
if exist runs\.iql_done  if exist runs\.vdn_done  if exist runs\.qmix_done goto :ALL_DONE
REM "timeout" konsolsuz/yonlendirilmis stdin ortaminda ("Input redirection
REM is not supported") hemen hata verip bekleme YAPMADAN cikiyor -- ping,
REM konsol olmadan da guvenilir calisan klasik bir gecikme yontemi.
ping -n 31 127.0.0.1 > nul
set /a WAITED+=30
set /a MIN=%WAITED%/60
echo   ... hala bekleniyor (%MIN% dakika gecti)
goto :WAIT_LOOP

:ALL_DONE
echo.
echo [3/4] Uc egitim de tamamlandi. Final karsilastirma tablosu uretiliyor...
%VENV% -m eval.evaluate --vdn-tag vdn_final --qmix-tag qmix_final --iql-tag iql_final

echo.
echo [4/4] Final grafikler uretiliyor...
%VENV% -m viz.plot_iql_report --final

echo.
echo ============================================================
echo  TAMAMLANDI - %date% %time%
echo  Sonuclar:
echo    runs\eval_report.md
echo    runs\viz\final_algorithm_comparison.png
echo    runs\viz\final_success_rate_stacked.png
echo    runs\viz\iql_final_demo_grids.png
echo    runs\viz\vdn_final_demo_grids.png
echo    runs\viz\qmix_final_demo_grids.png
echo ============================================================
