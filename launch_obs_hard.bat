@echo off
REM YENI ENGEL SISTEMI (env/obstacles.py) ile HARD zorlukta 3 algoritma.
REM Eski duvar sisteminden farkli tag: *_obs_hard (karistirmamak icin).
REM Egitilmis nowall modelleri uzerinden ince ayar (--init-from *_nowall_bfs);
REM gozlem boyutu (898) degismedigi icin o checkpointler hala uyumlu.
REM Bittiginde eval raporu OTOMATIK uretilir.
cd /d "%~dp0"

set VENV=.venv\Scripts\python.exe
set EPISODES=%1
if "%EPISODES%"=="" set EPISODES=6000
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4

for %%A in (iql vdn qmix) do (
  if exist runs\.%%A_obs_hard_done del /q runs\.%%A_obs_hard_done
)

echo ============================================================
echo  Yeni engel sistemi ^| HARD ^| %EPISODES% episode
echo  Baslangic: %date% %time%
echo ============================================================

for %%A in (iql vdn qmix) do (
  start "%%A obs-hard" cmd /k "%VENV% train.py --algo %%A --episodes %EPISODES% --tag %%A_obs_hard --difficulty hard --init-from %%A_nowall_bfs --seed 0 & echo done > runs\.%%A_obs_hard_done"
)

echo.
echo [1/2] 3 egitim baslatildi, bitmesi bekleniyor...

:WAIT_LOOP
if exist runs\.iql_obs_hard_done if exist runs\.vdn_obs_hard_done if exist runs\.qmix_obs_hard_done goto :ALL_DONE
ping -n 61 127.0.0.1 > nul
echo   ... hala bekleniyor (%time%)
goto :WAIT_LOOP

:ALL_DONE
echo.
echo [2/2] Eval raporu uretiliyor...
%VENV% -m eval.evaluate --iql-tag iql_obs_hard --vdn-tag vdn_obs_hard --qmix-tag qmix_obs_hard --difficulty hard
copy /y runs\eval_report.md runs\eval_report_obs_hard.md > nul

echo.
echo ============================================================
REM DIKKAT: asagidaki satirda "-^>" yazmak ZORUNLU. Suslu olmayan bir ">"
REM batch'te YONLENDIRME operatorudur: "echo ... -> dosya.md" ciktiyi o
REM dosyaya YAZAR ve az once uretilen raporun UZERINE gecer (yasandi).
echo  TAMAMLANDI - %date% %time%   -^>  runs\eval_report_obs_hard.md
echo ============================================================
