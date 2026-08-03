@echo off
REM ============================================================
REM  9'LU TOPLU KOSUM: IQL + VDN + QMIX  x  easy + medium + hard
REM
REM  Kullanim:  run_all_bfs.bat          -> 4000 episode (varsayilan)
REM             run_all_bfs.bat 6000     -> 6000 episode
REM
REM  Her kosu EGITILMIS nowall modeli uzerinden ince ayar yapar
REM  (--init-from *_nowall_bfs). O checkpointler yeni gozlem boyutuna
REM  agirlik cerrahisiyle tasindi ve DOGRULANDI (duvarsiz basari:
REM  IQL %97, VDN %84, QMIX %59.5 -- orijinalleriyle ayni).
REM
REM  SIFIRDAN baslatmiyoruz: onceki denemede duvarli ortamda sifirdan
REM  baslayan ajan hedefi hic bulamayip timeout dengesine kilitlendi
REM  (butun Q degerleri -17 = timeout cezasi + adim maliyeti).
REM
REM  Bittiginde 3 zorluk icin de eval raporu OTOMATIK uretilir.
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

set VENV=.venv\Scripts\python.exe
set SEED=0
set EPISODES=%1
if "%EPISODES%"=="" set EPISODES=4000

REM 9 surec x cok-is-parcacikli PyTorch, 10 cekirdek uzerinde birbirini
REM bogar. Surec basina 2 is parcacigi toplam verimi yukseltiyor.
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2

if not exist runs mkdir runs
for %%D in (easy medium hard) do (
  for %%A in (iql vdn qmix) do (
    if exist runs\.%%A_bfs_%%D_done del /q runs\.%%A_bfs_%%D_done
  )
)

echo ============================================================
echo  9'lu toplu kosum  ^| %EPISODES% episode  ^| seed %SEED%
echo  Baslangic: %date% %time%
echo ============================================================
echo.

for %%D in (easy medium hard) do (
  for %%A in (iql vdn qmix) do (
    start "%%A %%D" cmd /k "%VENV% train.py --algo %%A --episodes %EPISODES% --tag %%A_bfs_%%D --difficulty %%D --init-from %%A_nowall_bfs --seed %SEED% & echo done > runs\.%%A_bfs_%%D_done"
  )
)

echo [1/2] 9 egitim baslatildi, her biri kendi penceresinde.
echo       Bitmesi bekleniyor (saatler surebilir)...
echo.

:WAIT_LOOP
set READY=1
for %%D in (easy medium hard) do (
  for %%A in (iql vdn qmix) do (
    if not exist runs\.%%A_bfs_%%D_done set READY=0
  )
)
if "!READY!"=="1" goto :ALL_DONE
ping -n 61 127.0.0.1 > nul
echo   ... hala bekleniyor (%time%)
goto :WAIT_LOOP

:ALL_DONE
echo.
echo [2/2] Hepsi bitti. Zorluk basina eval raporu uretiliyor...
for %%D in (easy medium hard) do (
  echo   --- %%D ---
  %VENV% -m eval.evaluate --iql-tag iql_bfs_%%D --vdn-tag vdn_bfs_%%D --qmix-tag qmix_bfs_%%D --difficulty %%D
  copy /y runs\eval_report.md runs\eval_report_bfs_%%D.md > nul
)

echo.
echo ============================================================
echo  TAMAMLANDI - %date% %time%
echo    runs\eval_report_bfs_easy.md
echo    runs\eval_report_bfs_medium.md
echo    runs\eval_report_bfs_hard.md
echo ============================================================
