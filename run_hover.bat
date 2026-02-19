@echo off
REM Activate venv and run hover demo (1.8 kg morphing drone). Use --gui to watch.
set ROOT=%~dp0
"%ROOT%.venv\Scripts\python.exe" -m sim.demo.demo_04_hover_pd_urdf_morph %*
if errorlevel 1 pause
