@echo off
set ROOT=c:\Users\Watar\master_thesis\
set PYTHON="%ROOT%.venv\Scripts\python.exe"

echo Running baseline...
%PYTHON% -m sim.demo.demo_06_wall_effect --seconds 15 --hz 240 --physics-hz 240 ^
  --wall-orient side --wall-y 0.7 --wall-zone-x 3.0 --wall-len-x 4.0 --wall-range 1.5 ^
  --x-vel 0.8 --wall-axis roll --wall-model fixed_tau --wall-tau-fixed 0.2 ^
  --kp-xy 2.76 --kd-xy 2.4 --ki-xy 0.425 ^
  --morph-seconds 0 --phi-deg 0 --psi-deg 0 --theta-deg 0 --morph-symmetry mirror_xy ^
  --z-des 0.5 ^
  --log-csv out/demo06_baseline.csv --energy --energy-t0 0.0

echo Running fixed_morph...
%PYTHON% -m sim.demo.demo_06_wall_effect --seconds 15 --hz 240 --physics-hz 240 ^
  --wall-orient side --wall-y 0.7 --wall-zone-x 3.0 --wall-len-x 4.0 --wall-range 1.5 ^
  --x-vel 0.8 --wall-axis roll --wall-model fixed_tau --wall-tau-fixed 0.2 ^
  --kp-xy 2.76 --kd-xy 2.4 --ki-xy 0.425 ^
  --morph-seconds 0 --phi-deg 0 --psi-deg 11 --theta-deg 11 --morph-symmetry mirror_xy ^
  --z-des 0.5 ^
  --log-csv out/demo06_fixed_morph.csv --energy --energy-t0 0.0

echo Running inflight_morph...
%PYTHON% -m sim.demo.demo_06_wall_effect --seconds 15 --hz 240 --physics-hz 240 ^
  --wall-orient side --wall-y 0.7 --wall-zone-x 3.0 --wall-len-x 4.0 --wall-range 1.5 ^
  --x-vel 0.8 --wall-axis roll --wall-model fixed_tau --wall-tau-fixed 0.2 ^
  --kp-xy 2.76 --kd-xy 2.4 --ki-xy 0.425 ^
  --morph-start 2.0 --morph-seconds 1.5 --phi-start 0 --psi-start 0 --theta-start 0 --phi-deg 0 --psi-deg 11 --theta-deg 11 --morph-symmetry mirror_xy ^
  --z-des 0.5 ^
  --log-csv out/demo06_inflight_morph.csv --energy --energy-t0 0.0

echo Done.
