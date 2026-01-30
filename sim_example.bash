python -m sim.demo.demo_06_wall_effect --gui --cam-follow --seconds 15 --hz 240 --physics-hz 240 \
  --wall-orient side --wall-y 1.0 --wall-zone-x 3.0 --wall-len-x 4.0 --wall-range 1.5 \
  --x-vel 0.8 --wall-axis roll --wall-k 0.005 --wall-d0 0.3 \
  --morph-start 2.0 --morph-seconds 1.5 --phi-deg 5 --psi-deg 7 --theta-deg 30 --morph-symmetry mirror_xy \
  --cam-yaw-mode behind --cam-yaw-offset 10 --cam-dist 1.2 --cam-pitch -10 --cam-target-body 0.3 0 0.35 \
  --z-des 0.5 --win-width 2560 --win-height 1440 --record-mp4 demo_06_wall_effect_morphing.mp4

python -m sim.demo.demo_06_wall_effect --gui --cam-follow --seconds 15 --hz 240 --physics-hz 240 \
  --wall-orient side --wall-y 1.0 --wall-zone-x 3.0 --wall-len-x 4.0 --wall-range 1.5 \
  --x-vel 0.8 --wall-axis roll --wall-k 0.02 --wall-d0 0.3 \
  --morph-start 2.0 --morph-seconds 1.5 --phi-deg 0 --psi-deg 0 --theta-deg 0 --morph-symmetry mirror_xy \
  --cam-yaw-mode behind --cam-yaw-offset 10 --cam-dist 1.2 --cam-pitch -10 --cam-target-body 0.3 0 0.35 \
  --z-des 0.5 --cam-freeze-on-contact --win-width 2560 --win-height 1440 --record-mp4 demo_06_wall_effect_baseline.mp4