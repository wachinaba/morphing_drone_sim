python -m sim.demo.demo_06_wall_effect --gui --cam-follow --seconds 15 --hz 240 --physics-hz 240 \
  --wall-orient side --wall-y 0.7 --wall-zone-x 3.0 --wall-len-x 4.0 --wall-range 1.5 \
  --x-vel 0.8 --wall-axis roll --wall-model fixed_tau --wall-tau-fixed 0.2 \
  --kp-xy 2.76 --kd-xy 2.4 --ki-xy 0.425 \
  --morph-start 2.0 --morph-seconds 1.5 --phi-deg 5 --psi-deg 7 --theta-deg 30 --morph-symmetry mirror_xy \
  --cam-yaw-mode behind --cam-yaw-offset 10 --cam-dist 1.2 --cam-pitch -10 --cam-target-body 0.3 0 0.35 \
  --z-des 0.5 --win-width 2560 --win-height 1440 --record-mp4 demo_06_wall_effect_morphing.mp4 --plot

# Energy evaluation (3 conditions): baseline, fixed_morph, inflight_morph. Omit --gui for batch; add --gui for visual check.
# All use wall_tau_fixed=0.2 for same wall disturbance.
python -m sim.demo.demo_06_wall_effect --seconds 15 --hz 240 --physics-hz 240 \
  --wall-orient side --wall-y 0.7 --wall-zone-x 3.0 --wall-len-x 4.0 --wall-range 1.5 \
  --x-vel 0.8 --wall-axis roll --wall-model fixed_tau --wall-tau-fixed 0.2 \
  --kp-xy 2.76 --kd-xy 2.4 --ki-xy 0.425 \
  --morph-seconds 0 --phi-deg 0 --psi-deg 0 --theta-deg 0 --morph-symmetry mirror_xy \
  --z-des 0.5 \
  --log-csv out/demo06_baseline.csv --energy --energy-t0 0.0 --plot

python -m sim.demo.demo_06_wall_effect --seconds 15 --hz 240 --physics-hz 240 \
  --wall-orient side --wall-y 0.7 --wall-zone-x 3.0 --wall-len-x 4.0 --wall-range 1.5 \
  --x-vel 0.8 --wall-axis roll --wall-model fixed_tau --wall-tau-fixed 0.2 \
  --kp-xy 2.76 --kd-xy 2.4 --ki-xy 0.425 \
  --morph-seconds 0 --phi-deg 0 --psi-deg 11 --theta-deg 11 --morph-symmetry mirror_xy \
  --z-des 0.5 \
  --log-csv out/demo06_fixed_morph.csv --energy --energy-t0 0.0 --plot

python -m sim.demo.demo_06_wall_effect --seconds 15 --hz 240 --physics-hz 240 \
  --wall-orient side --wall-y 0.7 --wall-zone-x 3.0 --wall-len-x 4.0 --wall-range 1.5 \
  --x-vel 0.8 --wall-axis roll --wall-model fixed_tau --wall-tau-fixed 0.2 \
  --kp-xy 2.76 --kd-xy 2.4 --ki-xy 0.425 \
  --morph-start 2.0 --morph-seconds 1.5 --phi-start 0 --psi-start 0 --theta-start 0 --phi-deg 0 --psi-deg 11 --theta-deg 11 --morph-symmetry mirror_xy \
  --z-des 0.5 \
  --log-csv out/demo06_inflight_morph.csv --energy --energy-t0 0.0 --plot

# Wall-tau sweep: find boundary where morphing becomes energy-advantageous (output: out/sweep/wall_tau_sweep_*.png, *.csv, *_summary.md)
# python scripts/demo06_wall_tau_sweep.py --tau-min 0.05 --tau-max 0.35 --tau-step 0.025 --out-dir out/sweep

# Record MP4 at crossover (effective_tau=0.5): out/sweep/crossover_baseline.mp4, crossover_fixed_morph.mp4, crossover_inflight_morph.mp4
# python scripts/record_crossover_mp4.py

python -m sim.demo.demo_06_wall_effect --gui --cam-follow --seconds 15 --hz 240 --physics-hz 240 \
  --wall-orient side --wall-y 0.5 --wall-zone-x 3.0 --wall-len-x 4.0 --wall-range 1.5 \
  --x-vel 0.8 --wall-axis roll --wall-model fixed_tau --wall-tau-fixed 0.1 \
  --kp-xy 2.76 --kd-xy 2.4 --ki-xy 0.425 \
  --morph-start 2.0 --morph-seconds 1.5 --phi-deg 0 --psi-deg 0 --theta-deg 0 --morph-symmetry mirror_xy \
  --cam-yaw-mode behind --cam-yaw-offset 0 --cam-dist 1.2 --cam-pitch -10 --cam-target-body 0.3 0 0.35 \
  --z-des 0.5 --cam-freeze-on-contact --win-width 2560 --win-height 1440 --record-mp4 demo_06_wall_effect_baseline.mp4

  python -m sim.demo.demo_06_wall_effect --gui --cam-follow --seconds 30 --hz 240 --physics-hz 240 \
  --wall-orient side --wall-y 1.0 --wall-zone-x 3.0 --wall-len-x 4.0 --wall-range 1.5 \
  --x-vel 0.8 --wall-axis roll --wall-k 2 --wall-d0 0.3 --wall-model fixed_tau --wall-tau-fixed 0.2 --ki-xy 1.0 \
  --kp-xy 2.76 --kd-xy 2.4 --ki-xy 0.425 --i-xy-max 5.0 \
  --draw-trajectory --draw-trajectory-every 30 \
  --morph-start 2.0 --morph-seconds 1.5 --phi-deg 0 --psi-deg 0 --theta-deg 0 --morph-symmetry mirror_xy \
  --cam-yaw-mode behind --cam-yaw-offset 10 --cam-dist 1.2 --cam-pitch -10 --cam-target-body 0.3 0 0.35 \
  --z-des 0.5 --cam-freeze-on-contact --win-width 2560 --win-height 1440 --record-mp4 demo_06_wall_effect_baseline.mp4

# 軌道追従（円）: --wall-y 1.5 で壁衝突回避。--draw-trajectory で目標(緑)と実際(青)を線表示
python -m sim.demo.demo_06_wall_effect --gui --cam-follow --draw-trajectory --draw-trajectory-every 15 --seconds 30 --traj circle --circle-radius 0.8 --circle-freq 0.2 --circle-cx 8 --circle-cy 0 \
  --wall-orient side --wall-y 1.5 --wall-zone-x 3.0 --wall-len-x 4.0 --wall-range 1.5 --kp-xy 2.76 --kd-xy 2.4 --ki-xy 0.425 \
  --morph-start 2.0 --morph-seconds 1.5 --phi-deg 5 --psi-deg 7 --theta-deg 30 --morph-symmetry mirror_xy \
  --cam-dist 1.2 --cam-pitch -10 --cam-target-body 0.3 0 0.35 --z-des 0.5

#軌道追従（矩形）: 同様
python -m sim.demo.demo_06_wall_effect --gui --cam-follow --draw-trajectory --seconds 15 --traj rect --rect-width 1.6 --rect-height 1.2 --rect-speed 0.9 --rect-cx 5 --rect-cy 0 \
  --wall-orient side --wall-y 1.5 --wall-zone-x 3.0 --wall-len-x 4.0 --wall-range 1.5 --kp-xy 2.76 --kd-xy 2.4 --ki-xy 0.425 \
  --morph-start 2.0 --morph-seconds 1.5 --phi-deg 5 --psi-deg 7 --theta-deg 30 --morph-symmetry mirror_xy \
  --cam-yaw-mode behind_vel --cam-dist 1.2 --cam-pitch -10 --cam-target-body 0.3 0 0.35 --z-des 0.5


python -m sim.demo.demo_06_wall_effect --gui --draw-trajectory --draw-trajectory-every 15 --traj-line-width-des 4.0 --traj-line-width-act 4.0 --draw-target-marker --target-marker-size 0.04 --seconds 20 \
  --traj rect --rect-width 1.6 --rect-height 1.2 --rect-speed 0.8 --rect-cx 2.0 --rect-cy 0.0 --rect-phase 0.0 --yaw-spin-deg-s 120 \
  --wall-model off --wall-orient side --wall-zone-x 1000 --wall-len-x 1.0 --wall-y 1000 --wall-range 0.1 \
  --kp-xy 2.76 --kd-xy 2.4 --ki-xy 0.425 \
  --morph-seconds 0 --phi-deg 2 --psi-deg 0 --theta-deg 0 --morph-symmetry mirror_xy \
  --phi-amp 10 --phi-freq 0.5 --psi-amp 10 --psi-freq 0.5 --theta-amp 30 --theta-freq 0.5 \
  --cam-dist 2.8 --cam-pitch -30 --cam-target-mode traj_center \
  --z-des 0.5 \
  --win-width 2560 --win-height 1440 --record-mp4 demo_06_wall_effect_morphing_trajectory_rect.mp4 --record-fps 60 

  python -m sim.demo.demo_06_wall_effect --gui --draw-trajectory --draw-trajectory-every 15 --traj-line-width-des 4.0 --traj-line-width-act 4.0 --draw-target-marker --target-marker-size 0.04 --seconds 20 \
  --traj rect --rect-width 1.6 --rect-height 1.2 --rect-speed 0.8 --rect-cx 2.0 --rect-cy 0.0 --rect-phase 0.0 --yaw-spin-deg-s 120 \
  --wall-model off --wall-orient side --wall-zone-x 1000 --wall-len-x 1.0 --wall-y 1000 --wall-range 0.1 \
  --kp-xy 2.76 --kd-xy 2.4 --ki-xy 0.425 \
  --morph-seconds 0 --phi-deg 0 --psi-deg 0 --theta-deg 0 --morph-symmetry mirror_xy \
  --cam-dist 2.8 --cam-pitch -30 --cam-target-mode traj_center \
  --z-des 0.5 \
  --win-width 2560 --win-height 1440 --record-mp4 demo_06_wall_effect_baseline_trajectory_rect.mp4 --record-fps 60 --plot