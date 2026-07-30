#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="/home/hwm/CF_YOPO/YOPO_FlowMatching"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs/runtime6_goal10_evaluation/docker_mid360_vlines32}"
CHECKPOINT="${PROJECT_ROOT}/runs/icra2027_teacher_free_physical_mid360_seed0/checkpoints/best.pt"
COLLISION_MAP="${PROJECT_ROOT}/runs/icra_ros/maps/seed3.ply"
LAUNCH_FILE="${PROJECT_ROOT}/ros/icra_simulator_attitude_control.launch"
SIM_CONFIG="/workspace/YOPO/Simulator/src/config/config.yaml"

source /opt/ros/noetic/setup.bash
source /workspace/YOPO/Controller/devel/setup.bash
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
set -u

mkdir -p "${OUTPUT_ROOT}/logs"
config_backup="$(mktemp /tmp/yopo_sim_config.runtime6.XXXXXX.yaml)"
cp "${SIM_CONFIG}" "${config_backup}"

master_pid=""
sensor_pid=""
adapter_pid=""
controller_pid=""

cleanup_controller() {
  rosnode kill /quadrotor_simulator_so3 /network_controller_node >/dev/null 2>&1 || true
  if [[ -n "${controller_pid}" ]]; then
    kill -INT "${controller_pid}" >/dev/null 2>&1 || true
    wait "${controller_pid}" 2>/dev/null || true
    controller_pid=""
  fi
}

cleanup() {
  set +e
  cleanup_controller
  rosnode kill /sim_mid360_depth_adapter /sensor_simulator_node >/dev/null 2>&1
  [[ -n "${adapter_pid}" ]] && kill -INT "${adapter_pid}" >/dev/null 2>&1
  [[ -n "${sensor_pid}" ]] && kill -INT "${sensor_pid}" >/dev/null 2>&1
  [[ -n "${master_pid}" ]] && kill -INT "${master_pid}" >/dev/null 2>&1
  wait "${adapter_pid}" "${sensor_pid}" "${master_pid}" 2>/dev/null
  cp "${config_backup}" "${SIM_CONFIG}"
  rm -f "${config_backup}"
}
trap cleanup EXIT INT TERM

sed -i \
  -e 's/^  vertical_lines:.*/  vertical_lines: 32/' \
  -e 's/^  vertical_angle_start:.*/  vertical_angle_start: -7.0/' \
  -e 's/^  vertical_angle_end:.*/  vertical_angle_end: 52.0/' \
  "${SIM_CONFIG}"

roscore >"${OUTPUT_ROOT}/logs/roscore.log" 2>&1 &
master_pid=$!
for _ in $(seq 1 100); do
  rosnode list >/dev/null 2>&1 && break
  sleep 0.1
done

(
  cd /workspace/YOPO/Simulator
  source /opt/ros/noetic/setup.bash
  source devel/setup.bash
  exec rosrun sensor_simulator sensor_simulator_cuda
) >"${OUTPUT_ROOT}/logs/sensor_simulator.log" 2>&1 &
sensor_pid=$!

python3 "${PROJECT_ROOT}/ros/sim_mid360_depth_adapter.py" \
  >"${OUTPUT_ROOT}/logs/mid360_adapter.log" 2>&1 &
adapter_pid=$!
sleep 3

run_trial() {
  local goal_id="$1"
  local gx="$2"
  local gy="$3"
  local velocity="$4"
  local variant="runtime${velocity%.*}"
  local trial_dir="${OUTPUT_ROOT}/${goal_id}/${variant}"
  mkdir -p "${trial_dir}"
  cleanup_controller
  sleep 0.5

  (
    cd /workspace/YOPO/Controller
    source /opt/ros/noetic/setup.bash
    source devel/setup.bash
    exec roslaunch "${LAUNCH_FILE}" init_x:=2 init_y:=-2 init_z:=1.5
  ) >"${trial_dir}/controller.log" 2>&1 &
  controller_pid=$!

  timeout 15 rostopic echo -n 1 /sim/odom >/dev/null
  timeout 15 rostopic echo -n 1 /mid360_depth_image >/dev/null
  python3 "${PROJECT_ROOT}/ros/test_policy_icra_ros_matched.py" \
    --policy latticeflow \
    --checkpoint "${CHECKPOINT}" \
    --runtime-velocity "${velocity}" \
    --goal "${gx}" "${gy}" 1.5 \
    --depth-topic /mid360_depth_image \
    --collision-map "${COLLISION_MAP}" \
    --disable-continuity-selector \
    --no-visualize \
    --arrival-distance 1.0 \
    --max-runtime 45 \
    --stop-on-arrival \
    --stop-on-collision \
    --result-md "${trial_dir}/result.md" \
    --result-json "${trial_dir}/result.json" \
    >"${trial_dir}/policy.log" 2>&1
  cleanup_controller
  sleep 0.5
}

# All goals are exactly 10 m horizontally from start [2,-2,1.5].
goals=(
  "forward 12.000 -2.000"
  "left22 11.239 1.827"
  "left45 9.071 5.071"
  "right22 11.239 -5.827"
  "right45 9.071 -9.071"
)

for goal in "${goals[@]}"; do
  read -r goal_id gx gy <<<"${goal}"
  run_trial "${goal_id}" "${gx}" "${gy}" 4.0
  run_trial "${goal_id}" "${gx}" "${gy}" 6.0
done

python3 - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/runtime*/result.json")):
    data = json.loads(path.read_text())
    data["trial"] = path.parent.parent.name
    rows.append(data)
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")

header = (
    "| Trial | Runtime | Reached | Collision | Time (s) | Mean speed (m/s) | "
    "Min geometry (m) | P95 jerk (m/s^3) | Switch rate |\n"
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
)
lines = ["# Runtime 4/6 MID-360 Goal-10 Closed-loop", "", header.rstrip()]
for row in rows:
    lines.append(
        f"| {row['trial']} | {row['runtime_velocity_mps']:.0f} | "
        f"{row['goal_reached']} | {row['geometry_collision']} | "
        f"{row['time_to_arrival_s'] if row['time_to_arrival_s'] is not None else float('nan'):.3f} | "
        f"{row['mean_speed_mps']:.3f} | {row['minimum_geometry_distance_m']:.3f} | "
        f"{row['p95_command_jerk_mps3']:.3f} | {row['lattice_switch_rate']:.4f} |"
    )
(root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
PY

echo "Results: ${OUTPUT_ROOT}/SUMMARY.md"
