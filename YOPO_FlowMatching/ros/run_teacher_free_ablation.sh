#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="/home/hwm/CF_YOPO/YOPO_FlowMatching"
OUTPUT_ROOT="${PROJECT_ROOT}/runs/icra_ros_teacher_free"
CHECKPOINT="${PROJECT_ROOT}/runs/icra2027_teacher_free_physical_seed0/checkpoints/best.pt"
COLLISION_MAP="${PROJECT_ROOT}/runs/icra_ros/maps/seed3.ply"
LAUNCH_FILE="${PROJECT_ROOT}/ros/icra_simulator_attitude_control.launch"

source /opt/ros/noetic/setup.bash
source /workspace/YOPO/Controller/devel/setup.bash
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
set -u

mkdir -p "${OUTPUT_ROOT}/logs"
owned_master=0
owned_sensor=0

cleanup_controller() {
  rosnode kill /quadrotor_simulator_so3 /network_controller_node >/dev/null 2>&1 || true
  if [[ -n "${controller_pid:-}" ]]; then
    kill "${controller_pid}" >/dev/null 2>&1 || true
    wait "${controller_pid}" 2>/dev/null || true
    controller_pid=""
  fi
}

cleanup_all() {
  cleanup_controller
  if [[ "${owned_sensor}" == "1" ]]; then
    rosnode kill /sensor_simulator_node >/dev/null 2>&1 || true
    kill "${sensor_pid:-}" >/dev/null 2>&1 || true
  fi
  if [[ "${owned_master}" == "1" ]]; then
    kill "${master_pid:-}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_all EXIT INT TERM

if ! rosnode list >/dev/null 2>&1; then
  roscore >"${OUTPUT_ROOT}/logs/roscore.log" 2>&1 &
  master_pid=$!
  owned_master=1
  for _ in $(seq 1 50); do
    rosnode list >/dev/null 2>&1 && break
    sleep 0.1
  done
fi

if ! rosnode list 2>/dev/null | grep -q '^/sensor_simulator_node$'; then
  (cd /workspace/YOPO/Simulator && \
    source /opt/ros/noetic/setup.bash && \
    source devel/setup.bash && \
    exec rosrun sensor_simulator sensor_simulator_cuda) \
    >"${OUTPUT_ROOT}/logs/sensor_simulator.log" 2>&1 &
  sensor_pid=$!
  owned_sensor=1
  sleep 3
fi

run_trial() {
  local goal_id="$1"
  local gx="$2"
  local gy="$3"
  local gz="$4"
  local variant="$5"
  local selector_args=()
  if [[ "${variant}" == "teacher_free_raw" ]]; then
    selector_args+=(--disable-continuity-selector)
  fi

  local trial_dir="${OUTPUT_ROOT}/seed3_${goal_id}"
  mkdir -p "${trial_dir}"
  cleanup_controller
  sleep 0.5

  (cd /workspace/YOPO/Controller && \
    source /opt/ros/noetic/setup.bash && \
    source devel/setup.bash && \
    exec roslaunch "${LAUNCH_FILE}" init_x:=2 init_y:=-2 init_z:=2) \
    >"${trial_dir}/${variant}_controller.log" 2>&1 &
  controller_pid=$!

  timeout 15 rostopic echo -n 1 /sim/odom >/dev/null 2>&1
  timeout 15 rostopic echo -n 1 /depth_image >/dev/null 2>&1

  echo "Running ${goal_id}/${variant}: goal=(${gx}, ${gy}, ${gz})"
  python3 "${PROJECT_ROOT}/ros/test_policy_icra_ros_matched.py" \
    --policy latticeflow \
    --checkpoint "${CHECKPOINT}" \
    --goal "${gx}" "${gy}" "${gz}" \
    --collision-map "${COLLISION_MAP}" \
    --no-visualize --arrival-distance 1 --max-runtime 60 \
    --stop-on-arrival --stop-on-collision \
    "${selector_args[@]}" \
    --result-md "${trial_dir}/${variant}.md" \
    --result-json "${trial_dir}/${variant}.json" \
    >"${trial_dir}/${variant}_policy.log" 2>&1

  cleanup_controller
  sleep 0.5
}

goals=(
  "g1 22 -2 2"
  "g2 20 4 2"
  "g3 18 12 2"
  "g4 18 -14 2"
  "g5 16 16 2"
)

for variant in teacher_free_raw teacher_free_selector; do
  for goal in "${goals[@]}"; do
    read -r goal_id gx gy gz <<<"${goal}"
    run_trial "${goal_id}" "${gx}" "${gy}" "${gz}" "${variant}"
  done
done

python3 "${PROJECT_ROOT}/ros/summarize_icra_ros.py" \
  --root "${OUTPUT_ROOT}" \
  --output-md "${OUTPUT_ROOT}/ROS_SUMMARY.md" \
  --output-json "${OUTPUT_ROOT}/ROS_SUMMARY.json"

echo "Teacher-free ROS ablation complete: ${OUTPUT_ROOT}/ROS_SUMMARY.md"
