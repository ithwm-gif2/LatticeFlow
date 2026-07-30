#!/usr/bin/env bash
set -euo pipefail

SIM_ROOT="${SIM_ROOT:-/workspace/YOPO/Simulator}"
OUTPUT_PATH="${OUTPUT_PATH:-/workspace/YOPO/dataset_mid360}"
CAMERA_PITCH_DEG="${CAMERA_PITCH_DEG:--22.5}"
ENV_NUM="${ENV_NUM:-10}"
IMAGE_NUM="${IMAGE_NUM:-10000}"
CONFIG_PATH="${SIM_ROOT}/src/config/config.yaml"
GENERATOR="${SIM_ROOT}/devel/lib/sensor_simulator/dataset_generator"

if [[ -e "${OUTPUT_PATH}" && "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
  echo "Refusing to overwrite existing dataset: ${OUTPUT_PATH}" >&2
  echo "Set ALLOW_OVERWRITE=1 only when replacement is intentional." >&2
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" || ! -x "${GENERATOR}" ]]; then
  echo "Simulator config or generator is missing under ${SIM_ROOT}" >&2
  exit 2
fi

source /opt/ros/noetic/setup.bash
source "${SIM_ROOT}/devel/setup.bash"
backup="$(mktemp /tmp/yopo_mid360_config.XXXXXX.yaml)"
cp "${CONFIG_PATH}" "${backup}"
restore_config() {
  cp "${backup}" "${CONFIG_PATH}"
  rm -f "${backup}"
}
trap restore_config EXIT INT TERM

sed   -e "s|^  pitch:.*|  pitch: ${CAMERA_PITCH_DEG}           # camera up is negative in Simulator|"   -e "s|^save_path:.*|save_path: "${OUTPUT_PATH}/"|"   -e "s|^env_num:.*|env_num: ${ENV_NUM}|"   -e "s|^image_num:.*|image_num: ${IMAGE_NUM}|"   "${backup}" > "${CONFIG_PATH}"

cd "${SIM_ROOT}"
"${GENERATOR}"
