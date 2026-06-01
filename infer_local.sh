#!/bin/bash
# One-click reproduction script for Ditto local editing.
# Steps: download LoRA -> download test video -> detect GPUs -> run prompts in parallel.
set -e

# ---------- Paths & config ----------
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_ROOT}"

# Remote LoRA on HuggingFace: repo "QingyanBai/Ditto_models", file "models/ditto_local.safetensors"
LORA_REPO="QingyanBai/Ditto_models"
LORA_REMOTE_FILE="models/ditto_local.safetensors"
LORA_DIR="${REPO_ROOT}/lora_models"
LORA_PATH="${LORA_DIR}/$(basename "${LORA_REMOTE_FILE}")"

# Remote test video on HuggingFace dataset "QingyanBai/Ditto-1M"
DATA_REPO="QingyanBai/Ditto-1M"
VIDEO_REL_PATH="mini_test_videos/plane.mp4"
DATA_DIR="${REPO_ROOT}/data"
INPUT_VIDEO="${DATA_DIR}/${VIDEO_REL_PATH}"

# Inference hyperparameters (match training resolution & frame count)
OUTPUT_DIR="${REPO_ROOT}/results"
NUM_FRAMES=73
FPS=16
SEED=42
HEIGHT=480
WIDTH=832

# Local temp dir (avoid /tmp, which often lacks write permission on shared hosts)
LOCAL_TMP_DIR="${REPO_ROOT}/tmp"

# Output filename = "<source video stem>_<first N chars of prompt>.mp4".
# Linux NAME_MAX is 255 bytes and Windows caps a filename at ~255 chars too,
# so we keep the prompt slice short. 50 chars is descriptive but leaves
# plenty of headroom even for UTF-8 prompts (each char up to ~4 bytes).
SOURCE_STEM="$(basename "${INPUT_VIDEO%.*}")"
PROMPT_SLUG_LEN=50

# Prompts to evaluate (order matches data/local_prompts.txt)
PROMPTS=(
    "Add a vivid rainbow-colored contrail trailing behind the airplane, stretching across the sky"
    "Change the airplane's body color to bright red with a golden stripe along the fuselag"
    "Change the background to a dramatic sunset sky filled with layers of orange, pink, and purple clouds"
    "Change the background to outer space with stars and the Earth visible below"
    "Remove the plane in the sky"
)

mkdir -p "${LORA_DIR}" "${DATA_DIR}" "${OUTPUT_DIR}" "${LOCAL_TMP_DIR}"

# Redirect TMPDIR so huggingface-cli (and anything else) writes temp files under
# the repo, instead of /tmp which is often non-writable on shared machines.
export TMPDIR="${LOCAL_TMP_DIR}"

# ---------- Ensure huggingface-cli is available ----------
if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "[setup] installing huggingface_hub ..."
    pip install -q -U "huggingface_hub[cli]"
fi

# ---------- Download LoRA checkpoint ----------
# Pull the single file "models/ditto_local.safetensors" and flatten it into LORA_DIR.
if [ ! -f "${LORA_PATH}" ]; then
    echo "[download] LoRA: ${LORA_REPO}/${LORA_REMOTE_FILE}"
    TMP_LORA_DIR="$(mktemp -d -p "${LOCAL_TMP_DIR}")"
    huggingface-cli download "${LORA_REPO}" "${LORA_REMOTE_FILE}" \
        --repo-type model --local-dir "${TMP_LORA_DIR}"
    mv "${TMP_LORA_DIR}/${LORA_REMOTE_FILE}" "${LORA_PATH}"
    rm -rf "${TMP_LORA_DIR}"
else
    echo "[skip] LoRA already exists: ${LORA_PATH}"
fi

# ---------- Download test video ----------
if [ ! -f "${INPUT_VIDEO}" ]; then
    echo "[download] Video: ${DATA_REPO}/${VIDEO_REL_PATH}"
    huggingface-cli download "${DATA_REPO}" "${VIDEO_REL_PATH}" \
        --repo-type dataset --local-dir "${DATA_DIR}"
else
    echo "[skip] Video already exists: ${INPUT_VIDEO}"
fi

# ---------- Detect available GPUs ----------
if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NUM_GPUS=0
fi
NUM_PROMPTS=${#PROMPTS[@]}
echo "[info] detected ${NUM_GPUS} GPU(s), ${NUM_PROMPTS} prompts"

if [ "${NUM_GPUS}" -lt 1 ]; then
    echo "[error] no GPU detected"
    exit 1
fi

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
echo "[info] per-prompt logs will be written to: ${LOG_DIR}"
echo "[info] tail a log live with: tail -f ${LOG_DIR}/prompt_<idx>_gpu<id>.log"

# ---------- Launcher: one inference job on one GPU ----------
declare -a PIDS=()
launch_job() {
    local gpu_id=$1
    local idx=$2
    local prompt=$3
    local log_file="${LOG_DIR}/prompt_${idx}_gpu${gpu_id}.log"
    # Build a filesystem-safe slug from the prompt: lower-case, swap any
    # non [A-Za-z0-9._-] character for "_", squeeze repeats, trim, then cap
    # length. Final name stays well under the 255-byte/char limit on both
    # Linux and Windows.
    local slug
    slug="$(printf '%s' "${prompt}" \
        | tr '[:upper:]' '[:lower:]' \
        | tr -c 'a-z0-9._-' '_' \
        | tr -s '_' \
        | sed 's/^_//; s/_$//')"
    slug="${slug:0:${PROMPT_SLUG_LEN}}"
    slug="${slug%_}"
    local output_video="${OUTPUT_DIR}/${SOURCE_STEM}_${slug}.mp4"
    echo "[run] gpu=${gpu_id} idx=${idx} prompt=\"${prompt:0:100}...\" log=${log_file}"
    python "${REPO_ROOT}/inference/infer_ditto.py" \
        --lora_path "${LORA_PATH}" \
        --input_video "${INPUT_VIDEO}" \
        --output_video "${output_video}" \
        --device_id "${gpu_id}" \
        --num_frames "${NUM_FRAMES}" \
        --fps "${FPS}" \
        --seed "${SEED}" \
        --height "${HEIGHT}" \
        --width "${WIDTH}" \
        --prompt "${prompt}" \
        > "${log_file}" 2>&1 &
    PIDS+=($!)
}

# ---------- Parallel scheduling ----------
# Round-robin prompts across GPUs; wait for each full batch before starting the next.
for ((i=0; i<NUM_PROMPTS; i++)); do
    gpu_id=$((i % NUM_GPUS))
    launch_job "${gpu_id}" "${i}" "${PROMPTS[$i]}"
    if (( (i + 1) % NUM_GPUS == 0 )); then
        for pid in "${PIDS[@]}"; do wait "${pid}" || true; done
        PIDS=()
    fi
done

# Drain the last (possibly partial) batch
for pid in "${PIDS[@]}"; do wait "${pid}" || true; done

echo "[done] results saved to: ${OUTPUT_DIR}"
echo "[done] logs saved to:    ${LOG_DIR}"
