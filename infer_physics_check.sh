#!/bin/bash
# Batch local editing for physics_check videos using prompt.yaml.
# Steps: ensure LoRA -> parse prompt.yaml -> detect GPUs -> run all jobs in parallel.
set -euo pipefail

# ---------- Paths & config ----------
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_ROOT}"

PHYSICS_DIR="/data/shared-vilab/datasets/mj_data/physics_check"
PROMPT_YAML="/data/shared-vilab/datasets/mj_data/physics_check/prompt.yaml"
OUTPUT_DIR="${PHYSICS_DIR}/results"

# Pretrained model paths
PRETRAINED_DIR="/data/shared-vilab/pretrained_models"
DITTO_MODEL_DIR="${PRETRAINED_DIR}/Ditto_models"
WAN_MODEL_DIR="${PRETRAINED_DIR}/Wan-AI"
LORA_PATH="${DITTO_MODEL_DIR}/ditto_local.safetensors"

# Inference hyperparameters (match training resolution & frame count)
NUM_FRAMES=73
FPS=16
SEED=42
HEIGHT=480
WIDTH=832

# Local temp dir (avoid /tmp, which often lacks write permission on shared hosts)
LOCAL_TMP_DIR="${REPO_ROOT}/tmp"

# Output filename = "<src_stem>_L<level>_<prompt slug>.mp4"
PROMPT_SLUG_LEN=50

mkdir -p "${OUTPUT_DIR}" "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"

# ---------- Sanity checks ----------
if [ ! -f "${PROMPT_YAML}" ]; then
    echo "[error] prompt file not found: ${PROMPT_YAML}"
    exit 1
fi

if [ ! -f "${LORA_PATH}" ]; then
    echo "[error] Ditto LoRA not found: ${LORA_PATH}"
    exit 1
fi

if [ ! -d "${WAN_MODEL_DIR}" ]; then
    echo "[error] Wan model directory not found: ${WAN_MODEL_DIR}"
    exit 1
fi

# infer_ditto.py expects: {local_model_path}/Wan-AI/Wan2.1-VACE-14B/
# If weights live directly under Wan-AI/, symlink the expected subfolder.
WAN_MODEL_ROOT="${PRETRAINED_DIR}"
WAN_MODEL_EXPECTED="${WAN_MODEL_ROOT}/Wan-AI/Wan2.1-VACE-14B"
if [ ! -d "${WAN_MODEL_EXPECTED}" ]; then
    if [ -f "${WAN_MODEL_DIR}/Wan2.1_VAE.pth" ] || \
       compgen -G "${WAN_MODEL_DIR}/diffusion_pytorch_model*.safetensors" > /dev/null; then
        mkdir -p "${WAN_MODEL_ROOT}/Wan-AI"
        ln -sfn "${WAN_MODEL_DIR}" "${WAN_MODEL_EXPECTED}"
        echo "[setup] linked ${WAN_MODEL_EXPECTED} -> ${WAN_MODEL_DIR}"
    else
        echo "[error] Wan VACE weights not found under ${WAN_MODEL_DIR} or ${WAN_MODEL_EXPECTED}"
        exit 1
    fi
fi

echo "[info] Ditto LoRA: ${LORA_PATH}"
echo "[info] Wan VACE:   ${WAN_MODEL_EXPECTED}"

# ---------- Parse prompt.yaml into a tab-separated job list ----------
# Columns: idx | src_video | level | instruction
JOB_LIST="$(mktemp -p "${LOCAL_TMP_DIR}" physics_jobs.XXXXXX)"
python3 - "${PROMPT_YAML}" > "${JOB_LIST}" <<'PY'
import sys

prompt_path = sys.argv[1]
entries = []
current = {}

with open(prompt_path, encoding="utf-8") as f:
    for raw_line in f:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- instruction:"):
            if current:
                entries.append(current)
            current = {"instruction": line.split(":", 1)[1].strip()}
        elif line.startswith("src_video:"):
            current["src_video"] = line.split(":", 1)[1].strip()
        elif line.startswith("level:"):
            current["level"] = line.split(":", 1)[1].strip()

if current:
    entries.append(current)

for idx, entry in enumerate(entries):
    for key in ("instruction", "src_video", "level"):
        if key not in entry:
            raise SystemExit(f"[error] entry {idx} missing field: {key}")
    instruction = entry["instruction"].replace("\t", " ").replace("\n", " ")
    print(f"{idx}\t{entry['src_video']}\t{entry['level']}\t{instruction}")
PY

NUM_JOBS=$(wc -l < "${JOB_LIST}")
if [ "${NUM_JOBS}" -lt 1 ]; then
    echo "[error] no jobs found in ${PROMPT_YAML}"
    rm -f "${JOB_LIST}"
    exit 1
fi
echo "[info] loaded ${NUM_JOBS} editing jobs from ${PROMPT_YAML}"

# ---------- Detect available GPUs ----------
if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NUM_GPUS=0
fi
echo "[info] detected ${NUM_GPUS} GPU(s)"

if [ "${NUM_GPUS}" -lt 1 ]; then
    echo "[error] no GPU detected"
    rm -f "${JOB_LIST}"
    exit 1
fi

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
echo "[info] per-job logs will be written to: ${LOG_DIR}"
echo "[info] tail a log live with: tail -f ${LOG_DIR}/job_<idx>_gpu<id>.log"

make_slug() {
    local text=$1
    printf '%s' "${text}" \
        | tr '[:upper:]' '[:lower:]' \
        | tr -c 'a-z0-9._-' '_' \
        | tr -s '_' \
        | sed 's/^_//; s/_$//'
}

# ---------- Launcher: one inference job on one GPU ----------
declare -a PIDS=()
launch_job() {
    local gpu_id=$1
    local idx=$2
    local src_video=$3
    local level=$4
    local prompt=$5

    local input_video="${PHYSICS_DIR}/${src_video}"
    if [ ! -f "${input_video}" ]; then
        echo "[error] input video not found: ${input_video}"
        return 1
    fi

    local source_stem="${src_video%.*}"
    local slug
    slug="$(make_slug "${prompt}")"
    slug="${slug:0:${PROMPT_SLUG_LEN}}"
    slug="${slug%_}"

    local output_video="${OUTPUT_DIR}/${source_stem}_L${level}_${slug}.mp4"
    local log_file="${LOG_DIR}/job_${idx}_gpu${gpu_id}.log"

    echo "[run] gpu=${gpu_id} idx=${idx} src=${src_video} level=${level} log=${log_file}"
    python "${REPO_ROOT}/inference/infer_ditto.py" \
        --lora_path "${LORA_PATH}" \
        --local_model_path "${WAN_MODEL_ROOT}" \
        --skip_model_download \
        --input_video "${input_video}" \
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
# Round-robin jobs across GPUs; wait for each full batch before starting the next.
job_idx=0
while IFS=$'\t' read -r idx src_video level prompt; do
    gpu_id=$((job_idx % NUM_GPUS))
    launch_job "${gpu_id}" "${idx}" "${src_video}" "${level}" "${prompt}"
    job_idx=$((job_idx + 1))

    if (( job_idx % NUM_GPUS == 0 )); then
        for pid in "${PIDS[@]}"; do wait "${pid}" || true; done
        PIDS=()
    fi
done < "${JOB_LIST}"

# Drain the last (possibly partial) batch
for pid in "${PIDS[@]}"; do wait "${pid}" || true; done

rm -f "${JOB_LIST}"

echo "[done] results saved to: ${OUTPUT_DIR}"
echo "[done] logs saved to:    ${LOG_DIR}"
