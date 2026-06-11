#!/bin/bash
# Batch local editing for physics_check videos using prompt.yaml.
# Steps: ensure LoRA -> parse prompt.yaml -> detect GPUs -> run all jobs in parallel.
#
# Usage:
#   bash infer_physics_check.sh
#   bash infer_physics_check.sh --gpus 0,1
#   GPUS=2,3 bash infer_physics_check.sh
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash infer_physics_check.sh [--gpus GPU_IDS] [--quiet]

Options:
  --gpus, -g   Comma-separated CUDA device IDs to use (e.g. 0,1,3).
               Defaults to all visible GPUs.
  --quiet, -q  Suppress per-job python output (mp4 only).
  --load-on-gpu
               Load model weights directly onto GPU (needs large VRAM).

Environment:
  GPUS         Same as --gpus. CLI option takes precedence.
  LOAD_ON_GPU  Set to 1 to enable --load-on-gpu.
EOF
}

QUIET=0
LOAD_ON_GPU=0

# ---------- Paths & config ----------
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_ROOT}"

GPUS="${GPUS:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus|-g)
            if [ $# -lt 2 ]; then
                echo "[error] --gpus requires a value (e.g. 0,1)"
                usage
                exit 1
            fi
            GPUS="$2"
            shift 2
            ;;
        --quiet|-q)
            QUIET=1
            shift
            ;;
        --load-on-gpu)
            LOAD_ON_GPU=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[error] unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

PHYSICS_DIR="/data/shared-vilab/datasets/mj_data/physics_check"
PROMPT_YAML="/data/shared-vilab/datasets/mj_data/physics_check/prompt.yaml"
OUTPUT_DIR="/data/project-vilab/sy/ditto/pyhsics_results"

# Pretrained model paths
PRETRAINED_DIR="/data/shared-vilab/pretrained_models"
DITTO_MODEL_DIR="${PRETRAINED_DIR}/Ditto_models"
WAN_MODEL_DIR="${PRETRAINED_DIR}/Wan-AI"
LORA_PATH="${DITTO_MODEL_DIR}/ditto_local.safetensors"

# Inference hyperparameters
SEED=42

# Local temp dir (avoid /tmp, which often lacks write permission on shared hosts)
LOCAL_TMP_DIR="${REPO_ROOT}/tmp"

# Output filename = "<src_stem>_L<level>_<prompt slug>.mp4"
PROMPT_SLUG_LEN=50

mkdir -p "${OUTPUT_DIR}" "${LOCAL_TMP_DIR}"
export TMPDIR="${LOCAL_TMP_DIR}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-python}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "[error] python not found: ${PYTHON}"
    exit 1
fi
PYTHON="$(command -v "${PYTHON}")"

if ! "${PYTHON}" -c "import diffsynth; print('[info] diffsynth:', diffsynth.__file__)" 2>/dev/null; then
    echo "[error] diffsynth is not importable."
    echo "[error] python: $("${PYTHON}" -c 'import sys; print(sys.executable)')"
    echo "[error] PYTHONPATH: ${PYTHONPATH}"
    echo "[error] REPO_ROOT: ${REPO_ROOT}"
    echo "[fix]   conda activate ditto"
    echo "[fix]   cd ${REPO_ROOT} && pip install -e ."
    exit 1
fi
echo "[info] python: $("${PYTHON}" -c 'import sys; print(sys.executable)')"
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

for required_file in \
    "models_t5_umt5-xxl-enc-bf16.pth" \
    "Wan2.1_VAE.pth" \
    "diffusion_pytorch_model-00001-of-00007.safetensors"; do
    if [ ! -f "${WAN_MODEL_EXPECTED}/${required_file}" ]; then
        echo "[error] missing Wan model file: ${WAN_MODEL_EXPECTED}/${required_file}"
        exit 1
    fi
done
echo "[info] Wan T5/VAE/diffusion weights found under ${WAN_MODEL_EXPECTED}"

# ---------- Parse prompt.yaml into a tab-separated job list ----------
# Columns: idx | src_video | level | instruction
JOB_LIST="$(mktemp -p "${LOCAL_TMP_DIR}" physics_jobs.XXXXXX)"
"${PYTHON}" - "${PROMPT_YAML}" > "${JOB_LIST}" <<'PY'
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

# ---------- Select GPUs ----------
declare -a GPU_IDS=()
if [ -n "${GPUS}" ]; then
    IFS=',' read -ra _GPU_LIST <<< "${GPUS}"
    for gpu in "${_GPU_LIST[@]}"; do
        gpu="${gpu//[[:space:]]/}"
        if [ -z "${gpu}" ]; then
            continue
        fi
        if ! [[ "${gpu}" =~ ^[0-9]+$ ]]; then
            echo "[error] invalid GPU id: ${gpu}"
            rm -f "${JOB_LIST}"
            exit 1
        fi
        GPU_IDS+=("${gpu}")
    done
else
    if command -v nvidia-smi >/dev/null 2>&1; then
        while IFS= read -r gpu; do
            GPU_IDS+=("${gpu}")
        done < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    fi
fi

NUM_GPUS=${#GPU_IDS[@]}
if [ "${NUM_GPUS}" -lt 1 ]; then
    echo "[error] no GPU selected"
    rm -f "${JOB_LIST}"
    exit 1
fi
echo "[info] using ${NUM_GPUS} GPU(s): ${GPU_IDS[*]}"
echo "[info] outputs will be saved to: ${OUTPUT_DIR}"

# Verify selected CUDA devices exist before launching jobs.
"${PYTHON}" - "${GPU_IDS[@]}" <<'PY'
import sys
import torch

gpu_ids = [int(x) for x in sys.argv[1:]]
if not torch.cuda.is_available():
    raise SystemExit("[error] CUDA is not available")

count = torch.cuda.device_count()
print(f"[info] torch sees {count} CUDA device(s)")
for gpu_id in gpu_ids:
    if gpu_id < 0 or gpu_id >= count:
        raise SystemExit(
            f"[error] GPU {gpu_id} is invalid. Valid device ids: 0-{count - 1}"
        )
    props = torch.cuda.get_device_properties(gpu_id)
    print(f"[info] cuda:{gpu_id} -> {props.name} ({props.total_memory / 1024**3:.1f} GB)")
PY

LOAD_ON_GPU="${LOAD_ON_GPU:-0}"
if [ "${LOAD_ON_GPU}" = "1" ]; then
    echo "[info] load mode: GPU direct (--load-on-gpu)"
else
    echo "[info] load mode: CPU first, GPU for inference (default; high CPU during model load)"
fi

INFER_EXTRA_ARGS=()
if [ "${LOAD_ON_GPU}" = "1" ]; then
    INFER_EXTRA_ARGS+=(--load_on_gpu)
fi

make_slug() {
    local text=$1
    printf '%s' "${text}" \
        | tr '[:upper:]' '[:lower:]' \
        | tr -c 'a-z0-9._-' '_' \
        | tr -s '_' \
        | sed 's/^_//; s/_$//'
}

wait_batch() {
    local batch_failed=0
    for pid in "${PIDS[@]}"; do
        if ! wait "${pid}"; then
            batch_failed=$((batch_failed + 1))
        fi
    done
    PIDS=()

    local mp4_count
    mp4_count=$(find "${OUTPUT_DIR}" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)
    echo "[progress] ${job_idx}/${NUM_JOBS} jobs launched, ${mp4_count} mp4 saved"
    if [ "${batch_failed}" -gt 0 ]; then
        echo "[warn] ${batch_failed} job(s) failed in the last batch"
        FAILED_JOBS=$((FAILED_JOBS + batch_failed))
    fi
}

# ---------- Launcher: one inference job on one GPU ----------
declare -a PIDS=()
FAILED_JOBS=0
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

    echo "[run] gpu=${gpu_id} idx=${idx} src=${src_video} level=${level} -> ${output_video}"

    if [ "${QUIET}" -eq 1 ]; then
        "${PYTHON}" "${REPO_ROOT}/inference/infer_ditto.py" \
            --lora_path "${LORA_PATH}" \
            --local_model_path "${WAN_MODEL_ROOT}" \
            --skip_model_download \
            --match_input_video \
            --input_video "${input_video}" \
            --output_video "${output_video}" \
            --device_id "${gpu_id}" \
            --seed "${SEED}" \
            --prompt "${prompt}" \
            "${INFER_EXTRA_ARGS[@]}" \
            > /dev/null 2>&1 &
    else
        (
            set -o pipefail
            "${PYTHON}" "${REPO_ROOT}/inference/infer_ditto.py" \
                --lora_path "${LORA_PATH}" \
                --local_model_path "${WAN_MODEL_ROOT}" \
                --skip_model_download \
                --match_input_video \
                --input_video "${input_video}" \
                --output_video "${output_video}" \
                --device_id "${gpu_id}" \
                --seed "${SEED}" \
                --prompt "${prompt}" \
                "${INFER_EXTRA_ARGS[@]}" \
                2>&1 | sed -u "s/^/[job ${idx}|gpu${gpu_id}] /"
        ) &
    fi
    PIDS+=($!)
}

# ---------- Parallel scheduling ----------
# Round-robin jobs across GPUs; wait for each full batch before starting the next.
job_idx=0
while IFS=$'\t' read -r idx src_video level prompt; do
    gpu_id="${GPU_IDS[$((job_idx % NUM_GPUS))]}"
    launch_job "${gpu_id}" "${idx}" "${src_video}" "${level}" "${prompt}"
    job_idx=$((job_idx + 1))

    if (( job_idx % NUM_GPUS == 0 )); then
        wait_batch
    fi
done < "${JOB_LIST}"

# Drain the last (possibly partial) batch
if [ "${#PIDS[@]}" -gt 0 ]; then
    wait_batch
fi

rm -f "${JOB_LIST}"

if [ "${FAILED_JOBS}" -gt 0 ]; then
    echo "[done] finished with ${FAILED_JOBS} failed job(s). results in: ${OUTPUT_DIR}"
    exit 1
fi
echo "[done] results saved to: ${OUTPUT_DIR}"
