#!/usr/bin/env bash
#SBATCH --job-name=balrog-smoke
#SBATCH --partition=L40s_lu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8080}"

command -v python >/dev/null
command -v vllm >/dev/null
command -v curl >/dev/null
nvidia-smi

vllm serve "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  >"vllm-${SLURM_JOB_ID}.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

for _ in {1..60}; do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    cat "vllm-${SLURM_JOB_ID}.log"
    exit 1
  fi
  sleep 5
done

curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null

for agent in naive custom stall_reflection; do
  python eval.py \
    agent.type="$agent" \
    envs.names=crafter \
    eval.num_episodes.crafter=1 \
    eval.max_steps_per_episode=5 \
    eval.num_workers=1 \
    client.client_name=vllm \
    client.model_id="$MODEL_ID" \
    client.base_url="http://127.0.0.1:${PORT}/v1" \
    client.generate_kwargs.temperature=0
done
