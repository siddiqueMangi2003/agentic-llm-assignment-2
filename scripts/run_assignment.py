"""Run the same small BALROG study for each agent and recorded seed."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

AGENTS = ("naive", "custom", "stall_reflection")
SEEDS = (42, 43, 44, 45, 46)
ENVIRONMENTS = ("babyai", "crafter", "minihack")
ROOT = Path(__file__).resolve().parents[1]


def evaluation_command(model, client, base_url, agent, seed, output_dir):
    return [
        sys.executable, str(ROOT / "eval.py"),
        f"agent.type={agent}", "agent.max_image_history=0", "agent.max_text_history=16",
        "envs.names=babyai-crafter-minihack",
        "tasks.babyai_tasks=[BabyAI-MixedTrainLocal-v0/goto]",
        "tasks.minihack_tasks=[MiniHack-MazeWalk-9x9-v0]",
        "eval.num_episodes.babyai=1", "eval.num_episodes.crafter=1", "eval.num_episodes.minihack=1",
        "eval.max_steps_per_episode=50", "eval.num_workers=1", "eval.save_trajectories=true",
        f"envs.env_kwargs.seed={seed}", f"client.client_name={client}",
        f"client.model_id={model}", f"client.base_url={base_url}",
        "client.generate_kwargs.temperature=0", "client.generate_kwargs.max_tokens=1024",
        f"eval.output_dir={output_dir.as_posix()}",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--client", choices=("vllm", "gemini", "openai", "claude"), default="vllm")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--output", type=Path, default=ROOT / "assignment_results")
    args = parser.parse_args()
    key_name = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "claude": "ANTHROPIC_API_KEY"}.get(args.client)
    if key_name and not os.environ.get(key_name):
        parser.error(f"Set {key_name} in the environment; never put the key in the code.")
    model_dir = args.output.resolve() / args.client / args.model.replace("/", "_")
    if (ROOT / ".git").exists():
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    else:
        commit = (ROOT / "SOURCE_COMMIT").read_text(encoding="utf-8").strip()
    versions = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    for agent in AGENTS:
        for seed in SEEDS:
            destination = model_dir / agent / f"seed-{seed}"
            # Completed runs are reused; incomplete runs retain their diagnostic logs.
            if list(destination.glob("*/summary.json")):
                print(f"Already complete: {agent}, seed {seed}", flush=True)
                continue
            destination.mkdir(parents=True, exist_ok=True)
            command = evaluation_command(args.model, args.client, args.base_url, agent, seed, destination)
            manifest = {"command": command, "commit": commit, "python": sys.version, "packages": versions}
            (destination / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Running {args.model}: {agent}, seed {seed}", flush=True)
            with (destination / "console.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print((destination / "console.log").read_text(encoding="utf-8")[-6000:], flush=True)
                raise SystemExit(result.returncode)
    print(f"Experiments saved to {model_dir}", flush=True)


if __name__ == "__main__":
    main()
