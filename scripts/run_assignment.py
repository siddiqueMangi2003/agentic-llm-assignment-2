"""Run the same bounded BALROG study for one model across all three exercises."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ("naive", "custom", "stall_reflection")
SEEDS = (42, 43, 44, 45, 46)
SETTINGS = {
    "babyai_task": "BabyAI-MixedTrainLocal-v0/goto",
    "minihack_task": "MiniHack-MazeWalk-9x9-v0",
    "crafter_task": "default",
    "seeds": list(SEEDS), "steps": 50, "output_tokens": 8192,
    "temperature": 0, "text_history": 16, "image_history": 0, "workers": 1,
}


def evaluation_command(model, client, base_url, agent, seed, output_dir):
    return [
        sys.executable, str(ROOT / "eval.py"),
        f"agent.type={agent}", "agent.max_image_history=0",
        f"agent.max_text_history={SETTINGS['text_history']}",
        "envs.names=babyai-crafter-minihack",
        f"tasks.babyai_tasks=[{SETTINGS['babyai_task']}]",
        f"tasks.minihack_tasks=[{SETTINGS['minihack_task']}]",
        "eval.num_episodes.babyai=1", "eval.num_episodes.crafter=1", "eval.num_episodes.minihack=1",
        f"eval.max_steps_per_episode={SETTINGS['steps']}", "eval.num_workers=1",
        "eval.save_trajectories=true", f"envs.env_kwargs.seed={seed}",
        f"client.client_name={client}", f"client.model_id={model}", f"client.base_url={base_url}",
        f"client.generate_kwargs.temperature={SETTINGS['temperature']}",
        f"client.generate_kwargs.max_tokens={SETTINGS['output_tokens']}",
        f"eval.output_dir={output_dir.as_posix()}",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--client", choices=("vllm", "gemini", "nvidia"), required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--output", type=Path, default=ROOT / "assignment_results_fresh")
    args = parser.parse_args()
    base_url = args.base_url or (
        "https://integrate.api.nvidia.com/v1" if args.client == "nvidia"
        else "http://127.0.0.1:18080/v1")
    key_name = {"gemini": "GEMINI_API_KEY", "nvidia": "NVIDIA_API_KEY"}.get(args.client)
    if key_name and not os.environ.get(key_name, "").strip():
        parser.error(f"Set {key_name} in the environment, outside source files.")
    settings = dict(SETTINGS)
    if args.client == "nvidia":
        settings["enable_thinking"] = False
    model_dir = args.output.resolve() / args.client / args.model.replace("/", "_")
    commit = (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
              if (ROOT / ".git").exists() else (ROOT / "SOURCE_COMMIT").read_text().strip())
    versions = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    environment = os.environ.copy()
    environment.pop("GEMINI_API_KEYS", None)
    if args.client == "gemini":
        environment.pop("GOOGLE_API_KEY", None)
    for agent in AGENTS:
        for seed in SEEDS:
            destination = model_dir / agent / f"seed-{seed}"
            manifest_path = destination / "run.json"
            previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
            runs = sorted(path for path in destination.glob("*") if path.is_dir())
            if runs:
                if not previous or previous.get("settings") != settings:
                    parser.error(f"Different or unrecorded settings in {destination}; use a separate output folder.")
                if len(runs) != 1:
                    parser.error(f"Multiple attempts in {destination}; select one before resuming.")
            if runs and (runs[0] / "summary.json").exists():
                print(f"Already complete: {agent}, seed {seed}", flush=True)
                continue
            destination.mkdir(parents=True, exist_ok=True)
            command = evaluation_command(args.model, args.client, base_url, agent, seed, destination)
            if runs:
                command.append(f"eval.resume_from={runs[0].as_posix()}")
            manifest = {"command": command, "settings": settings, "commit": commit,
                        "python": sys.version, "packages": versions}
            if previous and runs:
                previous.setdefault("resumptions", []).append(manifest)
                manifest = previous
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Running {args.model}: {agent}, seed {seed}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True, env=environment)
    print(f"Experiments saved to {model_dir}", flush=True)


if __name__ == "__main__":
    main()
