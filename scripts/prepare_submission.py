"""Build performance tables and clearly labelled submission drafts from completed runs."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import subprocess
import zipfile

from run_assignment import AGENTS, ENVIRONMENTS, ROOT, SEEDS


def collect_rows(results):
    rows = []
    seen = set()
    for path in sorted(results.rglob("*_run_*.json")):
        episode = json.loads(path.read_text(encoding="utf-8"))
        # Only complete evaluations belong in the comparison; retain incomplete files as diagnostics.
        summary = next((p for p in path.parents if p != results and (p / "summary.json").is_file()), None)
        if summary is None:
            continue
        environment = path.relative_to(summary).parts[0]
        client, agent = episode["client"], episode["agent"]["type"]
        key = (client["client_name"], client["model_id"], agent, environment, episode["task"], episode["seed"])
        if key in seen:
            raise ValueError(f"Duplicate episode: {key}. Keep one version of each experiment.")
        seen.add(key)
        with path.with_suffix(".csv").open(encoding="utf-8", newline="") as trajectory:
            # NetHack observations contain NULs; remove them for parsing, leaving raw files intact.
            lines = (line.replace("\x00", "") for line in trajectory)
            reflections = sum("REFLECTION:" in step["Reasoning"] for step in csv.DictReader(lines, escapechar="˘"))
        rows.append(dict(zip(("client", "model", "agent", "benchmark", "task", "seed"), key),
                         progression=100 * episode["progression"], steps=episode["num_steps"],
                         invalid_actions=len(episode["failed_candidates"]), reflections=reflections,
                         input_tokens=episode["input_tokens"], output_tokens=episode["output_tokens"],
                         trajectory=path.with_suffix(".csv").relative_to(results).as_posix()))
    if not rows:
        raise ValueError("No completed evaluations found.")
    return rows


def performance_table(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["client"], row["model"], row["agent"], row["benchmark"])].append(row)
    table = []
    for key, episodes in sorted(groups.items()):
        scores = [row["progression"] for row in episodes]
        table.append(dict(zip(("client", "model", "agent", "benchmark"), key),
                          n=len(scores), mean_progression=statistics.mean(scores),
                          sd_progression=statistics.stdev(scores) if len(scores) > 1 else None,
                          **{f"mean_{field}": statistics.mean(row[field] for row in episodes)
                             for field in ("steps", "invalid_actions", "reflections", "input_tokens", "output_tokens")}))
    return table


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "assignment_results")
    parser.add_argument("--output", type=Path, default=ROOT / "submission")
    args = parser.parse_args()
    results = args.results.resolve()
    output = args.output.resolve()
    rows = collect_rows(results)
    table = performance_table(rows)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "episodes.csv", rows)
    write_csv(output / "performance.csv", table)
    matrix = defaultdict(dict)
    for row in table:
        key = (row["client"], row["model"], row["agent"])
        sd = "unknown" if row["sd_progression"] is None else f'{row["sd_progression"]:.2f}'
        matrix[key][row["benchmark"]] = f'{row["mean_progression"]:.2f} (SD {sd}, n={row["n"]})'
    matrix_rows = [dict(zip(("client", "model", "agent"), key),
                        **{env: matrix[key].get(env, "MISSING") for env in ENVIRONMENTS}) for key in sorted(matrix)]
    write_csv(output / "matrix.csv", matrix_rows)
    for exercise in (1, 2, 3):
        selected = AGENTS[:exercise]
        selected_rows = [row for row in rows if row["agent"] in selected]
        selected_table = [row for row in table if row["agent"] in selected]
        if not selected_rows:
            continue
        stage = output / f"Ass2-{exercise}"
        stage.mkdir(exist_ok=True)
        write_csv(stage / "episodes.csv", selected_rows)
        write_csv(stage / "performance.csv", selected_table)
        write_csv(stage / "matrix.csv", [row for row in matrix_rows if row["agent"] in selected])
        report = stage / "REPORT.md"
        # Preserve the group's written answers when regenerating tables.
        if not report.exists():
            questions = {
                1: "Model and benchmark choices; trajectory failures; reasoning strengths and weaknesses; ideas for improvement.",
                2: "Two-sentence agent idea; generic or benchmark-specific scope; comparison with naive; brittleness; harness design; what to improve.",
                3: "Two-sentence reflection idea; evidence that the critic was called; best/worst benchmark; stability; failures; suitability of self-reflection.",
            }
            report.write_text(f"# Exercise {exercise}: written answers (DRAFT)\n\n"
                              "Complete these answers from the tables and trajectory CSVs before submission.\n\n"
                              f"{questions[exercise]}\n\n"
                              "## Method and model choices\nTODO\n\n## Results and trajectory evidence\nTODO\n\n"
                              "## Reflection and limitations\nTODO\n", encoding="utf-8")
        models = sorted({(row["client"], row["model"]) for row in selected_rows})
        observed = {(row["client"], row["model"], row["agent"], row["benchmark"], row["seed"]) for row in selected_rows}
        missing = [(client, model, agent, env, seed) for client, model in models for agent in selected
                   for env in ENVIRONMENTS for seed in SEEDS if (client, model, agent, env, seed) not in observed]
        readme = (f"# Ass2-{exercise} (DRAFT)\n\nAgents: {', '.join(selected)}.\n\n"
                  "Install Python 3.10, then run `pip install -e . 'setuptools<81'`. "
                  "For Qwen, also install `vllm==0.31.0` and use the included Kaggle notebook.\n\n"
                  "Cloud: export the provider API key, then run `python scripts/run_assignment.py "
                  "--client gemini --model YOUR_MODEL_ID`. The runner uses the same seeds and budgets.\n\n"
                  "Study: BabyAI goto, Crafter default, MiniHack MazeWalk 9x9; seeds 42-46; one episode per seed; "
                  "50-step cap; temperature 0; max output 1024 tokens; history 16; no images. "
                  "This is a bounded study, not reproduction of the complete BALROG paper.\n\n"
                  "Metric: BALROG progression × 100; benchmark-specific progress, not a shared success rate. "
                  "SD is sample standard deviation across seeds, not proof of repeatability on the same seed. "
                  "Token counts include the extra critic call. Invalid actions are rejected model candidates.\n\n"
                  f"Recorded models: {models}. Missing planned episodes for these models: {len(missing)}.\n\n"
                  "Before Brightspace: include one open self-hosted model and one cloud API model; finish REPORT.md; "
                  "check trajectories and all five seeds; remove the DRAFT label only after review. "
                  "Raw JSON/CSV and run.json record prompts, actions, commands, dependency versions, and commit.\n")
        (stage / "README.md").write_text(readme, encoding="utf-8")
        sources = [ROOT / name for name in ("eval.py", "setup.py", "pyproject.toml", "LICENSE", "kaggle_qwen_baseline.ipynb")]
        sources += [p for p in (ROOT / "balrog").rglob("*") if p.is_file()
                    and not any(part in ("__pycache__", "tests") for part in p.relative_to(ROOT).parts)]
        sources += [ROOT / "scripts" / name for name in ("run_assignment.py", "prepare_submission.py")]
        with zipfile.ZipFile(output / f"Ass2-{exercise}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for source in sources:
                archive.write(source, source.relative_to(ROOT))
            archive.writestr("SECRETS", "OPENAI_API_KEY=\nANTHROPIC_API_KEY=\nGEMINI_API_KEY=\n")
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() if (ROOT / ".git").exists() else (ROOT / "SOURCE_COMMIT").read_text(encoding="utf-8").strip()
            archive.writestr("SOURCE_COMMIT", commit + "\n")
            for source in stage.iterdir():
                archive.write(source, source.name)
            # Include only this exercise's agents, preserving diagnostic logs and run manifests.
            for source in results.rglob("*"):
                relative = source.relative_to(results)
                if source.is_file() and len(relative.parts) > 3 and relative.parts[2] in selected:
                    archive.write(source, Path("results") / relative)
        print(f"Draft package: {output / f'Ass2-{exercise}.zip'}")
    print(f"Tables: {output / 'matrix.csv'}")


if __name__ == "__main__":
    main()
