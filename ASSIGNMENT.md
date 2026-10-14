# Agentic LLM Assignment 2

This repository is based on BALROG commit `b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5`.

## Research question

Does adding a critic after repeated actions improve progression over a naive agent and a planning agent, and how much extra token usage does it require?

## What we compare

| Exercise | Agents | Models | Environments |
| --- | --- | --- | --- |
| 1: baseline | `naive` | Qwen2.5-7B-Instruct and one cloud model | BabyAI, Crafter, MiniHack |
| 2: reasoning | `naive`, `custom` | same models | same environments |
| 3: reflection | `naive`, `custom`, `stall_reflection` | same models | same environments |

`custom` maintains a plan and explicitly replans after two repeated actions. `stall_reflection` first proposes a plan and action, then asks a critic for a revised plan and action only after the same action has occurred twice.

The instructor-provided `BALROG_sweep_final.log` is a reference for choosing experiments and checking whether our results are plausible. Its numbers must not be reported as results from this repository.

## Combined experiment

Open `kaggle_qwen_baseline.ipynb` on Kaggle with Internet enabled and GPU T4 x2.
The existing `LLM-assignment` notebook is updated with this same code. Qwen starts once;
the runner evaluates all three agents on seeds 42, 43, 44, 45, 46. Each seed runs
one episode in each of the three environments: 45 episodes total for each model.
Completed evaluations are skipped when resuming from the same result directory.

All comparisons retain the pilot settings: 50 steps per episode, temperature 0,
1024 output tokens, 16 text observations, no images, and one evaluation worker.
This is an explicitly bounded experiment, not reproduction of the full paper.
MiniHack and Crafter reaching the limit must not be described as natural game termination.
Five different seeds measure variation across scenarios. They do not establish
repeatability of an identical scenario; discuss this distinction in the report.

The notebook writes raw results under `/kaggle/working/assignment_results` and
tables plus three **draft** ZIPs under `/kaggle/working/submission`.
`matrix.csv` gives model × benchmark comparisons for each agent;
`performance.csv` adds sample SD, episode count, steps, invalid actions, critic
calls, and input/output tokens. `episodes.csv` points to the trajectory evidence.
Do not treat a missing row, or a single episode with zero SD, as a reliable result.

## Cloud model (person 2)

In the same notebook set `CLIENT = "gemini"` (or `openai` or `claude`), and set
`MODEL` to the exact model ID available in your free/student account. Add the
corresponding Kaggle secret: `GEMINI_API_KEY`, `OPENAI_API_KEY`, or
`ANTHROPIC_API_KEY`. The notebook uses Kaggle's secret manager and needs no GPU
in cloud mode. No paid service or billing is enabled by this code.
Select a model that accepts the configured temperature/output-token settings;
record the exact model and whether it is a specialised reasoning model.

The same study can run from an activated Python 3.10 environment:

```bash
python scripts/run_assignment.py --client gemini --model YOUR_MODEL_ID
python scripts/prepare_submission.py
```

Set the API key in the environment before the cloud command. For a local Qwen
server, use `--client vllm --model Qwen/Qwen2.5-7B-Instruct` instead.
Download each person's result archive, combine their `assignment_results`
folders (the client/model subfolders keep runs separate), then run
`prepare_submission.py --results PATH_TO_COMBINED_RESULTS --output submission`.
Do not add the earlier one-seed pilot into the five-seed results: seed 42 is
rerun as part of the controlled study.

## Remaining written work

The ZIPs remain DRAFT until both models and all planned runs have been reviewed.
Fill each `REPORT.md` with the assignment's written answers and trajectory
evidence; the packaging script preserves your edits. Exercise 1 covers model
and benchmark choices, failures, reasoning, and ideas. Exercise 2 covers the
planning idea, scope, comparative results, brittleness, and harness design.
Exercise 3 covers how the critic self-reflects, comparative results, stability,
failures, and suitability for reasoning. The philosophy question is optional.
Regenerate packages after writing reports. Only then submit `Ass2-1.zip`,
`Ass2-2.zip`, and `Ass2-3.zip` on Brightspace.

The prompts are in `balrog/agents/naive.py`, `custom.py`, `stall_reflection.py`,
the benchmark environment code, and the prompt builder. They are included
with the required BALROG library, entry point, installation files, and licence.
Caches, keys, models, Git history, and unrelated project files are excluded.

## Interpreting the experiment

Repeated actions are only a heuristic for stagnation: moving forward twice can
be correct. The critic diagnoses a possible failure from the current observation,
plan and candidate action; it does not train model weights or retain memory
between episodes. Critic calls are labelled `REFLECTION:` in trajectory CSVs.
The custom baseline already replans on repetition, so the additional critic call
is the difference tested in Exercise 3. Cost and benefit must both be reported.

## Minimal workflow

1. Run the Qwen baseline on BabyAI, Crafter, and MiniHack.
2. Run `custom` and `stall_reflection` with identical model settings and seeds.
3. Inspect trajectory CSV files and confirm that reflection occurs only after a repeated action.
4. Repeat runs with recorded seeds. Expand the number of runs only when the pilot is unstable.
5. Report mean progression and variation, token usage, representative trajectories, failures, and the exact commands used.

Do not put API keys in `SECRETS` or commit them. Export keys in the shell immediately before a cloud run.

## REL setup

Connect to the SLURM frontend from Windows:

```powershell
ssh -J s5257875@ssh.liacs.nl s5257875@fs.cc.liacs.nl
```

Clone this group repository on the frontend, create a Python 3.10 environment, and install BALROG there. Run compute work through SLURM; do not run evaluation or model serving on the frontend.

After the environment is installed and activated:

```bash
sbatch scripts/rel_baseline.sbatch
squeue -u "$USER"
```

The baseline job writes its scheduler log to `slurm-balrog-baseline-<job-id>.out` and BALROG outputs under `results/`. Both paths are ignored by Git.

## Team split

- Person 1 owns the shared repository, REL environment, local Qwen runs, and implementation checks.
- Person 2 owns the cloud baseline, trajectory review, result tables, and first drafts of the three reflections.
- Each person works in a separate branch and opens a merge request. Only merge completed, reproducible changes into `main`.

This split lets both people work independently while using the same agent code and experiment table.
