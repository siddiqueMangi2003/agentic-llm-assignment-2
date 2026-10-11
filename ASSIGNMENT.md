# Agentic LLM Assignment 2

This repository is based on BALROG commit `b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5`.

## Research question

Does reflection triggered only when an agent repeats an action improve BALROG progression while avoiding the cost and instability of reflecting after every action?

## What we compare

| Exercise | Agents | Models | Environments |
| --- | --- | --- | --- |
| 1: baseline | `naive` | Qwen2.5-7B-Instruct and one cloud model | BabyAI, Crafter, MiniHack |
| 2: reasoning | `naive`, `custom` | same models | same environments |
| 3: reflection | `naive`, `custom`, `stall_reflection` | same models | same environments |

`custom` maintains a plan and explicitly replans after two repeated actions. `stall_reflection` first proposes a plan and action, then asks a critic for a revised plan and action only after the same action has occurred twice.

The instructor-provided `BALROG_sweep_final.log` is a reference for choosing experiments and checking whether our results are plausible. Its numbers must not be reported as results from this repository.

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
