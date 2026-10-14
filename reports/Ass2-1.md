# Exercise 1: Running the benchmark

**Draft: Qwen results complete; cloud-model comparison and group review pending.**

## 1. Submitted code, prompts and instructions

The package contains the BALROG library, `eval.py`, the Kaggle notebook,
`scripts/run_assignment.py`, the result/packaging script and a README.
Agent prompts are in `balrog/agents/`; benchmark instructions are in
`balrog/environments/`; history construction is in `balrog/prompt_builder/`.
Raw JSON/CSV trajectories, commands, dependency versions and source commit are
included. The professor's sweep log is a reference, not our experimental result.

## 2. Models, benchmarks and experimental settings

We ran Qwen/Qwen2.5-7B-Instruct through our own vLLM server on Kaggle T4 x2.
Although Kaggle supplies remote hardware, we host the model ourselves; this is
different from calling a provider-managed model API. Qwen is an instruction-tuned
model; these results do not isolate specialised reasoning-model training.

Cloud model: **pending**. Record its exact model ID, provider, free/student
allowance and inference settings before completing the cross-model comparison.

We selected BabyAI goto for instruction following and navigation, Crafter default
for resource collection and prerequisites, and MiniHack MazeWalk 9x9 for maze
navigation. This gives different challenges without exhaustively running all six
BALROG environments or all tasks. The book recommends a focused, affordable study.

Each agent/model uses one episode per benchmark for each seed 42-46: five
episodes per benchmark. Settings are 50 maximum steps, temperature 0, 1024 maximum
output tokens per model call, 16 text observations, no images and one evaluation
worker. This is a bounded study, not a reproduction of the full BALROG leaderboard.

## 3. Performance matrix and meaning of the metric

Mean progression percentage with sample standard deviation across five seeds:

| Model / naive agent | BabyAI goto | Crafter | MiniHack MazeWalk |
| --- | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct | 80.00 (44.72) | 9.09 (3.21) | 0.00 (0.00) |
| Cloud model | Pending | Pending | Pending |

BALROG multiplies each episode's progression by 100. For BabyAI goto and this
MiniHack task, progression is success (1) or failure (0), so the mean equals the
success percentage. Crafter progression is the number of distinct achievements
unlocked divided by 22: 9.09% corresponds to two achievements per episode on
average, not a 9.09% game-winning rate. These are different benchmark-specific
quantities. A zero MiniHack score says that the selected goal was not completed
within our budget, not that no useful movement occurred.

See `performance.csv` for steps, invalid actions and input/output token counts;
`episodes.csv` links each result to its raw trajectory. Standard deviation here
is across different scenarios/seeds; it is not a confidence guarantee or a
measurement of identical-prompt repeatability.

## 4. Reflection supported by trajectories

**What failed and how we handled it.** The university scheduler was unavailable
or delayed, so we ran the same bounded study on Kaggle. Python compatibility and
the legacy NetHack `pkg_resources` import required Python 3.10 and `setuptools<81`.
All 45 Qwen episodes completed; packaging subsequently failed on NUL characters
in NetHack CSV observations. Removing NULs only while parsing recovered the
tables without changing raw results or rerunning inference.

**What the benchmarks reveal.** BabyAI seed 42 succeeds in three actions. In
contrast, seed 46 alternates `turn left` and `turn right` for all 50 steps, with
25 of each action. All actions are legal but they make no progress: action
validity alone is not reasoning competence. In Crafter seed 42 the agent proposes
15 invalid actions, including `Collect Wood`, `Cut Tree` and `Move SouthEast`,
and repeatedly tries `Make Wood Pickaxe` late in the episode. The environment
falls back to Noop on invalid proposals. Inspect the corresponding Observation
fields as well as the Action column: Action records the proposal, not necessarily
the action actually executed after validation. MiniHack fails on all five seeds.

Evidence paths below are relative to the included results directory:

- `vllm/Qwen_Qwen2.5-7B-Instruct/naive/seed-42/2026-10-14_15-31-24_naive_Qwen_Qwen2.5-7B-Instruct/crafter/default/default_run_00.csv`
- `vllm/Qwen_Qwen2.5-7B-Instruct/naive/seed-46/2026-10-14_15-39-28_naive_Qwen_Qwen2.5-7B-Instruct/babyai/BabyAI-MixedTrainLocal-v0/goto/BabyAI-MixedTrainLocal-v0/goto_run_00.csv`

**Are LLMs good at reasoning?** These runs show useful short navigation behavior,
but also legal action loops and incorrect action vocabulary. They do not justify
a universal claim about reasoning, all games or all LLMs. The 50-step cap and
one selected task per environment limit what we can conclude.

**Ideas for the next exercises.** Keep an explicit plan, warn the model when an
action repeats, and test whether a separate critic can revise a failing plan.
These hypotheses motivate Exercises 2 and 3; improvement must be measured.

## Remaining work

Add the cloud-model row and model-choice rationale, compare its trajectories,
and review the interpretations together before final submission.

## References

- van Duijn, van der Meer, Plaat and van Stein. *Agentic Large Language Models*,
  arXiv:2610.13455v1, Chapters 6-9 and 10.1-10.3.
- Paglieri et al. *BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games*,
  arXiv:2411.13543. Our budget/task subset differs from the paper.
- Course assignment, Exercise 1 and evaluation guidance.
