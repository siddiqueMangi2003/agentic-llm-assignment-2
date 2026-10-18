# Assignment 2: reasoning and reflection

This fresh study uses BALROG for three exercises:
1. `naive`: baseline action selection.
2. `custom`: the supplied planning agent, extended to replan after two repeated actions.
3. `stall_reflection`: the custom agent plus a second call to the same model to critique and revise a candidate when actions repeat.

The repetition rule is a simple signal of possible stagnation. It can trigger during useful repeated actions and can miss alternating loops. It is not a full implementation of Reflexion.

## Fixed study settings

| Setting | Value |
| --- | --- |
| Models | Qwen/Qwen2.5-7B-Instruct; gemini-3.5-flash-lite; nvidia/nemotron-3.5-lightning-30b-a3b |
| Hosting | Qwen self-hosted with vLLM on Kaggle; Gemini and Nemotron via API |
| Tasks | BabyAI-MixedTrainLocal-v0/goto; Crafter default; MiniHack-MazeWalk-9x9-v0 |
| Seeds | 42, 43, 44, 45, 46; one episode per task and seed |
| Episode limit | 50 environment actions |
| Generation | Temperature 0; at most 8,192 output tokens per call |
| History | 16 text messages; no images |
| Workers | One |
| Nemotron built-in thinking | Enabled for all three agents |

The 50-step cap is our chosen study condition, not an assignment requirement.
The 8,192-token limit follows BALROG's default configuration; it is an allowance, not a target response length.
Identical numeric budgets do not guarantee equal reasoning compute across providers.
API errors and missing final answers stop evaluation instead of becoming game scores.
NVIDIA responses are streamed with token usage requested. This transport change leaves the study settings unchanged; it does not guarantee protection from provider timeouts.
A critic call adds to the token use even though it does not add an environment action.

This study has 45 episodes per model, 135 in total. We will use only fresh results for the new report.
Old code, results and the earlier draft report remain separately in the original workspace and branch.

## Run on Kaggle

Open `assignment_kaggle.ipynb`, enable Internet, and run the cells in order.
The existing Kaggle notebook is `siddique200303/llm-assignment`.
Select the model using `CLIENT` in its first code cell:
- `nvidia`: CPU; grant access to Kaggle secret `NVIDIA_API_KEY`.
- `gemini`: CPU; grant access to existing secret `GEMINI_API_KEY_1`. It is exposed to the child process as `GEMINI_API_KEY`. There is no key rotation.
- `vllm`: GPU T4 x2. The notebook starts Qwen with a 32,768-token context window.

No SSH connection is required. API availability and free-tier quotas must support the requested model.
The notebook does not enable paid billing or call a separate test experiment.
It downloads our public code from branch `siddique/assignment-fresh`.

The final cell saves `assignment-results-fresh.zip`. Download it after each model.
It contains summaries, trajectories, logs and a manifest recording settings, code version and package versions.
Keep the three model archives for the report. Do not load the old results into this study.
A fresh Kaggle runtime does not automatically retain results from a previous session.

## Run from an existing Linux environment

Use Python 3.10 and install once:

```bash
python -m pip install -e . 'setuptools<81'
```

Run one model at a time with the corresponding key set in the environment:

```bash
python scripts/run_assignment.py --client nvidia --model nvidia/nemotron-3.5-lightning-30b-a3b
python scripts/run_assignment.py --client gemini --model gemini-3.5-flash-lite
python scripts/run_assignment.py --client vllm --model Qwen/Qwen2.5-7B-Instruct
```

Qwen requires an already running vLLM server on `http://127.0.0.1:18080/v1`.
API models do not need vLLM.
Completed agent/seed runs are reused only with matching recorded study settings.
Interrupted batches retain their diagnostics; they are rerun unless complete.
Keep credentials outside the code and report.

## Analysis and submission

After the runs, compare all three models for each exercise and compare the agents within each model.
Report BabyAI and MiniHack task success rates within the cap. Crafter progression is the fraction of distinct achievements unlocked out of 22.
Inspect trajectories for invalid actions, loops, revised plans and critic calls.
Record sample counts and variation across seeds. Different seeds measure different scenarios, not repeated identical prompts.
Separate environment failures, output truncation and API failures.

Write one combined report answering the assignment questions, including the matrices, explanations, trajectory analysis, limitations and required reading.
No fresh results or final report exist yet.
Prepare `Ass2-1`, `Ass2-2` and `Ass2-3` packages with the required code, scripts, prompts and README.
Preserve BALROG's licence. Do not include credentials, environment installations or temporary setup files.

BALROG's original documentation remains in `docs/`; its upstream repository is https://github.com/balrog-ai/BALROG.
