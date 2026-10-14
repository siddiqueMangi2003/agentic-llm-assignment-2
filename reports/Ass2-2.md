# Exercise 2: Our reasoning agent

**Draft: Qwen comparison complete; cloud-model results and group review pending.**

## 1. Code, prompts and reproduction

Our changes are in `balrog/agents/custom.py`. The supplied CustomAgent is a
starting example, not an agent that we wrote entirely ourselves. It already
maintains a plan, inserts that plan into the prompt, and parses the model's
`PLAN:` and `ACTION:` output. We retain that mechanism.

Our additions are `last_action`, `repeat_count`, `_record_action()` and
`stall_threshold = 2`. After two identical consecutive proposed actions, we append
a warning asking the model to revise the plan. We also reset the plan and action
counters between episodes to avoid carrying state into the next seed.
There is still one model call per step and no model-weight training.

The existing experiment runner selects `agent.type=custom`; the package contains
the code, prompts, raw evidence and exact commands in `run.json`.

## 2. Performance matrix

Mean progression percentage (sample SD); five episodes per benchmark/agent:

| Model | Agent | BabyAI goto | Crafter | MiniHack MazeWalk |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct | naive | 80.00 (44.72) | 9.09 (3.21) | 0.00 (0.00) |
| Qwen2.5-7B-Instruct | custom | 80.00 (44.72) | 0.91 (2.03) | 0.00 (0.00) |
| Cloud model | naive / custom | Pending | Pending | Pending |

Settings match Exercise 1: seeds 42-46, 50 steps, temperature 0, 1024 output
tokens per call, history 16, text only and one worker. BabyAI/MiniHack measure
goal success here; Crafter measures distinct achievements divided by 22.

## 3. Idea in two sentences and scope

We extend the supplied planning agent to detect two consecutive identical action
proposals and ask it to revise its plan before choosing the next action.
The aim is to reduce repeated-action stagnation while keeping one model call per step.

This is a **generic harness change**: it uses the same code and threshold in all
three benchmarks, with no game-specific rules or Qwen-specific adaptation.
Generic design does not establish that it works on all benchmarks or models.
For Qwen, the best absolute custom score is BabyAI, but there is no progression
improvement over naive there. Crafter gets worse and MiniHack remains at zero.
Cross-model conclusions must wait for the cloud run.

## 4. Reflection

**Choices and runs.** Qwen is an openly hosted model that fits the available
dual T4s. The three tasks test different interaction demands. There are 15
custom episodes and 15 paired naive episodes: one per seed and benchmark.
These are repeated trials across scenarios, not five reruns of the same prompt.

**Brittleness.** Small changes in wording, histories, seeds or model can change
behavior substantially (book 6.3.1). Custom BabyAI succeeds on seeds 42-45 but
fails on 46; therefore an aggregate 80% must not be described as stable universal
success. Five seeds and the short cap provide limited evidence.

**What actually happened.** In Crafter seed 42, custom proposes `Move West` twice
then `Move North-West` for the remaining 48 steps. All 48 diagonal proposals are
invalid and default to Noop; the repeated-action warning does not correct them.
The final Observation explicitly reports the invalid-action fallback. This is
evidence of a failed improvement, not successful reasoning merely because the
model emits a plan. The relevant trajectory is:

`vllm/Qwen_Qwen2.5-7B-Instruct/custom/seed-42/2026-10-14_15-42-08_custom_Qwen_Qwen2.5-7B-Instruct/crafter/default/default_run_00.csv`

Custom averages 29.8 invalid Crafter proposals per episode versus naive's 17.0.
On BabyAI it averages 23 steps per episode versus 13.8 for naive, counting failed
capped episodes as well as successes. Planning is therefore neither more
successful nor more efficient in these observed runs. Its average output-token
counts are also higher than naive on all three benchmarks; see performance.csv.

**Harness difficulty and generality.** Adding the counter and prompt warning is
small. Correctly resetting state, parsing action output, distinguishing proposed
from executed actions, and controlling settings for a fair comparison need care.
The trigger compares raw action strings: it misses alternating left/right loops,
and can flag useful repetitions such as walking forward twice. It tests a
heuristic for stagnation, not an observation-based proof of no progress.

**What to improve next time.** A future experiment could compare consecutive
observations, detect alternating cycles and constrain output to legal actions.
We have not implemented or tested those changes in this study. They should not
be silently added after seeing the results and reported as the tested method.

**Suitability for reasoning.** The model can solve some navigation cases, but the
planning harness alone does not resolve action grounding or loops. This supports
a conditional conclusion about this model, method and budget, not a general
claim that agentic reasoning always works or never works.

## Remaining work and references

Add the cloud naive/custom matrix rows and evaluate whether the failure pattern
is model-specific. Review the written answers before submitting.

- van Duijn et al., *Agentic Large Language Models*, arXiv:2610.13455v1,
  6.3.1, 9.2-9.3 and 10.2.5-10.3.
- Paglieri et al., BALROG, arXiv:2411.13543.
- Course assignment, Exercise 2; supplied CustomAgent example acknowledged above.
