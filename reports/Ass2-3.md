# Exercise 3: Self-reflection after repeated actions

**Draft: Qwen results complete; cloud comparison and group review pending.**

## 1. Submitted code and prompts

`balrog/agents/stall_reflection.py` extends our custom planner. The actor first
proposes a plan and action. If two consecutive previous action proposals are
identical, the same model receives a critic prompt with the observation/history,
candidate plan and candidate action. It returns `REFLECTION:`, `PLAN:` and
`ACTION:`. The revised action is executed and the revised plan is retained for
the current episode. Actor and critic input/output tokens are both counted.

The actor always makes one call; a triggered critic adds a second call. The
counter, plan and history are reset for each episode. There is no weight update,
no separate scalar evaluator and no long-term reflection memory between episodes.
This is our own small self-critique approach, not a full reproduction of Reflexion.

## 2. Performance matrix

Mean progression percentage (sample SD), five seeds per row and benchmark:

| Model | Agent | BabyAI goto | Crafter | MiniHack MazeWalk |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct | naive | 80.00 (44.72) | 9.09 (3.21) | 0.00 (0.00) |
| Qwen2.5-7B-Instruct | custom | 80.00 (44.72) | 0.91 (2.03) | 0.00 (0.00) |
| Qwen2.5-7B-Instruct | stall_reflection | 60.00 (54.77) | 3.64 (3.80) | 20.00 (44.72) |
| Cloud model | All three agents | Pending | Pending | Pending |

Each row uses the same seeds 42-46, tasks, 50-step cap, temperature 0, output
budget 1024 tokens per call, history 16 and no images. Critic calls increase
the per-step total token budget; their measured cost is part of the comparison.

## 3. Idea in two sentences

When repeated proposed actions suggest stagnation, the model critiques its own
candidate plan and action and proposes a revised plan/action before execution.
The aim is to obtain useful correction only when the trigger fires, while
measuring the added token cost and possible instability.

This uses the book's idea of the same LLM in different prompted roles (6.2.4.2,
9.2.1.4 and 10.4). A critic can be wrong and can share the actor's misconceptions.

## 4. Focus and observed benefit

We focus on Qwen and the same three tasks, with a cloud model still pending.
The design is generic across tasks. Its clearest observed progression gain is
MiniHack: one success out of five versus zero for both baselines. Crafter improves
over custom but remains below naive. BabyAI falls from four successes out of five
to three; it is incorrect to claim a general benefit.

MiniHack seed 44 succeeds in 29 steps and contains 11 critic calls. For example,
at step 7 the critic identifies moving back and forth and proposes `east`; later
actions include diagonal moves, unlike a simple north/south loop. This is
consistent with useful exploration, but does not prove that the critique caused
success: other prompt changes and the extra inference calls are also factors.
The saved path is:

`vllm/Qwen_Qwen2.5-7B-Instruct/stall_reflection/seed-44/2026-10-14_16-13-39_stall_reflection_Qwen_Qwen2.5-7B-Instruct/minihack/MiniHack-MazeWalk-9x9-v0/MiniHack-MazeWalk-9x9-v0_run_00.csv`

## 5. Stability, failures and suitability

**Failure evidence.** In Crafter seed 42, the first critic claims that a tool must
be crafted before wood can be collected. This is incorrect advice for this game
and risks a circular prerequisite. A later critique proposes using `Do` to collect
wood. Criticism can therefore introduce inconsistent explanations even in a run
with improved progression. In BabyAI seed 45, the last ten proposed actions
alternate turns and the run fails; our consecutive-repeat trigger can miss that
type of cycle.

Crafter seed 42 evidence:
`vllm/Qwen_Qwen2.5-7B-Instruct/stall_reflection/seed-42/2026-10-14_15-59-51_stall_reflection_Qwen_Qwen2.5-7B-Instruct/crafter/default/default_run_00.csv`

**Cost.** Mean critic calls per episode are 4.6 in BabyAI, 23 in Crafter and 16.4
in MiniHack. For Crafter, reflection averages 121,853.8 input and 3,442.8 output
tokens versus custom's 82,084.8 and 711.4. For MiniHack it averages 187,211.4 input
and 2,908.8 output tokens versus custom's 148,880 and 819.2. Added compute does
not guarantee improvement. We compare token usage, not monetary API cost, and
do not claim a matched-compute ablation.

**Stability.** MiniHack reflection scores are four zeros and one 100, giving SD
44.72 percentage points. This is a variable result, not evidence of dependable
maze solving. Different seeds test different scenarios; identical-prompt
repeatability and significance need more experiments. The 50-step cap can hide
later success and limits conclusions about long-horizon planning.

**What to change with more time.** Compare observations before triggering,
recognise alternating cycles and study action-validity constraints. A stronger
independent critic or memory across trials would be a different method requiring
new experiments; neither is present in these results.

**Is reflection suited to reasoning?** It can help exploration in some cases but
also adds wrong advice, costs tokens and sometimes reduces success. The evidence
supports a limited, task-dependent benefit, consistent with the book's discussion
of brittle feedback and interacting prompts (Chapter 8).

## Remaining work and references

Add cloud results, compare whether the same failure patterns recur, and review
the final answers. The optional human-versus-agent philosophy question is not
answered in this draft.

- van Duijn et al., *Agentic Large Language Models*, arXiv:2610.13455v1,
  6.2.4.2, 8.3.2-8.4, 9.2.1.4 and 10.4.
- Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*,
  cited by the book; our method differs as described in question 1.
- Paglieri et al., BALROG, arXiv:2411.13543.
- Course assignment, Exercise 3 and evaluation guidance.
