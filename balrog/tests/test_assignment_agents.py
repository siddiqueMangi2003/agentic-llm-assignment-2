from collections import namedtuple
from types import SimpleNamespace

from balrog.agents.custom import CustomAgent
from balrog.agents.stall_reflection import StallReflectionAgent


Response = namedtuple(
    "Response", "model_id completion stop_reason input_tokens output_tokens reasoning"
)


class FakeClient:
    def __init__(self, completions):
        self.completions = iter(completions)
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return Response("fake", next(self.completions), "stop", 2, 3, None)


class FakePromptBuilder:
    def __init__(self):
        self.actions = []
        self.reset_calls = 0

    def update_action(self, action):
        self.actions.append(action)

    def update_observation(self, obs):
        pass

    def get_prompt(self):
        return [SimpleNamespace(role="user", content="observation")]

    def reset(self):
        self.reset_calls += 1


def make_agent(agent_type, completions):
    client = FakeClient(completions)
    builder = FakePromptBuilder()
    return agent_type(lambda: client, builder), client, builder


def test_custom_agent_reset_clears_episode_state():
    agent, _, builder = make_agent(CustomAgent, [])
    agent.plan = "old plan"
    agent.last_action = "left"
    agent.repeat_count = 3

    agent.reset()

    assert agent.plan is None
    assert agent.last_action is None
    assert agent.repeat_count == 0
    assert builder.reset_calls == 1


def test_reflection_runs_only_after_two_equal_actions():
    agent, client, _ = make_agent(
        StallReflectionAgent,
        [
            "PLAN: explore\nACTION: left",
            "PLAN: explore\nACTION: left",
            "PLAN: explore\nACTION: left",
            "REFLECTION: left is blocked\nPLAN: try another route\nACTION: right",
        ],
    )

    agent.act({}, prev_action=None)
    agent.act({}, prev_action="left")
    result = agent.act({}, prev_action="left")

    assert client.calls == 4
    assert result.completion == "right"
    assert result.input_tokens == 4
    assert result.output_tokens == 6
    assert "left is blocked" in result.reasoning
