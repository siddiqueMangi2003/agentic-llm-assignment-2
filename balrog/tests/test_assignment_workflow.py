"""Local checks for matching settings, safe resumption and provider responses."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_assignment


def test_agents_and_models_share_the_same_study_budget(tmp_path):
    commands = [run_assignment.evaluation_command("model", client, "endpoint", agent, 42, tmp_path)
                for client in ("vllm", "gemini", "nvidia") for agent in run_assignment.AGENTS]
    normalized = [[arg for arg in command if not arg.startswith(("agent.type=", "client.client_name="))]
                  for command in commands]
    assert all(command == normalized[0] for command in normalized)
    assert "client.generate_kwargs.max_tokens=8192" in commands[0]
    assert "eval.max_steps_per_episode=50" in commands[0]
    assert run_assignment.SEEDS == (42, 43, 44, 45, 46)


@pytest.mark.parametrize("client,key", [("nvidia", "NVIDIA_API_KEY"), ("gemini", "GEMINI_API_KEY"), ("vllm", None)])
def test_resume_reuses_only_matching_completed_runs(tmp_path, monkeypatch, client, key):
    if key:
        monkeypatch.setenv(key, "private-test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "wrong-key")
    monkeypatch.setenv("GEMINI_API_KEYS", "obsolete-pool")
    monkeypatch.setattr(run_assignment, "SEEDS", (42,))
    monkeypatch.setattr(run_assignment.subprocess, "check_output", lambda *a, **kw: "commit")
    monkeypatch.setattr(run_assignment.importlib.metadata, "distributions", lambda: [])
    monkeypatch.setattr(sys, "argv", ["run", "--client", client, "--model", "model", "--output", str(tmp_path)])
    calls = []

    def evaluate(command, **kwargs):
        assert kwargs["check"] is True
        assert "GEMINI_API_KEYS" not in kwargs["env"]
        if client == "gemini":
            assert "GOOGLE_API_KEY" not in kwargs["env"]
        calls.append(command)
        destination = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("eval.output_dir=")))
        completed = destination / "finished"
        completed.mkdir()
        (completed / "summary.json").write_text("{}")

    monkeypatch.setattr(run_assignment.subprocess, "run", evaluate)
    run_assignment.main()
    run_assignment.main()
    assert len(calls) == 3
    manifests = list(tmp_path.rglob("run.json"))
    assert len(manifests) == 3
    assert all("private-test-key" not in p.read_text() for p in manifests)
    manifest = json.loads(manifests[0].read_text())
    assert manifest["settings"]["output_tokens"] == 8192
    if client == "nvidia":
        assert manifest["settings"]["enable_thinking"] is True
    manifest["settings"]["output_tokens"] = 1024
    manifests[0].write_text(json.dumps(manifest))
    with pytest.raises(SystemExit):
        run_assignment.main()
    assert len(calls) == 3


def test_failed_api_batch_stops_before_other_agents(tmp_path, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "private-test-key")
    monkeypatch.setattr(run_assignment.subprocess, "check_output", lambda *a, **kw: "commit")
    monkeypatch.setattr(run_assignment.importlib.metadata, "distributions", lambda: [])
    monkeypatch.setattr(sys, "argv", ["run", "--client", "nvidia", "--model", "model", "--output", str(tmp_path)])
    evaluate = Mock(side_effect=subprocess.CalledProcessError(1, ["eval.py"]))
    monkeypatch.setattr(run_assignment.subprocess, "run", evaluate)
    with pytest.raises(subprocess.CalledProcessError):
        run_assignment.main()
    assert evaluate.call_count == 1
    assert not list(tmp_path.rglob("summary.json"))


def test_nvidia_preserves_reasoning_and_does_not_repeat_missing_final_answers(monkeypatch):
    source = ROOT / "balrog" / "client.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    classes = [n for n in module.body if isinstance(n, ast.ClassDef)
               and n.name in ("LLMClientWrapper", "OpenAIWrapper")]
    sdk = Mock()
    namespace = {"OpenAI": sdk, "os": os, "time": Mock(), "logger": Mock(),
                 "LLMResponse": lambda **kw: SimpleNamespace(**kw)}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace)
    monkeypatch.setenv("NVIDIA_API_KEY", "private-test-key")
    response = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="north", reasoning_content="The exit is north."),
        finish_reason="stop")], usage=SimpleNamespace(prompt_tokens=30, completion_tokens=15))
    sdk.return_value.chat.completions.create.return_value = response
    client = namespace["OpenAIWrapper"](SimpleNamespace(
        client_name="nvidia", model_id="model", base_url="https://integrate.api.nvidia.com/v1",
        timeout=60, generate_kwargs={"temperature": 0, "max_tokens": 8192},
        max_retries=5, delay=0, alternate_roles=False))
    result = client.generate([])
    sdk.assert_called_once_with(api_key="private-test-key",
                                base_url="https://integrate.api.nvidia.com/v1", max_retries=0)
    payload = sdk.return_value.chat.completions.create.call_args.kwargs
    assert payload["max_tokens"] == 8192 and payload["temperature"] == 0
    assert payload["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert result.completion == "north" and result.reasoning == "The exit is north."
    assert result.output_tokens == 15
    response.choices[0].message.content = None
    response.choices[0].finish_reason = "length"
    with pytest.raises(RuntimeError, match="No final action"):
        client.generate([])
    assert sdk.return_value.chat.completions.create.call_count == 2


def test_environment_secrets_work_without_a_secrets_file(tmp_path, monkeypatch):
    source = ROOT / "balrog" / "utils.py"
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)
             and n.name in ("load_secrets", "setup_environment")]
    namespace = {"os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    monkeypatch.setenv("GEMINI_API_KEY", "private-test-key")
    namespace["setup_environment"](original_cwd=str(tmp_path))
    assert os.environ["GEMINI_API_KEY"] == "private-test-key"
