"""Checks for paired experiment settings and honest result aggregation."""

import csv
import ast
import json
from pathlib import Path
import sys
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from prepare_submission import collect_rows, performance_table
from prepare_submission import main as prepare_submission
from run_assignment import AGENTS, SEEDS, evaluation_command
import run_assignment


def test_all_agents_share_the_same_settings(tmp_path):
    commands = [evaluation_command("model", "vllm", "http://localhost/v1", agent, 42, tmp_path)
                for agent in AGENTS]
    without_agent = [[arg for arg in command if not arg.startswith("agent.type=")] for command in commands]
    assert without_agent[0] == without_agent[1] == without_agent[2]
    assert "envs.env_kwargs.seed=42" in commands[0]
    assert "eval.num_episodes.crafter=1" in commands[0]
    assert len(SEEDS) == len(set(SEEDS)) == 5


@pytest.mark.parametrize("returncode", [0, 1])
def test_gemini_keys_are_assigned_only_to_unfinished_runs(tmp_path, monkeypatch, capsys, returncode):
    keys = [f"private-key-{i}" for i in range(5)]
    monkeypatch.setenv("GEMINI_API_KEYS", json.dumps(keys))
    monkeypatch.setenv("GOOGLE_API_KEY", "other-project-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(run_assignment, "SEEDS", (42, 43, 44))
    monkeypatch.setattr(run_assignment.subprocess, "check_output", lambda *args, **kwargs: "commit\n")
    monkeypatch.setattr(run_assignment.importlib.metadata, "distributions", lambda: [])
    monkeypatch.setattr(sys, "argv", ["run_assignment", "--client", "gemini", "--model", "model", "--output", str(tmp_path)])
    completed = tmp_path / "gemini" / "model" / "naive" / "seed-42" / "previous"
    completed.mkdir(parents=True)
    (completed / "summary.json").write_text("{}")
    calls = []

    def evaluate(command, **kwargs):
        environment = kwargs["env"]
        assert "GEMINI_API_KEYS" not in environment
        assert "GOOGLE_API_KEY" not in environment
        calls.append(environment["GEMINI_API_KEY"])
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(run_assignment.subprocess, "run", evaluate)
    if returncode:
        with pytest.raises(SystemExit) as error:
            run_assignment.main()
        assert error.value.code == returncode
        assert calls == keys[:1]  # Failed evaluations stop without switching projects.
    else:
        run_assignment.main()
        assert calls == keys + keys[:3]
    manifests = [json.loads(path.read_text()) for path in tmp_path.rglob("run.json")]
    assert sorted(manifest["gemini_key_slot"] for manifest in manifests) == sorted(keys.index(key) + 1 for key in calls)
    assert all(key not in json.dumps(manifests) for key in keys)
    output = capsys.readouterr().out
    assert all(key not in output for key in keys)
    assert (completed / "summary.json").read_text() == "{}"


@pytest.mark.parametrize("value", ["not-json", "{}", "[]", '[" "]', '[123]', '["same", "same"]'])
def test_invalid_gemini_key_lists_fail_before_evaluation(monkeypatch, value):
    monkeypatch.setenv("GEMINI_API_KEYS", value)
    monkeypatch.setattr(sys, "argv", ["run_assignment", "--client", "gemini", "--model", "model"])
    with pytest.raises(SystemExit) as error:
        run_assignment.main()
    assert error.value.code == 2


def cloud_generate_method():
    # Exercise the real method's error handling without importing optional provider SDKs.
    source = Path(__file__).resolve().parents[1] / "client.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    wrapper = next(node for node in module.body if isinstance(node, ast.ClassDef)
                   and node.name == "GoogleGenerativeAIWrapper")
    generate = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == "generate")
    namespace = {"logger": Mock(), "LLMResponse": lambda **fields: SimpleNamespace(**fields)}
    exec(compile(ast.Module(body=[generate], type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def test_exhausted_cloud_retries_stop_evaluation():
    namespace = cloud_generate_method()
    fake = SimpleNamespace(_initialize_client=Mock(), convert_messages=Mock(return_value=[]),
                           execute_with_retries=Mock(side_effect=RuntimeError("quota exhausted")),
                           model_id="test-cloud-model", max_retries=5)
    with pytest.raises(RuntimeError, match="Cloud inference failed") as error:
        namespace["generate"](fake, [])
    assert str(error.value.__cause__) == "quota exhausted"


def test_gemini_pacing_is_shared_by_actor_and_critic_clients():
    namespace = cloud_generate_method()
    now, requests = [100.0], []
    clock = Mock()
    clock.monotonic.side_effect = lambda: now[0]
    clock.sleep.side_effect = lambda seconds: now.__setitem__(0, now[0] + seconds)
    namespace.update(time=clock, GoogleGenerativeAIWrapper=SimpleNamespace(_last_request_at=0.0))
    response = SimpleNamespace(usage_metadata=None, candidates=[SimpleNamespace(finish_reason="STOP")])
    provider = Mock()
    provider.models.generate_content.side_effect = lambda **kwargs: requests.append(now[0]) or response
    options = dict(_initialize_client=Mock(), convert_messages=Mock(return_value=[]),
                   execute_with_retries=lambda call: call(), client=provider, generation_config={},
                   extract_completion=lambda response: "north", extract_reasoning=lambda response: None,
                   _extract_reasoning_token_count=lambda response: 0, model_id="test-model", max_retries=5)
    actor, critic = SimpleNamespace(**options), SimpleNamespace(**options)
    for client in (actor, critic, actor):
        namespace["generate"](client, [])
    assert requests == [100.0, 105.0, 110.0]


def write_episode(root, run, seed, progression):
    path = root / "vllm" / "model" / "naive" / f"seed-{seed}" / run
    path.mkdir(parents=True)
    (path / "summary.json").write_text("{}")
    episode = path / "babyai" / "task_run_00.json"
    episode.parent.mkdir()
    episode.write_text(json.dumps({
        "client": {"client_name": "vllm", "model_id": "model"}, "agent": {"type": "naive"},
        "task": "goto", "seed": seed, "progression": progression, "num_steps": 5,
        "input_tokens": 20, "output_tokens": 8, "failed_candidates": ["invalid"],
    }))
    with episode.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Reasoning"])
        writer.writeheader()
        writer.writerow({"Reasoning": "REFLECTION: try another route"})
    return path


def test_scores_variation_and_critic_counts(tmp_path):
    write_episode(tmp_path, "first", 42, 0)
    write_episode(tmp_path, "second", 43, 1)
    rows = collect_rows(tmp_path)
    result = performance_table(rows)[0]
    assert result["mean_progression"] == 50
    assert result["sd_progression"] == pytest.approx(70.710678)
    assert result["mean_reflections"] == result["mean_invalid_actions"] == 1
    assert performance_table(rows[:1])[0]["sd_progression"] is None


def test_nethack_control_characters_are_removed_only_while_reading(tmp_path, monkeypatch):
    run = write_episode(tmp_path, "first", 42, 0)
    path = run / "babyai" / "task_run_00.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Reasoning", "Observation"], escapechar="˘")
        writer.writeheader()
        writer.writerow({"Reasoning": "REFLECTION: investigate\nPLAN: explore", "Observation": 'map\x00\n"room"˘'})
    original = path.read_bytes()
    reader = csv.DictReader

    def python310_reader(lines, **kwargs):
        lines = list(lines)
        assert all("\x00" not in line for line in lines)
        return reader(lines, **kwargs)

    monkeypatch.setattr(csv, "DictReader", python310_reader)
    assert collect_rows(tmp_path)[0]["reflections"] == 1
    assert path.read_bytes() == original


def test_duplicates_fail_and_incomplete_runs_are_excluded(tmp_path):
    write_episode(tmp_path, "first", 42, 1)
    duplicate = write_episode(tmp_path, "second", 42, 0)
    with pytest.raises(ValueError, match="Duplicate episode"):
        collect_rows(tmp_path)
    (duplicate / "summary.json").unlink()
    assert len(collect_rows(tmp_path)) == 1


def test_package_has_raw_evidence_blank_credentials_and_preserves_report(tmp_path, monkeypatch):
    results, output = tmp_path / "results", tmp_path / "submission"
    write_episode(results, "first", 42, 1)
    monkeypatch.setattr(sys, "argv", ["prepare_submission", "--results", str(results), "--output", str(output)])
    prepare_submission()
    report = output / "Ass2-1" / "REPORT.md"
    report.write_text("Our written answers", encoding="utf-8")
    prepare_submission()
    with zipfile.ZipFile(output / "Ass2-1.zip") as archive:
        assert archive.read("REPORT.md").decode() == "Our written answers"
        assert archive.read("SECRETS").decode() == "OPENAI_API_KEY=\nANTHROPIC_API_KEY=\nGEMINI_API_KEY=\n"
        assert "scripts/run_assignment.py" in archive.namelist()
        assert "SOURCE_COMMIT" in archive.namelist()
        assert any(name.endswith("task_run_00.json") for name in archive.namelist())
        assert not any(".git/" in name or "__pycache__" in name for name in archive.namelist())
