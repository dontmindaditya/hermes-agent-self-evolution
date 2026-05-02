"""Tests for evolution configuration defaults."""

from pathlib import Path

import pytest

from evolution.core.config import EvolutionConfig, discover_hermes_agent_path, get_hermes_agent_path


def test_config_constructs_without_installed_hermes_repo(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    config = EvolutionConfig()

    assert config.hermes_agent_path == tmp_path


def test_discover_hermes_agent_path_reads_env_var(monkeypatch, tmp_path):
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    monkeypatch.setenv("HERMES_AGENT_REPO", str(repo))

    assert discover_hermes_agent_path() == repo


def test_get_hermes_agent_path_still_raises_when_required(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_AGENT_REPO", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    with pytest.raises(FileNotFoundError):
        get_hermes_agent_path()
