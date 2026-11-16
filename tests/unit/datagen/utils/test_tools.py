"""
Tests for `find_tool`, the one lookup every external executable goes through, i.e.:

Step 1: Ensure the lookup order: env var, then tools.toml, then PATH
    - test_env_var_wins_over_toml
    - test_toml_entry_wins_over_path
    - test_toml_relative_path_resolves_against_toml_dir
    - test_path_fallback_when_unconfigured
    - test_path_fallback_searches_exe_name
Step 2: Ensure a broken or stale configuration fails closed: None, never raised, never PATH
    - test_configured_but_missing_path_does_not_fall_back_to_path
    - test_unreadable_toml_returns_none
    - test_non_string_entry_returns_none
    - test_unconfigured_and_not_on_path_returns_none
Step 3: Ensure the real lookup reads the repo-root tools.toml, beside its committed example
    - test_tools_toml_sits_at_repo_root_beside_example

Scope: the lookup only. Each runner's not-found flag is tested with that runner.
"""

import os

import pytest

from src.datagen.utils import tools
from src.datagen.utils.tools import find_tool

KEY = "fftesttool"
ENV_VAR = tools.ENV_PREFIX + KEY.upper()
REPO_TOOLS_TOML = tools.TOOLS_TOML  # Captured before the fixture redirects it


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """A per-test tools.toml location and a PATH holding only an empty dir, env var unset."""
    monkeypatch.setattr(tools, "TOOLS_TOML", tmp_path / "tools.toml")
    monkeypatch.delenv(ENV_VAR, raising=False)
    (tmp_path / "path").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "path"))
    return tmp_path


def _exe(directory, name=KEY):
    """An empty file PATH lookup accepts as an executable on this platform."""
    p = directory / (name + ".exe" if os.name == "nt" else name)
    p.write_text("")
    p.chmod(0o755)
    return p


def _write_toml(directory, body):
    (directory / "tools.toml").write_text(body)


def _same(found, expected):
    return found is not None and os.path.samefile(found, expected)


# region Step 1
def test_env_var_wins_over_toml(isolated, monkeypatch):
    env_exe, toml_exe = _exe(isolated, "from_env"), _exe(isolated, "from_toml")
    _write_toml(isolated, f"[tools]\n{KEY} = '{toml_exe}'\n")
    monkeypatch.setenv(ENV_VAR, str(env_exe))
    assert _same(find_tool(KEY), env_exe)


def test_toml_entry_wins_over_path(isolated):
    toml_exe = _exe(isolated, "from_toml")
    _exe(isolated / "path")
    _write_toml(isolated, f"[tools]\n{KEY} = '{toml_exe}'\n")
    assert _same(find_tool(KEY), toml_exe)


def test_toml_relative_path_resolves_against_toml_dir(isolated, monkeypatch):
    (isolated / "bin").mkdir()
    exe = _exe(isolated / "bin")
    _write_toml(isolated, f"[tools]\n{KEY} = 'bin/{exe.name}'\n")
    monkeypatch.chdir(isolated / "path")  # A cwd-relative resolution would miss
    assert _same(find_tool(KEY), exe)


def test_path_fallback_when_unconfigured(isolated):
    exe = _exe(isolated / "path")
    _write_toml(isolated, "[tools]\nsome_other_tool = 'elsewhere.exe'\n")
    assert _same(find_tool(KEY), exe)


def test_path_fallback_searches_exe_name(isolated):
    exe = _exe(isolated / "path", "ff_binary_name")
    assert _same(find_tool(KEY, "ff_binary_name"), exe)
# endregion


# region Step 2
@pytest.mark.parametrize("source", ["env", "toml"])
def test_configured_but_missing_path_does_not_fall_back_to_path(isolated, monkeypatch, source):
    _exe(isolated / "path")
    missing = isolated / "gone.exe"
    if source == "env":
        monkeypatch.setenv(ENV_VAR, str(missing))
    else:
        _write_toml(isolated, f"[tools]\n{KEY} = '{missing}'\n")
    assert find_tool(KEY) is None


@pytest.mark.parametrize("body", ["[tools\n", "tools = 'not a table'\n"])
def test_unreadable_toml_returns_none(isolated, body):
    _exe(isolated / "path")
    _write_toml(isolated, body)
    assert find_tool(KEY) is None


def test_non_string_entry_returns_none(isolated):
    _exe(isolated / "path")
    _write_toml(isolated, f"[tools]\n{KEY} = 3\n")
    assert find_tool(KEY) is None


def test_unconfigured_and_not_on_path_returns_none(isolated):
    assert find_tool(KEY) is None
# endregion


# region Step 3
def test_tools_toml_sits_at_repo_root_beside_example():
    assert REPO_TOOLS_TOML.name == "tools.toml"
    assert (REPO_TOOLS_TOML.parent / "pytest.ini").is_file()
    assert (REPO_TOOLS_TOML.parent / "tools.example.toml").is_file()
# endregion
