"""
Locates the external executables (meshers, solvers) FlowField drives. None of them ship with it:
each is installed by the user and pointed at from the repo-root `tools.toml` (gitignored, see
`tools.example.toml`).

Lookup order per tool key: env var `FLOWFIELD_TOOL_<KEY>`, then `tools.toml`'s `[tools]` table,
then PATH. Never raises: a tool that cannot be found returns None, which each runner reports as its
own not-found flag at run time.
"""

import logging
import os
import shutil
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)

TOOLS_TOML = Path(__file__).resolve().parents[3] / "tools.toml"
ENV_PREFIX = "FLOWFIELD_TOOL_"


def find_tool(key: str, exe_name: str | None = None) -> str | None:
    """
    Resolves the executable configured for a tool.

    Args:
        key (str): The tool's key in `tools.toml`, e.g. "c2d", "su2_cfd"
        exe_name (str | None): The name to search PATH for, defaults to `key`

    Returns:
        str | None: Absolute path to the executable, or None if not found. A path set by the env
            var or `tools.toml` is authoritative: if it does not exist, None, with no PATH fallback.
    """
    env_var = ENV_PREFIX + key.upper()
    env_val = os.environ.get(env_var)
    if env_val:
        return _existing(env_val, source=env_var)

    try:
        tools = _read_tools_table()
    except (OSError, ValueError) as e:  # A broken config fails closed instead of falling to PATH
        logger.error("Could not read %s: %s", TOOLS_TOML, e)
        return None

    configured = tools.get(key)
    if configured is not None:
        if not isinstance(configured, str):
            logger.error("%s: tools.%s must be a path string, got %r", TOOLS_TOML, key, configured)
            return None
        path = Path(configured)
        if not path.is_absolute():
            path = TOOLS_TOML.parent / path
        return _existing(path, source=f"{TOOLS_TOML} [tools.{key}]")

    found = shutil.which(exe_name or key)
    return os.path.abspath(found) if found else None


def _read_tools_table() -> dict:
    """The `[tools]` table of `tools.toml`, empty if the file does not exist."""
    if not TOOLS_TOML.is_file():
        return {}
    with open(TOOLS_TOML, "rb") as f:
        tools = tomllib.load(f).get("tools", {})
    if not isinstance(tools, dict):
        raise ValueError("[tools] must be a table")
    return tools


def _existing(path, source: str) -> str | None:
    """Absolute `path` if it is a file, else None with a warning naming where it was set."""
    if os.path.isfile(path):
        return os.path.abspath(path)
    logger.warning("%s points at %s, which does not exist", source, path)
    return None
