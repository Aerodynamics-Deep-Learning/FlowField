"""
What a batch hands each of its runs, and the error that stops it, kept apart from the executor so that a slot's
worker can take them without loading the meshing stack (torch, the meshers, the sampler).

    RunContext       what every run of a batch shares: settings, ids, provenance, the work root
    StudySetupError  the study's setup is broken (no c2d, a bad input), not the config
"""

import socket
import uuid
from dataclasses import dataclass
from typing import Optional

from src.datagen.meshing.optimizer.manifest.store import Optim_git_state


class StudySetupError(RuntimeError):
    """The study cannot mesh anything as set up (no c2d, an input the gates refuse): abort the ladder."""


@dataclass(frozen=True)
class RunContext:
    """What every run of a batch shares. Picklable, so a pool's workers can each take one."""
    settings: dict
    batch_id: str
    host: str
    git_commit: str
    git_dirty: Optional[int]
    config_schema_fp: str
    work_root: str

    @classmethod
    def create(cls, settings: dict, work_root, batch_id: Optional[str] = None) -> "RunContext":
        """
        A batch's context, with its host, git state and config fingerprint taken now.

        Args:
            settings (dict): `settings.Optim_build_settings()`'s row
            work_root (str | Path): The folder each run meshes in a subfolder of
            batch_id (str | None): The batch's id, a new uuid4 if None
        """
        from src.datagen.meshing.optimizer.common.settings import Optim_config_schema_fp  # loads c2d's schemas

        commit, dirty = Optim_git_state()
        return cls(settings=settings, batch_id=batch_id or str(uuid.uuid4()), host=socket.gethostname(),
                   git_commit=commit, git_dirty=dirty, config_schema_fp=Optim_config_schema_fp(),
                   work_root=str(work_root))
