from .schemas import SU2_ConvergenceFlag, SU2_SolutionStrategy

from pathlib import Path
from typing import Union

def SU2_UpdateManifest(manifest_path: Union[str, Path], sim_id: int, convergence: SU2_ConvergenceFlag, strategy: SU2_SolutionStrategy, compute_time: float):
    return None
