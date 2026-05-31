from pydantic import BaseModel, Field
from enum import IntEnum
from typing import Optional, Union
from pathlib import Path

from ...schemas import Airfoil, Freestream

class SU2_ConvergenceFlag(IntEnum):
    """
    Strict convergence flag, to identify convergence of what the orchestrator gave, additional to SU2's internal checks
    """
    TEMP = -10 # Placeholder flag within the pipeline
    UNKNOWN = -9 # For flags that are completely unknown/somehow avoided all the flagging checks
    TIMEOUT = -2 # Self explanatory
    FATAL = -1 # A fatal error occured, e.g. missing file, no solution given at all
    DIVERGED = 0 # A complete/partial divergence of the solution
    OSCILLATORY = 1 # Limit cycle or oscillatory behavior of the solution, no convergence but not diverged either
    STAGNATED = 2 # Solver is trapped in linear stiffness, and some grads are close to being noise
    ITER_LIMITED = 3 # Hit the iteration limit without convergence
    CONVERGED = 4 # Perfectly normal exit by SU2, with conv proven by SU2's internal checks

class SU2_SolutionStrategy(IntEnum):
    """
    Strict flag for solution strategy being used
    """
    COLD = 0
    WARM_EULER = 1
    MACH_SEQ = 2

class SU2_SolverConfig(BaseModel):
    """
    Contract for the solver configuration specifics to be used in config building, most values included here have their own .get 
    implementations within the pipeline to default to a certain value, since user cannot know beforehand which solver strategy
    will be chosen.
    """
    # Markers
    marker_farfield: str = Field("MARKER_FARFIELD", description="Marker returned by GMSH to identify the farfield")
    marker_airfoil: str = Field("MARKER_AIRFOIL", description="Marker returned by GMSH to identify the airfoil")
    # CFL related
    cfl_number: float = Field(0.1, description="The CFL number to be used for the run, if applicable to the strategy chosen")
    cfl_adapt_1: float = Field(0.1, description="The first CFL number to be used for the adaptive CFL strategy, if applicable to the strategy chosen")
    cfl_adapt_2: float = Field(2.0, description="The second CFL number to be used for the adaptive CFL strategy, if applicable to the strategy chosen")
    cfl_adapt_3: float = Field(10.0, description="The third CFL number to be used for the adaptive CFL strategy, if applicable to the strategy chosen")
    # Convergence related
    conv_residual_minval: float = Field(-8, description="The minimum value for the residuals in log10 scale to be considered converged, if applicable to the strategy chosen")
    cauchy_eps: float = Field(1e-5, description="The epsilon value to be used for the Cauchy convergence criteria, if applicable to the strategy chosen")
    cauchy_elems: float = Field(100, description="The number of elements to be used for the Cauchy convergence criteria, if applicable to the strategy chosen")
    max_iterations: int = Field(2500, gt=0, description="The maximum number of iterations to run the solver for, if reached without convergence, will be flagged as not converged")
    # Other
    timeout_sec: int = Field(300, gt=0, description="The maximum time to wait for the SU2 run to finish before terminating and flagging as timeout")

class SU2_In(BaseModel):
    """
    Contract for the input given to SU2
    """
    # Uniform
    airfoil: Airfoil = Field(..., description="Airfoil geometry specifications")
    freestream: Freestream = Field(..., description="Freestream flow specifications")
    manifest_path: Union[str, Path] = Field(..., description="The exact path for the manifest db being used to keep track of sims")
    # Specific to case
    working_dir: str = Field(..., description="Directory for where the input script and results will be written")
    solver_cfg: SU2_SolverConfig = Field(..., description="Specifics for the solver to build the .cfg files")
    sim_id: int = Field(..., description="Specific id for the exact simulation")
    config_path_list: Optional[list[str]] = Field(None, description="List of config paths to run SU2 with, if None, will be generated based on the route chosen and provided configs")

class SU2_Out(BaseModel):
    """
    Contract for the output given by the SU2 runner
    """
    # Uniform
    airfoil: Airfoil = Field(..., description="Airfoil geometry specifications")
    freestream: Freestream = Field(..., description="Freestream flow specifications")
    manifest_path: Union[str, Path] = Field(..., description="The exact path for the manifest db being used to keep track of sims")
    # Specific to case
    working_dir: str = Field(..., description="Directory for where the input script and results will be written")
    solver_cfg: SU2_SolverConfig = Field(..., description="Specifics for the solver to build the .cfg files")
    sim_id: int = Field(..., description="Specific id for the exact simulation")
    # Results
    config_path_list: list[str] = Field(..., description="Exact path of the configs, for the strategy chosen")
    strategy: SU2_SolutionStrategy = Field(..., description="Solution strategy used")
    convergence: SU2_ConvergenceFlag = Field(..., description="Resultant convergence flag after running")
    compute_time: float = Field(..., description="Exact total runtime when running SU2")

