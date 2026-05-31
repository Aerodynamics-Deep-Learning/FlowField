import time

from .schemas import SU2_In, SU2_Out
from .utils import SU2_StrategyIdentify, SU2_KeepDeleteVTU
from .build_cfg.main import SU2_BuildCfg
from .runcfd import SU2_RunCFD
from .convergence import SU2_CheckConvergence
from .data_handling.handler import SU2_ComprWipeData
from .manifest import SU2_UpdateManifest

def SU2_Runner(su2_in: SU2_In) -> SU2_Out:
    """
    Builds, runs, and handles i/o of the SU2 solver procedure.

    Args:
        su2_in (SU2_In): The standard input agreement for SU2

    Returns:
        SU2_Out: The standard SU2 output agreement
    """
    # Identifies the strategy first
    strategy = SU2_StrategyIdentify(su2_in.freestream)

    # Using the identified solver strategy, builds the config files needed for the run
    config_path_list = SU2_BuildCfg(
        Freestream=su2_in.freestream,
        Airfoil=su2_in.airfoil,
        SolverConfig=su2_in.solver_cfg, 
        strategy=strategy, 
        working_dir=SU2_In.working_dir
    )

    start_time = time.perf_counter()
    # Using the built configs, runs the stuff with the solver SU2_CFD, gets the stdout to do later conv checking
    convergence_flag_temp, stdout, stderr = SU2_RunCFD(
        config_path_list=config_path_list,
        timeout_sec=su2_in.solver_cfg.timeout_sec,
        strategy=strategy
    )
    compute_time = time.perf_counter() - start_time

    # Using the stdout and history.csv, determine the convergence flag
    convergence_flag = SU2_CheckConvergence(
        convergence_flag_temp=convergence_flag_temp,
        stdout=stdout,
        stderr=stderr,
        working_dir=su2_in.working_dir
    )

    # Depending on the convergence flag, either completely wipe out data or compress it
    clean_paraview = SU2_KeepDeleteVTU() # Specifically for the vtu, decides on keeping or deleting it
    SU2_ComprWipeData(
        convergence=convergence_flag,
        strategy=strategy,
        working_dir=su2_in.working_dir,
        clean_paraview=clean_paraview
    )

    # Log the results to the manifest
    SU2_UpdateManifest(
        manifest_path=su2_in.manifest_path,
        sim_id=su2_in.sim_id, 
        convergence=convergence_flag,
        strategy=strategy,
        compute_time=compute_time
    )

    return SU2_Out(
        airfoil=su2_in.airfoil,
        freestream=su2_in.freestream,
        manifest_path=su2_in.manifest_path,
        working_dir=su2_in.working_dir,
        solver_cfg=su2_in.solver_cfg,
        sim_id=su2_in.sim_id,
        config_path_list=config_path_list,
        strategy=strategy,
        convergence=convergence_flag,
        compute_time=compute_time
    )

