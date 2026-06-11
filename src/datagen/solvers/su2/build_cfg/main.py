"""
This is a placeholder file for later on modularization changes to the config_build related functions

1- A base config builder will buid common items for all configuration paths in a dict
2- Then the additional items will be added to the dicts in individual functions
3- A well structure output handling function will write everything in a readable format
"""

from ..schemas import SU2_SolverConfig, SU2_SolutionStrategy
from ....schemas import Freestream, Airfoil

from .cold import _SU2_BuildCfg_Cold
from .warmeuler import _SU2_BuildCfg_Warm_Euler, _SU2_BuildCfg_Warm_Restart
from .machseq import _SU2_BuildCfg_MachSeq

from .output import _write_cfg

from typing import Union

import logging
logger = logging.getLogger(__name__)

def SU2_BuildCfg(Freestream: Freestream, Airfoil: Airfoil, SolverConfig: SU2_SolverConfig, Strategy: SU2_SolutionStrategy, working_dir: str) -> list[str]:
    """
    Builds the configs given the solver config and solution strategy. first builds the base config, then adds the
    additional items calling individual functionf for each strategy.

    Args:
        Freestream (Freestream): The freestream conditions
        Airfoil (Airfoil): The airfoil geometry
        SolverConfig (SU2_SolverConfig): The solver configuration
        Strategy (SU2_SolutionStrategy): The solution strategy to build the config for
        working_dir (str): The working directory for the current solver instance

    Returns:
        list[str]: The list of config paths built for the given strategy
    """

    base_dict = _SU2_BuildCfg_Base(Freestream=Freestream, Airfoil=Airfoil, SolverConfig=SolverConfig) # Gets the base dict
    config_path_list = [] # Initializes the config path list to be returned at the end, to be filled in each strategy

    if Strategy == SU2_SolutionStrategy.COLD:
        cold_build_path = f"{working_dir}/cold_build.cfg"

        cold_dict = _SU2_BuildCfg_Cold(base_dict=base_dict, freestream=Freestream, airfoil=Airfoil, SolverConfig=SolverConfig) # Builds the cold start config from the base dict
        _write_cfg(cold_dict, cold_build_path) # Writes the config to the path
        
        config_path_list.append(cold_build_path) # Appends the path to the list to be returned
        return config_path_list

    elif Strategy == SU2_SolutionStrategy.WARM_EULER:
        warm_euler_build_path = f"{working_dir}/warm_euler_build.cfg"
        warm_restart_build_path = f"{working_dir}/warm_restart_build.cfg"
        
        euler_dict = _SU2_BuildCfg_Warm_Euler(base_dict=base_dict, freestream=Freestream, airfoil=Airfoil, SolverConfig=SolverConfig)
        restart_dict = _SU2_BuildCfg_Warm_Restart(base_dict=base_dict, freestream=Freestream, airfoil=Airfoil, SolverConfig=SolverConfig)
        _write_cfg(euler_dict, warm_euler_build_path) # Writes the euler config to the path
        _write_cfg(restart_dict, warm_restart_build_path) # Writes the restart config to the path

        config_path_list.extend([warm_euler_build_path, warm_restart_build_path]) # Appends the paths to the list to be returned
        return config_path_list
    
    elif Strategy == SU2_SolutionStrategy.MACH_SEQ:
        pass # Not implemented yet

    else:
        logger.warning(f"Identified but not implemented route: {SU2_SolutionStrategy}, defaulting to cold start")
        cold_build_path = f"{working_dir}/cold_build.cfg"

        cold_dict = _SU2_BuildCfg_Cold(base_dict=base_dict) # Builds the cold start config from the base dict
        _write_cfg(cold_dict, cold_build_path) # Writes the config to the path
        
        config_path_list.append(cold_build_path) # Appends the path to the list to be returned
        return config_path_list


def _SU2_BuildCfg_Base(freestream: Freestream, airfoil: Airfoil, SolverConfig: SU2_SolverConfig) -> dict[str, dict[str, Union[str, int, float]]]:
    """
    Gets the fundamental items for config building from freestream, airfoil, solver config, and returns a base dict
    to be modified by specific strategies.

    Args:
        freestream (Freestream): The schema for freestream inputs
        airfoil (Airfoil): The schema for airfoil inputs
        SolverConfig (SolverConfig): The schema for solver configs

    Returns:
        dict: A base dict implementing the fundamental items, with structured sub-dicts
    """

    base_dict = {
        "INPUT / OUTPUT": {
            "MESH_FILENAME": f"{airfoil.airfoil_name}_mesh.su2", # Implemented as such for default in GMSH pipeline
            "MESH_FORMAT": "SU2",
            "WRT_SOL_FREQ": SolverConfig.max_iterations, # Writes sol only once
            "WRT_CON_FREQ": 1, # Writes conv history every iteration
            "OUTPUT_FILES": "( RESTART, PARAVIEW, SURFACE_CSV, CSV )", # Outs restart file, paraview file, flowfield csv, and csv with forces history
            "TABULAR_FORMAT": "CSV",
            "CONV_FILENAME": "history",
            "RESTART_FILENAME": "flow",
            "VOLUME_FILENAME": "solution_flow",
            "SURFACE_FILENAME": "surface_flow",
            "HISTORY_OUTPUT": "( INNER_ITER, WALL_TIME, NONPHYSICAL_POINTS, RMS_RES, DRAG, LIFT, MOMENT_X, MOMENT_Y, FORCE_X, FORCE_Y )",
            "SCREEN_OUTPUT": "( INNER_ITER, WALL_TIME, NONPHYSICAL_POINTS, RMS_RES, DRAG, LIFT, MOMENT_X, MOMENT_Y, FORCE_X, FORCE_Y )",
        },
        "REFERENCE VALUES": {
            "REF_AREA": airfoil.chord,
            "REF_LENGTH": airfoil.chord,
            "REF_ORIGIN_MOM_X": airfoil.chord * 0.25,
            "REF_ORIGIN_MOM_Y": 0.0,
            "REF_ORIGIN_MOM_Z": 0.0,
            "AERO_COEFF_FORMAT": "WIND",
        },
        "FLUID MODEL": {
            "FLUID_MODEL": "STANDARD_AIR",
        },
        "FREESTREAM PARAMETERS": {
            "MACH_NUMBER": freestream.mach,
            "AOA": freestream.alpha,
            "FREESTREAM_TEMPERATURE": freestream.temp,
        },
        "BOUNDARY MARKERS": {
            "MARKER_FAR": f"( {SolverConfig.marker_farfield} )",
            "MARKER_PLOTTING": f"( {SolverConfig.marker_airfoil} )",
            "MARKER_MONITORING": f"( {SolverConfig.marker_airfoil} )",
        },
        "SPATIAL DISCRETIZATION": {
            "NUM_METHOD_GRAD": "WEIGHTED_LEAST_SQUARES",
            "SPATIAL_ORDER_FLOW": "2ND_ORDER_LIMITER",
            "SLOPE_LIMITER_FLOW": "VENKATAKRISHNAN",
            "VENKAT_LIMITER_COEFF": 0.05,
        },
        "MULTIGRID PARAMETERS": {
            # MGLEVEL is defined later
            "MGCYCLE": "V_CYCLE",
            "MG_PRE_SMOOTH": "( 1, 2, 3, 3 )",
            "MG_POST_SMOOTH": "( 0, 0, 0, 0 )",
            "MG_CORRECTION_SMOOTH": "( 0, 0, 0, 0 )",
            "MG_DAMP_RESTRICTION": 0.75,
            "MG_DAMP_PROLONGATION": 0.75,
        },
        "CONVERGENCE PARAMETERS": {
            "CONV_CRITERIA": "RESIDUAL, CAUCHY",
            "CONV_RESIDUAL_MINVAL": SolverConfig.conv_residual_minval,
            "CONV_FIELD": "RMS_DENSITY",
            "CAUCHY_ELEMS": SolverConfig.cauchy_elems,
            "CAUCHY_EPS": SolverConfig.cauchy_eps,
            "CAUCHY_FUNC_FLOW": "DRAG",
            "INNER_ITER": SolverConfig.max_iterations,
        },
    }

    return base_dict